import lightning as L
import math
from matplotlib import pyplot as plt
from monai.losses.dice import DiceLoss
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.nvae.decoder import Decoder
from arch.nvae.distribution import Normal
from arch.nvae.encoder import Encoder
from const import ACDC, FRDS_MODEL_PATH
from utils.eval import frds, get_samples_and_reconstructions
from utils.utils import clamp, discretise, show_samples

class NVAE(L.LightningModule):
    """
    Nouveau Variational Autoencoder (NVAE) is a deep hierarchical VAE that
    achieves SOTA in image generation tasks among non-autoregressive models.
    This class adapts from the framework proposed in [1], primarily based on
    details described in the paper and the original codebase
    (https://github.com/NVlabs/NVAE).
    
    Note: The paper refers to each latent layer in the VAE as a "latent scale".
    We will use the word "layer", as it is a more common term in hierarchical
    model literature (e.g. Ladder VAE).
    
    Indexing of n latent layers is as follows: 0 corresponds to the shallowest
    layer and n-1 corresponds to the topmost layer.
    
    [1]: Vahdat A, Kautz J. NVAE: A deep hierarchical variational autoencoder.
    Advances in neural information processing systems. 2020;33:19667-79.
    """
    
    def __init__(
        self,
        in_channels: int=4,
        initial_channels: int=64,
        z_channels: int=20,
        num_latent_layers: int=3,
        # Topmost layer has fewest groups (i.e. 1)
        # Shallowest layer has most groups (i.e. 4)
        num_groups_per_layer: list[int]=[4, 2, 1],
        initial_downsample_factor: int=8,
        max_epochs: int=50,
        beta_per_layer: list[float]=[1.0, 1.0, 1.0],
        kl_warmup_steps: int=500,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        self.img_width = ACDC.WIDTH
        
        # Table 6: # initial channels in enc. (NVAE paper)
        self.stem = nn.Conv2d(
            self.hparams.in_channels,
            self.hparams.initial_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        
        self.encoder = Encoder(
            num_latent_layers=self.hparams.num_latent_layers,
            num_groups_per_layer=self.hparams.num_groups_per_layer,
            initial_channels=self.hparams.initial_channels,
            z_channels=self.hparams.z_channels,
            initial_downsample_factor=self.hparams.initial_downsample_factor,
        )
        
        top_latent_dim = self._get_latent_dim(self.hparams.num_latent_layers - 1)

        self.decoder = Decoder(
            num_latent_layers=self.hparams.num_latent_layers,
            num_groups_per_layer=self.hparams.num_groups_per_layer[::-1],
            initial_channels=self.hparams.initial_downsample_factor * self.hparams.initial_channels * (2 ** (self.hparams.num_latent_layers - 1)),
            top_latent_shape=(top_latent_dim, top_latent_dim),
            z_channels=self.hparams.z_channels,
            final_upsample_factor=self.hparams.initial_downsample_factor,
        )
        
        # This is the opposite of the stem
        self.conditional_coder = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(
                self.hparams.initial_channels,
                self.hparams.in_channels,
                kernel_size=3,
                padding=1,
            ),
        )

    def configure_optimizers(self):
        optimiser = torch.optim.Adamax(
            self.parameters(),
            lr=1e-2,
            eps=1e-3,
            weight_decay=3e-4,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimiser,
            T_max=self.hparams.max_epochs,
            eta_min=1e-4,
        )
        
        return [optimiser], [lr_scheduler]
    
    def _get_latent_dim(self, layer: int) -> int:
        # Layer 0 is the shallowest layer
        # Layer @(num_latent_layers - 1) is the deepest (topmost) layer
        return (self.img_width // self.hparams.initial_downsample_factor) // (2 ** layer)

    def _get_layer_index(self, latent_dim: int) -> int:
        # Inverse of _get_latent_dim
        idx = math.log2((self.img_width // self.hparams.initial_downsample_factor) // latent_dim)
        assert idx.is_integer()
        return int(idx)
    
    def _compute_gamma(self) -> float:
        """
        Perform KL annealing with a linear schedule: 0 for the first 10
        steps, then warm-up to 1 over the next @kl_warmup_steps steps.
        
        The coefficient is clamped at >= 0.0001 for stability.
        
        Returns:
            gamma (float): KL coefficient for the current step.
        """
        constant_steps = 10
        gamma = (self.global_step - constant_steps) / self.hparams.kl_warmup_steps
        return clamp(gamma, 0.0001, 1.0)
    
    def _balance_kl(self, kl_divs, gamma=1.0, balance=True):
        """
        Perform the balancing mechanism on KL divergence terms as described in
        the appendix of the NVAE paper. This consists of (1) scale all KL values
        by the coefficient @gamma and (2) rescale KL values over all groups by
        their magnitude.
        
        The balancing mechanism is applied only during the warm-up period. After
        the warm-up period, gamma is 1 and the method simply returns the average
        over all groups.
        
        noqa: This method should only be called during training.
        
        Args:
            kl_divs (torch.Tensor): KL per group. bxg tensor for batch size b
                and g groups.
            gamma (float): KL coefficient for current step. gamma of 1 indicates
                warm-up is complete and balancing mechanism is not applied.
                Default: 1.0.
            balance (bool): Whether to apply the balancing mechanism. This
                should be on at train time and off at test time. Default: True.
        
        Returns:
            balanced_kl_divs (torch.Tensor): Balanced KL per group, averaged
                over the batch. g-dim tensor for g groups.
        """
        assert gamma <= 1.0
        
        if balance and gamma < 1.0:
            # Set balancing coefficient proportional to the KL term for each
            # group
            gamma_i = torch.abs(kl_divs)
            gamma_i = torch.mean(gamma_i, dim=0, keepdim=True) + 0.01
            total_kl = torch.sum(gamma_i)

            gamma_i = gamma_i * total_kl
            # Rescale gamma_i to sum to 1
            gamma_i = gamma_i / torch.mean(gamma_i, dim=1, keepdim=True)
            kl_divs = kl_divs * gamma_i.detach()

        return gamma * torch.mean(kl_divs, dim=0)
    
    def _weight_kl(self, balanced_kl_divs: torch.Tensor, kl_layers: torch.Tensor):
        """
        For each layer, weight the KL divergence by the corresponding beta.
        Equivalent to beta-VAE but applied to each layer.
        
        Args:
            balanced_kl_divs (torch.Tensor): KL per group after balancing,
                averaged over the batch. g-dim tensor for g groups.
            kl_layers (torch.Tensor): Layer index for each KL term in
                balanced_kl_divs.
        
        Returns:
            weighted_kls (torch.Tensor): Weighted KL per layer. n-dim tensor for
                n layers.
        """
        weighted_kls = torch.empty(self.hparams.num_latent_layers)
        
        for layer_idx in range(self.hparams.num_latent_layers):
            # Sum KL within each layer
            balanced_kl_div_layer = balanced_kl_divs[kl_layers == layer_idx].sum()
            # Weight
            weighted_kls[layer_idx] = self.hparams.beta_per_layer[layer_idx] * balanced_kl_div_layer
        
        return weighted_kls
    
    def _kl_divergence(
        self,
        qs: list[Normal],
        ps: list[Normal],
        log_qs: list[torch.Tensor],
        log_ps: list[torch.Tensor],
        log_components: bool=True,
    ) -> torch.Tensor:
        """
        Compute the weighted and balanced KL divergence between the approximate
        posterior and prior, summed over all latent layers. The KL divergence
        of each layer is the sum over all groups within the layer.
        """
        # log_p, log_q and kl_diag are for metrics purposes
        
        # For each group, compute KL of batch
        # For n groups, this is a n-list of b-dim tensors (b=batch size)
        kl_divs = []
        # Logging
        kl_diag = []
        # Record which layer each KL corresponds to (for logging marginal KL)
        kl_layers = []
        
        log_p: float = 0
        log_q: float = 0
        
        for q, p, log_q_conv, log_p_conv in zip(qs, ps, log_qs, log_ps):
            assert q.mu.shape == p.mu.shape
            
            _, z_channels, width, height = q.mu.shape
            assert width == height
            assert z_channels == self.hparams.z_channels
            kl_layers.append(self._get_layer_index(width))
            
            # TODO Change this for normalising flow
            kl_per_var = q.kl(p)

            kl_div = torch.sum(kl_per_var, dim=[1, 2, 3])

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_divs.append(kl_div)
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])
            
        kl_layers = torch.tensor(kl_layers)
        
        gamma = self._compute_gamma()
        
        # Stack list to bxn tensor
        kl_divs = torch.stack(kl_divs, dim=1)
        # Average KL over batch: n-dim tensor
        kl_divs_batch_avg = torch.mean(kl_divs, dim=0)
        
        balanced_kl_divs_batch_avg = self._balance_kl(kl_divs, gamma)
        weighted_kls = self._weight_kl(balanced_kl_divs_batch_avg, kl_layers)
        
        # Compute and log KL per layer
        if log_components:
            for layer_idx in range(self.hparams.num_latent_layers):
                # Log the original (unbalanced) KL per layer which indicates
                # amount of information encoded at each layer
                kl_div_layer = kl_divs_batch_avg[kl_layers == layer_idx].mean()
                self.log(f"kl_div_{layer_idx}", kl_div_layer)
        
        return weighted_kls.sum()

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss using cross-entropy.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input.
        
        Returns:
            recon_loss (torch.Tensor): Reconstruction loss.
        """
        batch_size = x.size(0)
        return F.cross_entropy(x_hat, x, reduction="sum") / batch_size
    
    def loss(
        self,
        x: torch.Tensor,
        x_hat: torch.Tensor,
        qs: list[Normal],
        ps: list[Normal],
        log_qs: list[torch.Tensor],
        log_ps: list[torch.Tensor],
        log_components: bool=True,
    ) -> torch.Tensor:
        """
        Compute the NVAE loss: sum of reconstruction loss and beta-weighted
        balanced KL divergence regulariser term.
        """
        recon_loss = self.reconstruction_loss(x, x_hat)
        balanced_kl_div = self._kl_divergence(qs, ps, log_qs, log_ps, log_components)
        
        print(f"Reconstruction loss: {recon_loss}")
        print(f"Weighted KL divergence: {balanced_kl_div}")
        
        if log_components:
            self.log("recon_loss", recon_loss)
            self.log("kl_div", balanced_kl_div)
        
        return recon_loss + balanced_kl_div
    
    def forward(
        self,
        feats: torch.Tensor,
        test: bool=False,
        num_shared_layers: int=-1,
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass through the NVAE encoder and decoder.
        
        Args:
            feats (torch.Tensor): One-hot encoded input segmentations.
            test (bool): Indicates whether test mode is enabled (compared to
                train or validation mode). Default: False.
            num_shared_layers (int): Number of latent layers shared with the
                decoder from the topmost layer. If -1, all layers are shared.
                See docstring of Decoder forward pass for details. Default: -1.
        
        Returns:
            feats_hat, qs, ps, log_qs, log_ps
            feats_hat (torch.Tensor): Logits of reconstruction of input.
            qs (list[Normal]): Approximate posterior distributions.
            ps (list[Normal]): Prior distributions.
            log_qs (list[torch.Tensor]): Log probabilities of samples drawn from
                the residual distribution with respect to the approximate
                posterior.
            log_ps (list[torch.Tensor]): Log probabilities of samples drawn from
                the residual distribution with respect to the prior.
        """
        # Convert one-hot encoded inputs [0, 1] to [-1, 1] for train stability
        x = self.stem(2 * feats - 1.0)
        
        # Pass through encoder
        x, xs, enc_combiner_cells = self.encoder(x)
        
        # Reverse buffers and modules for decoder
        xs = xs[::-1]
        enc_combiner_cells = enc_combiner_cells[::-1]
        enc_samplers = self.encoder.samplers[::-1]
        
        # Pass through decoder
        x_hat, qs, ps, log_qs, log_ps = self.decoder(
            x,
            xs,
            enc_combiner_cells,
            enc_samplers,
            test=test,
            num_shared_layers=num_shared_layers,
        )
        
        # Compute logits
        feats_hat: torch.Tensor = self.conditional_coder(x_hat)

        assert feats.shape == feats_hat.shape
        
        return feats_hat, qs, ps, log_qs, log_ps
        
    def training_step(self, feats: torch.Tensor) -> torch.Tensor:
        feats_hat, qs, ps, log_qs, log_ps = self(feats)
        
        # Compute loss
        loss = self.loss(feats, feats_hat, qs, ps, log_qs, log_ps)
        self.log("train_loss", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, feats: torch.Tensor) -> torch.Tensor:
        feats_hat, qs, ps, log_qs, log_ps = self(feats)
        
        # Compute loss
        loss = self.loss(feats, feats_hat, qs, ps, log_qs, log_ps)
        self.log("val_loss", loss)
        
        print(f"Val loss: {loss}")
        
    def test_step(self, feats: torch.Tensor, batch_idx: int) -> torch.Tensor:
        assert batch_idx == 0, "Only 1 batch allowed"
        self.log_reconstruction_metrics(feats)
        self.log_generation_metrics(feats)

    def log_reconstruction_metrics(self, x: torch.Tensor):
        """
        Log reconstruction metrics to TensorBoard. This includes average
        reconstruction loss and Dice score across the batch. Log visualisations
        of samples and their reconstructions.
        
        Args:
            x (torch.Tensor): Batch of input samples.
        """
        x_hat_logits, _, _, _, _ = self(x, test=True)
        
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        self.log("test_recon_loss", recon_loss)
        
        # Compute Dice score
        x_hat = torch.softmax(x_hat_logits, dim=1)
        x_hat_onehot = discretise(x_hat)
        dl = DiceLoss(reduction="mean", include_background=False)
        dice_score = 1 - dl(input=x_hat_onehot, target=x)
        self.log("dice_score", dice_score)

        # Visualise samples and reconstructions
        samples_and_reconstructions = get_samples_and_reconstructions(x[:20], x_hat_logits[:20])
        show_samples(samples_and_reconstructions, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())

    def log_generation_metrics(self, feats: torch.Tensor):
        """
        Log generation metrics to TensorBoard. This includes the Frechet Resnet
        Distance with SimCLR (FRDS) metric across the batch. Log visualisations
        of sample generations.
        
        Args:
            x (torch.Tensor): Batch of input samples.
        """
        num_samples, _, _, _ = feats.shape
        
        # Generate probabilistic segmentation maps
        x_fake = self.decoder.generate(num_samples, device=feats.device)
        feats_fake = self.conditional_coder(x_fake)

        # Discretise probabilistic map then view generations
        generations = torch.argmax(feats_fake[:40], dim=1).unsqueeze(1)
        show_samples(generations, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/generations", plt.gcf())
        
        frds_value = frds(
            feats,
            discretise(feats_fake),
            resnet_path=FRDS_MODEL_PATH,
            device=self.device,
        )

        self.log("frds", frds_value)
