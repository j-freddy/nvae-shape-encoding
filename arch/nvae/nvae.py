from collections import defaultdict
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
from utils.const import ACDC, CARDIAC_WIDTH, FRDS_MODEL_PATH, MASK_CLASSES
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.eval import compute_dice_score, compute_frds, get_samples_and_reconstructions_pixel_diff
from utils.utils import clamp, discretise, show_samples

class NVAE(L.LightningModule):
    """
    Nouveau Variational Autoencoder (NVAE) is a deep hierarchical VAE that
    achieves SOTA in image generation tasks among non-autoregressive models.
    This class adapts from the framework proposed in [1], primarily based on
    details described in the paper and the original codebase
    (https://github.com/NVlabs/NVAE).
    
    Indexing of n layers is as follows: 0 corresponds to the shallowest layer
    and n-1 corresponds to the topmost layer.
    
    We have extended the framework such that some layers do not encode shared
    latent variables. Such a layer only consists of residual cells and does not
    have a combiner nor sampler. For ease of reference, we introduce 2 indices:
    (1) layer index, and (2) latent index.
    
    Example:
        Layer 0: Shared (latent index 0)    (64x64 latent space)
        Layer 1: Not shared                 (32x32 feature space)
        Layer 2: Shared (latent index 1)    (16x16 latent space)
        Layer 3: Not shared                 (8x8 feature space)
        Layer 4: Shared (latent index 2)    (4x4 latent space)
    
    The NVAE paper refers to each latent layer in the VAE as a "latent scale".
    We will use the term "latent layer", as it is a more common term in
    hierarchical model literature (e.g. Ladder VAE). We will also use the term
    "shared layer" interchangeably with "latent layer", since only shared layers
    have an explicit latent space; the only exception is if a latent layer is
    not shared when performing an ablation study.
    
    [1]: Vahdat A, Kautz J. NVAE: A deep hierarchical variational autoencoder.
    Advances in neural information processing systems. 2020;33:19667-79.
    """
    
    def __init__(
        self,
        in_channels: int=4,
        initial_channels: int=64,
        min_channels: int=0,
        z_channels: int=20,
        num_groups_per_layer: list[int]=[4, 2, 1],
        is_layer_shared: list[bool]=[True, True, True],
        initial_downsample_factor: int=8,
        max_epochs: int=50,
        beta_per_layer: list[float]=[1.0, 1.0, 1.0],
        kl_warmup_steps: int=500,
        use_sr: bool=False,
    ):
        """
        Create an instance of the NVAE model. All constructor arguments are
        saved in the checkpoint as hyperparameters.
        
        Args:
            in_channels (int): Number of input channels. Corresponds to number
                of classes in segmentation mask. Default: 4.
            initial_channels (int): Number of channels @in_channels gets
                projected to. This is done immediately in the stem, then
                reverted at the very end of the pass in the conditional coder.
                Default: 64.
            min_channels (int): Minimum number of channels anywhere. By default,
                the number of channels is doubled at each deeper layer. For
                example, if @initial_channels is 16 and there are 3 layers, it
                goes 16 -> 32 -> 64 -> 128. But if @initial_channels is 4 and
                @min_channels is 16, it goes 16 -> 16 -> 16 -> 32 instead of 4
                -> 8 -> 16 -> 32. The idea is to not have an initial bottleneck,
                which may be caused by limited capacity from the small number of
                channels. Default: 16.
            z_channels (int): Number of channels in the latent space. For
                example, if the latent space is 4x4 for a particular layer, the
                latent variable is @z_channelsx4x4. Default: 20.
            num_groups_per_layer (list[int]): Number of groups in each layer.
                This is traversed sequentially, so more groups allow for more
                corrections and refinement within a single layer. Order: from
                shallowest to topmost layer. Default: [4, 2, 1].
            is_layer_shared (list[bool]): Whether the latent space in each layer
                is shared with the decoder. If a layer is not shared, it only
                consists of residual cells and does not have a combiner nor
                sampler. Order corresponds to @num_groups_per_layer. Default:
                [True, True, True].
            initial_downsample_factor (int): Downsample factor in the
                preprocess stage of the encoder tower. This corresponds to the
                upsample factor in the postprocess stage of the decoder tower.
                For example, if @initial_downsample_factor is 8 and the input is
                128x128, the preprocess stage downsamples to 16x16. Default: 8.
            max_epochs (int): Maximum number of epochs for training. Default:
                50.
            beta_per_layer (list[float]): Beta coefficient for each shared
                layer. Order corresponds to the shared layers of
                @is_layer_shared. Default: [1.0, 1.0, 1.0].
            kl_warmup_steps (int): Number of steps to perform KL annealing.
                Each epoch has 214 steps. Default: 500.
            use_sr (bool): If True, use spectral regularisation. Default: False.
        """
        super().__init__()
        
        assert len(num_groups_per_layer) == len(is_layer_shared)
        assert sum(is_layer_shared) == len(beta_per_layer)
        
        self.save_hyperparameters()
        
        self.img_width = CARDIAC_WIDTH
        self.num_layers = len(self.hparams.num_groups_per_layer)
        self.num_latent_layers = len(self.hparams.beta_per_layer)
        
        self.layer_idx_to_latent_idx = self._get_layer_idx_to_latent_idx_map()
        
        # Table 6: # initial channels in enc. (NVAE paper)
        self.stem = nn.Conv2d(
            self.hparams.in_channels,
            max(self.hparams.initial_channels, self.hparams.min_channels),
            kernel_size=3,
            padding=1,
            bias=True,
        )
        
        self.encoder = Encoder(
            num_groups_per_layer=self.hparams.num_groups_per_layer,
            is_layer_shared=self.hparams.is_layer_shared,
            initial_channels=self.hparams.initial_channels,
            min_channels=self.hparams.min_channels,
            z_channels=self.hparams.z_channels,
            initial_downsample_factor=self.hparams.initial_downsample_factor,
        )
        
        top_latent_dim = self._get_latent_dim(self.num_layers - 1)

        self.decoder = Decoder(
            num_groups_per_layer=self.hparams.num_groups_per_layer[::-1],
            is_layer_shared=self.hparams.is_layer_shared[::-1],
            initial_channels=self.hparams.initial_downsample_factor * self.hparams.initial_channels * (2 ** (self.num_layers - 1)),
            min_channels=self.hparams.min_channels,
            top_latent_shape=(top_latent_dim, top_latent_dim),
            z_channels=self.hparams.z_channels,
            final_upsample_factor=self.hparams.initial_downsample_factor,
        )
        
        # This is the opposite of the stem
        self.conditional_coder = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(
                max(self.hparams.initial_channels, self.hparams.min_channels),
                self.hparams.in_channels,
                kernel_size=3,
                padding=1,
            ),
        )
                
        # Convolutional layers are used for spectral regularisation
        self.conv_layers = self._get_conv_layers()
        
        # To keep track of test set and generated samples during test time, to
        # compute FRDS
        self.feats_buffer: list[torch.Tensor] = []
        self.feats_fake_buffer: list[torch.Tensor] = []
    
    def _get_conv_layers(self) -> list[nn.Conv2d]:
        conv_layers = []
        
        for _, layer in self.named_modules():
            if isinstance(layer, nn.Conv2d):
                conv_layers.append(layer)
        
        return conv_layers

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
    
    def _get_layer_idx_to_latent_idx_map(self) -> dict[int, int]:
        latent_idx_to_layer_idx = [
            i for i, is_latent in enumerate(self.hparams.is_layer_shared)
            if is_latent
        ]
        
        mapping = {
            layer_idx: latent_idx for latent_idx, layer_idx in enumerate(latent_idx_to_layer_idx)
        }
        
        return mapping
    
    def _get_latent_dim(self, layer: int) -> int:
        # Layer 0 is the shallowest layer
        # Layer @(num_layers - 1) is the deepest (topmost) layer
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
    
    def _weight_kl(self, balanced_kl_divs: torch.Tensor, kl_latent_layers: torch.Tensor):
        """
        For each shared layer, weight the KL divergence by the corresponding
        beta. Equivalent to beta-VAE but applied to each shared layer.
        
        Args:
            balanced_kl_divs (torch.Tensor): KL per group after balancing,
                averaged over the batch. g-dim tensor for g groups.
            kl_latent_layers (torch.Tensor): Latent layer index for each KL term
                in balanced_kl_divs.
        
        Returns:
            weighted_kls (torch.Tensor): Weighted KL per layer. n-dim tensor for
                n layers.
        """
        weighted_kls = torch.empty(self.num_latent_layers)
        
        for latent_idx in range(self.num_latent_layers):
            # Sum KL within each layer
            balanced_kl_div_layer = balanced_kl_divs[kl_latent_layers == latent_idx].sum()
            # Weight
            weighted_kls[latent_idx] = self.hparams.beta_per_layer[latent_idx] * balanced_kl_div_layer
        
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
        
        kl_latent_layers = [self.layer_idx_to_latent_idx[layer_idx] for layer_idx in kl_layers]
        kl_latent_layers = torch.tensor(kl_latent_layers)
        
        gamma = self._compute_gamma()
        
        # Stack list to bxn tensor
        kl_divs = torch.stack(kl_divs, dim=1)
        # Average KL over batch: n-dim tensor
        kl_divs_batch_avg = torch.mean(kl_divs, dim=0)
        
        balanced_kl_divs_batch_avg = self._balance_kl(kl_divs, gamma)
        weighted_kls = self._weight_kl(balanced_kl_divs_batch_avg, kl_latent_layers)
        
        # Compute and log KL per layer
        if log_components:
            for latent_idx in range(self.num_latent_layers):
                # Log the original (unbalanced) KL per layer which indicates
                # amount of information encoded at each layer
                kl_div_layer = kl_divs_batch_avg[kl_latent_layers == latent_idx].mean()
                self.log(f"loss/kl_div_{latent_idx}", kl_div_layer)
            
            num_groups = len(kl_divs_batch_avg)
            
            for i in range(num_groups):
                # Log the original KL per group
                # Shallowest group is 0
                # Topmost group is 6
                group_idx = num_groups - i - 1
                self.log(f"loss/kl_div_group_{group_idx}", kl_divs_batch_avg[i])
    
        return weighted_kls.sum()

    def _spectral_norm(self) -> torch.Tensor:
        # Dictionary: weight shape -> weight matrix
        # So we can later stack weight matrices of the same shape and compute in
        # parallel
        weights = defaultdict(lambda: [])

        for layer in self.conv_layers:
            weight = layer.weight
            weight_matrix = weight.view(weight.size(0), -1)
            weights[weight_matrix.shape].append(weight_matrix)

        loss = 0
        
        # U and V matrices of singular value decomposition
        sr_u = {}
        sr_v = {}

        for shape in weights.keys():
            weights[shape] = torch.stack(weights[shape], dim=0)

            with torch.no_grad():
                num_iter = 4

                if shape not in sr_u:
                    n, row, col = weights[shape].shape
                    sr_u[shape] = F.normalize(
                        torch.ones(n, row).normal_(0, 1).to(self.device),
                        dim=1,
                        eps=1e-3,
                    )
                    sr_v[shape] = F.normalize(
                        torch.ones(n, col).normal_(0, 1).to(self.device),
                        dim=1,
                        eps=1e-3,
                    )

                    # First occurance: increase number of iterations
                    num_iter = 40

                # SVD: u^T W v
                # Approximate u, v via power iteration
                for _ in range(num_iter):
                    sr_v[shape] = F.normalize(
                        torch.matmul(sr_u[shape].unsqueeze(1),weights[shape]).squeeze(1),
                        dim=1,
                        eps=1e-3,
                    )
                    sr_u[shape] = F.normalize(
                        torch.matmul(weights[shape], sr_v[shape].unsqueeze(2)).squeeze(2),
                        dim=1,
                        eps=1e-3,
                    )

            sigma = torch.matmul(
                sr_u[shape].unsqueeze(1),
                torch.matmul(weights[shape], sr_v[shape].unsqueeze(2)),
            )
            loss += torch.sum(sigma)

        return loss

    def reconstruction_loss(self, x: torch.Tensor, x_hat_logits: torch.Tensor) -> torch.Tensor:
        """
        Compute the reconstruction loss using cross-entropy.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input.
        
        Returns:
            recon_loss (torch.Tensor): Reconstruction loss.
        """
        batch_size = x.size(0)
        return F.cross_entropy(x_hat_logits, x, reduction="sum") / batch_size
    
    def loss(
        self,
        x: torch.Tensor,
        x_hat_logits: torch.Tensor,
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
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        balanced_kl_div = self._kl_divergence(qs, ps, log_qs, log_ps, log_components)
        
        print(f"Reconstruction loss: {recon_loss}")
        print(f"Weighted KL divergence: {balanced_kl_div}")
        
        if log_components:
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", balanced_kl_div)
        
        # Spectral regularisation
        if self.hparams.use_sr:
            weighted_sr_loss = 0.1 * self._spectral_norm()
            print(f"Spectral regularisation loss: {weighted_sr_loss}")
            
            if log_components:
                self.log("sr_loss", weighted_sr_loss)
            
            return recon_loss + balanced_kl_div + weighted_sr_loss
        
        return recon_loss + balanced_kl_div

    def get_latent(self, feats: torch.Tensor, test: bool=True) -> list[torch.Tensor]:
        """
        Given an input tensor, return its latent representations in each latent
        layer by passing it through the encoder-decoder.
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
        _, _, _, _, _, zs = self.decoder(
            x,
            xs,
            enc_combiner_cells,
            enc_samplers,
            test=test,
            return_latents=True,
        )

        return zs
    
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
            feats_hat_logits (torch.Tensor): Logits of reconstruction of input.
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
        x_hat_logits, qs, ps, log_qs, log_ps = self.decoder(
            x,
            xs,
            enc_combiner_cells,
            enc_samplers,
            test=test,
            num_shared_layers=num_shared_layers,
        )
        
        # Compute logits
        feats_hat_logits: torch.Tensor = self.conditional_coder(x_hat_logits)

        assert feats.shape == feats_hat_logits.shape
        
        return feats_hat_logits, qs, ps, log_qs, log_ps
        
    def training_step(self, feats: torch.Tensor) -> torch.Tensor:
        feats_hat_logits, qs, ps, log_qs, log_ps = self(feats)
        
        # Compute loss
        loss = self.loss(feats, feats_hat_logits, qs, ps, log_qs, log_ps)
        self.log("loss/train", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, feats: torch.Tensor):
        feats_hat_logits, qs, ps, log_qs, log_ps = self(feats)
        
        # Compute loss
        loss = self.loss(feats, feats_hat_logits, qs, ps, log_qs, log_ps)
        self.log("loss/val", loss)
        
        print(f"Val loss: {loss}")
        
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Testing uses ACDC3DDataModule instead of ACDCDataModule to compute 3D
        Dice scores.
        """
        _, feats, _, _ = batch
        
        # 3D data module ensures 1 batch only, but each data point is 4D of
        # shape (S, C, W, H) where S is the number of slices.
        feats = feats.squeeze(0)
        
        self.log_reconstruction_metrics(feats)
        self.log_generation_metrics(feats)
        
        self.feats_buffer.append(feats)

    def log_reconstruction_metrics(self, feats: torch.Tensor):
        """
        Log reconstruction metrics to TensorBoard. This includes average
        reconstruction loss and Dice score across the batch.
        
        Args:
            feats (torch.Tensor): Batch of input samples.
        """
        feats_hat_logits, _, _, _, _ = self(feats, test=True)
        
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(feats, feats_hat_logits)
        self.log("loss/test_recon", recon_loss)
        
        # Compute Dice score
        feats_hat = torch.softmax(feats_hat_logits, dim=1)
        feats_hat_onehot = discretise(feats_hat)
        
        dice_score, dice_score_per_class = compute_dice_score(
            feats,
            feats_hat_onehot,
            self.device,
            is_3d=True,
            dice_per_class=True,
        )
        
        self.log("loss/dsc", dice_score)
        
        for i, dice_score in enumerate(dice_score_per_class):
            # i + 1 as excluding background class
            class_label = MASK_CLASSES[i + 1]
            self.log(f"loss/dsc_{class_label}", dice_score)

    def log_generation_metrics(self, feats: torch.Tensor):
        """
        Log generation metrics to TensorBoard. This includes the Frechet Resnet
        Distance with SimCLR (FRDS) metric across the batch.
        
        Args:
            feats (torch.Tensor): Batch of input samples.
        """
        num_samples, _, _, _ = feats.shape
        
        # Generate probabilistic segmentation maps
        x_fake = self.decoder.generate(num_samples, device=self.device)
        feats_fake = self.conditional_coder(x_fake)
        
        # Percentage of anatomically valid generations
        num_valid = 0
        
        for discretised_feat_fake in discretise(feats_fake):
            AV = AnatomicalValidityChecker(discretised_feat_fake)
            if AV.count_violations() == 0:
                num_valid += 1
        
        self.log("gen/anatomically_valid", num_valid / num_samples)
        
        # Keep track of all generations to compute FRDS
        self.feats_fake_buffer.append(feats_fake)
    
    def log_reconstruction_visualisation(self, feats: torch.Tensor):
        num_data = feats.shape[0]
        samples_idx = torch.randperm(num_data)[:40]
        feats = feats[samples_idx]
        feats_hat_logits, _, _, _, _ = self(feats, test=True)
        
        samples, reconstruction_pixel_error = get_samples_and_reconstructions_pixel_diff(feats, feats_hat_logits)
        show_samples(samples, reconstruction_pixel_error, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())
    
    def log_generation_visualisation(self, feats_fake: torch.Tensor):
        generations = torch.argmax(feats_fake[:40], dim=1).unsqueeze(1)
        show_samples(generations, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/generations", plt.gcf())

    def on_test_end(self):
        feats = torch.cat(self.feats_buffer, dim=0)
        feats_fake = torch.cat(self.feats_fake_buffer, dim=0)
        
        frds_value = compute_frds(
            feats,
            discretise(feats_fake),
            resnet_path=FRDS_MODEL_PATH,
            device=self.device,
        )

        print(f"FRDS: {frds_value}")
        self.logger.experiment.add_scalar("gen/frds", frds_value, 0)
        
        # Visualise samples and reconstructions
        self.log_reconstruction_visualisation(feats)
        
        # View generations
        self.log_generation_visualisation(feats_fake)
