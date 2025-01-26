import lightning as L
import math
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.cnvae.decoder import Decoder
from arch.nvae.distribution import Normal
from arch.nvae.encoder import Encoder
from utils.const import CARDIAC_WIDTH, MASK_CLASSES
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.eval import compute_dice_score, get_samples_and_reconstructions_pixel_diff
from utils.utils import clamp, discretise, show_samples

class CNVAE(L.LightningModule):
    """
    Conditional Nouveau VAE.
    """
    
    def __init__(
        self,
        in_channels: int=1,
        initial_channels: int=64,
        out_channels: int=4,
        min_channels: int=0,
        z_channels: int=20,
        num_groups_per_layer: list[int]=[4, 2, 1],
        is_layer_shared: list[bool]=[True, True, True],
        initial_downsample_factor: int=8,
        max_epochs: int=50,
        cbeta_per_layer: list[float]=[1.0, 1.0, 1.0],
        beta_per_layer: list[float]=[1.0, 1.0, 1.0],
        kl_warmup_steps: int=500,
        freeze_decoder: bool=False,
    ):
        """
        Create an instance of the Conditional NVAE model. All constructor
        arguments are saved in the checkpoint as hyperparameters.
        
        Args:
            in_channels (int): Number of input channels. Corresponds to number
                of colour channels in the input image. Default: 1.
            initial_channels (int): Number of channels @in_channels gets
                projected to. This is done immediately in the stem, then
                reverted at the very end of the pass in the conditional coder.
                Default: 64.
            out_channels (int): Number of output channels. Corresponds to number
                of classes in segmentation mask. Default: 4.
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
            cbeta_per_layer (list[float]): Beta coefficient for each shared
                layer, for the KL divergence between the variational posterior
                and the conditional prior. Order corresponds to the shared layers of @is_layer_shared. Default: [1.0, 1.0, 1.0].
            beta_per_layer (list[float]): Beta coefficient for each shared
                layer, for the KL divergence between the variational posterior
                and the unconditional prior. Default: [1.0, 1.0, 1.0].
            kl_warmup_steps (int): Number of steps to perform KL annealing.
                Each epoch has 214 steps. Default: 500.
            freeze_decoder (bool): If True, freeze the decoder and conditional 
                coder weights. Default: False.
        """
        
        super().__init__()
        
        assert len(num_groups_per_layer) == len(is_layer_shared)
        assert sum(is_layer_shared) == len(beta_per_layer)
        assert len(cbeta_per_layer) == len(beta_per_layer)
        
        self.save_hyperparameters()
        
        self.img_width = CARDIAC_WIDTH
        self.num_layers = len(self.hparams.num_groups_per_layer)
        self.num_latent_layers = len(self.hparams.beta_per_layer)
        
        self.layer_idx_to_latent_idx = self._get_layer_idx_to_latent_idx_map()
        
        # Conditional NVAE has 2 encoders that take in the image and the mask
        # respectively
        
        self.bottom_up = nn.ModuleDict({
            "image": nn.ModuleDict(),
            "mask": nn.ModuleDict(),
        })
        
        for key in self.bottom_up.keys():
            if key == "image":
                in_channels = self.hparams.in_channels
                use_extended_combiner = False
            else:
                # The mask encoder has the same number of channels as the output
                in_channels = self.hparams.out_channels
                use_extended_combiner = True
            
            # Table 6: # initial channels in enc. (NVAE paper)
            self.bottom_up[key]["stem"] = nn.Conv2d(
                in_channels,
                max(self.hparams.initial_channels, self.hparams.min_channels),
                kernel_size=3,
                padding=1,
                bias=True,
            )
            
            self.bottom_up[key]["encoder"] = Encoder(
                num_groups_per_layer=self.hparams.num_groups_per_layer,
                is_layer_shared=self.hparams.is_layer_shared,
                initial_channels=self.hparams.initial_channels,
                min_channels=self.hparams.min_channels,
                z_channels=self.hparams.z_channels,
                initial_downsample_factor=self.hparams.initial_downsample_factor,
                use_extended_combiner=use_extended_combiner,
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
                self.hparams.out_channels,
                kernel_size=3,
                padding=1,
            ),
        )
        
        if self.hparams.freeze_decoder:
            print("Freezing decoder and conditional coder weights.")
            
            self.decoder.requires_grad_(False)
            self.conditional_coder.requires_grad_(False)
            
            # Do not update running estimates for BatchNorm
            self.decoder.eval()
            self.conditional_coder.eval()
        
        # To keep track of test set and generated samples during test time, to
        # compute FRDS
        self.scans_buffer: list[torch.Tensor] = []
        self.feats_buffer: list[torch.Tensor] = []
    
    def get_image_stem(self):
        return self.bottom_up["image"]["stem"]

    def get_image_encoder(self):
        return self.bottom_up["image"]["encoder"]

    def get_mask_stem(self):
        return self.bottom_up["mask"]["stem"]
    
    def get_mask_encoder(self):
        return self.bottom_up["mask"]["encoder"]
    
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
    
    def _weight_kl(
        self,
        balanced_kl_divs: torch.Tensor,
        kl_latent_layers: torch.Tensor,
        beta_per_layer: list[float],
    ):
        """
        For each shared layer, weight the KL divergence by the corresponding
        beta. Equivalent to beta-VAE but applied to each shared layer.
        
        Args:
            balanced_kl_divs (torch.Tensor): KL per group after balancing,
                averaged over the batch. g-dim tensor for g groups.
            kl_latent_layers (torch.Tensor): Latent layer index for each KL term
                in balanced_kl_divs.
            beta_per_layer (list[float]): Beta multiplier for each shared layer.
        
        Returns:
            weighted_kls (torch.Tensor): Weighted KL per layer. n-dim tensor for
                n layers.
        """
        weighted_kls = torch.empty(self.num_latent_layers)
        
        for latent_idx in range(self.num_latent_layers):
            # Sum KL within each layer
            balanced_kl_div_layer = balanced_kl_divs[kl_latent_layers == latent_idx].sum()
            # Weight
            weighted_kls[latent_idx] = beta_per_layer[latent_idx] * balanced_kl_div_layer
        
        return weighted_kls
    
    def _kl_divergence(
        self,
        qs: list[Normal],
        ps: list[Normal],
        log_qs: list[torch.Tensor],
        log_ps: list[torch.Tensor],
        beta_per_layer: list[float],
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
        weighted_kls = self._weight_kl(
            balanced_kl_divs_batch_avg,
            kl_latent_layers,
            beta_per_layer,
        )
        
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
        cps: list[Normal],
        ps: list[Normal],
        log_qs: list[torch.Tensor],
        log_cps: list[torch.Tensor],
        log_ps: list[torch.Tensor],
        log_components: bool=True,
    ) -> torch.Tensor:
        """
        Compute the NVAE loss: sum of reconstruction loss and beta-weighted
        balanced KL divergence regulariser term.
        """
        assert x.shape == x_hat_logits.shape
        
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        
        balanced_kl_div = self._kl_divergence(
            qs,
            cps,
            log_qs,
            log_cps,
            beta_per_layer=self.hparams.beta_per_layer,
            log_components=log_components,
        )
        
        balanced_conditional_kl_div = self._kl_divergence(
            cps,
            ps,
            log_cps,
            log_ps,
            beta_per_layer=self.hparams.cbeta_per_layer,
            log_components=log_components,
        )
        
        print(f"Reconstruction loss: {recon_loss}")
        print(f"Weighted KL divergence: {balanced_kl_div}")
        print(f"Weighted conditional KL divergence: {balanced_conditional_kl_div}")
        
        if log_components:
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", balanced_kl_div)
            self.log("loss/kl_div_conditional", balanced_conditional_kl_div)
        
        return recon_loss + balanced_conditional_kl_div + balanced_kl_div

    def forward(
        self,
        scans: torch.Tensor,
        feats: torch.Tensor,
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:  
        """
        Forward pass at train (and validation) time. For inference, use the
        inference() method.
        """
        
        # Convert one-hot encoded inputs [0, 1] to [-1, 1] for train stability
        x = self.get_image_stem()(2 * scans - 1.0)
        y = self.get_mask_stem()(2 * feats - 1.0)
        
        # Pass through encoder
        x, xs, img_enc_combiner_cells = self.get_image_encoder()(x)
        y, ys, mask_enc_combiner_cells = self.get_mask_encoder()(y)
        
        # Reverse buffers and modules for decoder
        
        xs = xs[::-1]
        img_enc_combiner_cells = img_enc_combiner_cells[::-1]
        img_enc_samplers = self.get_image_encoder().samplers[::-1]
        
        ys = ys[::-1]
        mask_enc_combiner_cells = mask_enc_combiner_cells[::-1]
        mask_enc_samplers = self.get_mask_encoder().samplers[::-1]
        
        # Pass through decoder
        x_hat_logits, qs, cps, ps, log_qs, log_cps, log_ps = self.decoder(
            x,
            xs,
            y,
            ys,
            img_enc_combiner_cells,
            img_enc_samplers,
            mask_enc_combiner_cells,
            mask_enc_samplers,
        )
        
        # Compute logits
        feats_hat_logits: torch.Tensor = self.conditional_coder(x_hat_logits)
        
        return feats_hat_logits, qs, cps, ps, log_qs, log_cps, log_ps
    
    def inference(self, scans: torch.Tensor) -> torch.Tensor:
        """
        Forward pass at test time.
        """
        # Convert one-hot encoded inputs [0, 1] to [-1, 1] for train stability
        x = self.get_image_stem()(2 * scans - 1.0)
        
        # Pass through encoder
        x, xs, img_enc_combiner_cells = self.get_image_encoder()(x)
        
        # Reverse buffers and modules for decoder
        
        xs = xs[::-1]
        img_enc_combiner_cells = img_enc_combiner_cells[::-1]
        img_enc_samplers = self.get_image_encoder().samplers[::-1]
        
        # Pass through decoder
        x_hat_logits = self.decoder.inference(
            x,
            xs,
            img_enc_combiner_cells,
            img_enc_samplers,
        )
        
        # Compute logits
        feats_hat_logits: torch.Tensor = self.conditional_coder(x_hat_logits)
        
        return feats_hat_logits
    
    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]) -> torch.Tensor:
        scans, feats, _, _ = batch
        
        feats_hat_logits, qs, cps, ps, log_qs, log_cps, log_ps = self(
            scans,
            feats,
        )
        
        # Compute loss
        loss = self.loss(
            feats,
            feats_hat_logits,
            qs,
            cps,
            ps,
            log_qs,
            log_cps,
            log_ps,
            log_components=False,
        )
        self.log("loss/train", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        scans, feats, _, _ = batch
        
        feats_hat_logits, qs, cps, ps, log_qs, log_cps, log_ps = self(
            scans,
            feats,
        )
        
        # Compute loss
        loss = self.loss(
            feats,
            feats_hat_logits,
            qs,
            cps,
            ps,
            log_qs,
            log_cps,
            log_ps,
            log_components=False,
        )
        self.log("loss/val", loss)
        
        print(f"Val loss: {loss}")
        
        recon_loss = self.reconstruction_loss(feats, feats_hat_logits)
        self.log("loss/val_recon", recon_loss)
        
        # Also compute loss without mask prior
        feats_hat_logits = self.inference(scans)
        
        recon_loss = self.reconstruction_loss(feats, feats_hat_logits)
        self.log("loss/val_recon_no_mask_prior", recon_loss)
    
    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]):
        """
        Testing uses ACDC3DDataModule instead of ACDCDataModule to compute 3D
        Dice scores.
        """
        scans, feats, condition, ed = batch
        
        condition_label = f"condition_{int(condition)}"
        phase_label = "ed" if ed else "es"
        
        # 3D data module ensures 1 batch only, but each data point is 4D of
        # shape (S, C, W, H) where S is the number of slices.
        scans = scans.squeeze(0)
        feats = feats.squeeze(0)
        
        self.log_reconstruction_metrics(scans, feats, condition_label, phase_label)
        
        self.scans_buffer.append(scans)
        self.feats_buffer.append(feats)

    def log_reconstruction_metrics(self, scans: torch.Tensor, feats: torch.Tensor, condition: str, phase: str):
        """
        Log reconstruction metrics to TensorBoard. This includes average
        reconstruction loss and Dice score across the batch.
        
        Args:
            feats (torch.Tensor): Batch of input samples.
        """
        num_samples, _, _, _ = feats.shape
        
        feats_hat_logits = self.inference(scans)
        
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
        self.log(f"loss/dsc_{phase}", dice_score)
        self.log(f"loss/dsc_{condition}", dice_score)
        
        for i, dice_score in enumerate(dice_score_per_class):
            # i + 1 as excluding background class
            class_label = MASK_CLASSES[i + 1]
            self.log(f"loss/dsc_{class_label}", dice_score)
            self.log(f"loss/dsc_{phase}_{class_label}", dice_score)
            self.log(f"loss/dsc_{condition}_{class_label}", dice_score)
        
        # Compute anatomical validity
        num_valid = 0
        
        for discretised_feat_fake in discretise(feats_hat_logits):
            AV = AnatomicalValidityChecker(discretised_feat_fake)
            if AV.count_violations() == 0:
                num_valid += 1

        self.log("gen/anatomically_valid_recon", num_valid / num_samples)
        self.log(f"gen/anatomically_valid_recon_{phase}", num_valid / num_samples)
        self.log(f"gen/anatomically_valid_recon_{condition}", num_valid / num_samples)

    def log_reconstruction_visualisation(self, scans: torch.Tensor, feats: torch.Tensor):
        num_data = feats.shape[0]
        samples_idx = torch.randperm(num_data)[:40]
        scans = scans[samples_idx]
        feats = feats[samples_idx]
        feats_hat_logits = self.inference(scans)
        
        samples, reconstructions, reconstruction_pixel_error = get_samples_and_reconstructions_pixel_diff(feats, feats_hat_logits, return_reconstructions=True)
        
        show_samples(reconstructions, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())
        
        show_samples(samples, reconstruction_pixel_error, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions_diff", plt.gcf())

    def on_test_end(self):
        scans = torch.cat(self.scans_buffer, dim=0)
        feats = torch.cat(self.feats_buffer, dim=0)
        
        # Visualise samples and reconstructions
        self.log_reconstruction_visualisation(scans, feats)
