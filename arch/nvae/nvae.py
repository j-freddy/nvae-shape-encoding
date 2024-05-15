from collections import defaultdict
import math
import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.nvae.decoder import Decoder
from arch.nvae.distribution import Normal
from arch.nvae.encoder import Encoder
from utils import show_samples

class NVAE(L.LightningModule):
    """
    Nouveau Variational Autoencoder (NVAE) is a deep hierarchical VAE that
    achieves SOTA in image generation tasks among non-autoregressive models.
    This is an implementation of the framework proposed in [1], primarily based
    on details described in the paper and the official codebase.
    
    Indexing of latent groups is as follows: 0 corresponds to the shallowest
    group and n-1 corresponds to the topmost group, for n latent scales.
    
    [1]: Vahdat A, Kautz J. NVAE: A deep hierarchical variational autoencoder.
    Advances in neural information processing systems. 2020;33:19667-79.
    """
    
    def __init__(
        self,
        in_channels: int=4,
        initial_channels: int=64,
        num_latent_scales: int=3,
        # Topmost latent scale has fewest groups (i.e. 1)
        # Shallowest latent scale has most groups (i.e. 4)
        num_groups_per_scale: list[int]=[4, 2, 1],
        initial_downsample_factor: int=8,
        max_epochs: int=50,
        beta_per_scale: list[float]=[1.0, 1.0, 1.0],
        kl_warmup_steps: int=500,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        # TODO Do not hardcode this
        self.img_width = 128
        
        # Table 6: # initial channels in enc. (NVAE paper)
        self.stem = nn.Conv2d(
            self.hparams.in_channels,
            self.hparams.initial_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        
        self.encoder = Encoder(
            num_latent_scales=self.hparams.num_latent_scales,
            num_groups_per_scale=self.hparams.num_groups_per_scale,
            initial_channels=self.hparams.initial_channels,
            initial_downsample_factor=self.hparams.initial_downsample_factor,
        )
        
        top_latent_dim = self._get_latent_dim(self.hparams.num_latent_scales - 1)
        print(f"Top latent dim: {top_latent_dim}")

        self.decoder = Decoder(
            initial_channels=self.hparams.initial_channels * (2 ** (self.hparams.num_latent_scales - 1)),
            num_latent_scales=self.hparams.num_latent_scales,
            num_groups_per_scale=self.hparams.num_groups_per_scale[::-1],
            top_latent_shape=(top_latent_dim, top_latent_dim),
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
        # Layer @(num_latent_scales - 1) is the deepest (topmost) layer
        return (self.img_width // self.hparams.initial_downsample_factor) // (2 ** layer)

    def _get_layer_index(self, latent_dim: int) -> int:
        # Inverse of _get_latent_dim
        idx = math.log2((self.img_width // self.hparams.initial_downsample_factor) // latent_dim)
        assert idx.is_integer()
        return int(idx)
    
    def _kl_divergence(
        self,
        qs: list[Normal],
        ps: list[Normal],
        log_qs: list[torch.Tensor],
        log_ps: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # log_p, log_q and kl_diag are for metrics purposes
        
        kl_div = defaultdict(lambda: [])
        kl_diag = []
        
        log_p: float = 0
        log_q: float = 0
        
        for q, p, log_q_conv, log_p_conv in zip(qs, ps, log_qs, log_ps):
            assert q.mu.shape == p.mu.shape
            
            _, _, width, height = q.mu.shape
            assert width == height
            curr_layer = self._get_layer_index(width)
            
            # TODO Change this for normalising flow
            kl_per_var = q.kl(p)

            kl_diag.append(torch.mean(torch.sum(kl_per_var, dim=[2, 3]), dim=0))
            kl_div[curr_layer].append(torch.sum(kl_per_var, dim=[1, 2, 3]))
            log_q += torch.sum(log_q_conv, dim=[1, 2, 3])
            log_p += torch.sum(log_p_conv, dim=[1, 2, 3])
        
        weighted_kl_divs = []
        
        for layer_idx, kl_div_per_layer in kl_div.items():
            kl_div = torch.sum(torch.stack(kl_div_per_layer, dim=1), dim=1).mean()
            weighted_kl_divs.append(
                self.hparams.beta_per_scale[layer_idx] * kl_div,
            )
        
        return weighted_kl_divs.sum() / len(weighted_kl_divs)

    def reconstruction_loss(self, x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return F.binary_cross_entropy_with_logits(x_hat, x, reduction="sum") / batch_size
    
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
        recon_loss = self.reconstruction_loss(x, x_hat)
        weighted_kl_div = self._kl_divergence(qs, ps, log_qs, log_ps)
        
        # Linear KL warm-up
        if self.global_step < self.hparams.kl_warmup_steps:
            weighted_kl_div *= self.global_step / self.hparams.kl_warmup_steps
        
        print(f"Reconstruction loss: {recon_loss}")
        print(f"Weighted KL divergence: {weighted_kl_div}")
        
        if log_components:
            self.log("recon_loss", recon_loss)
            self.log("kl_div", weighted_kl_div)
        
        return recon_loss + weighted_kl_div
    
    def forward(self, feats: torch.Tensor) -> tuple[torch.Tensor, list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor]]:
        # TODO Official NVAE implementation uses s = self.stem(2 * x - 1.0)
        x = self.stem(feats)
        
        # Pass through encoder
        x, xs, enc_combiner_cells = self.encoder(x)
        
        # Reverse buffers and modules for decoder
        xs = xs[::-1]
        enc_combiner_cells = enc_combiner_cells[::-1]
        enc_samplers = self.encoder.samplers[::-1]
        
        # Pass through decoder
        x_hat, qs, ps, log_qs, log_ps = self.decoder(x, xs, enc_combiner_cells, enc_samplers)
        
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
        
        # TODO Using the first 20 samples only
        feats = feats[:20]

        # Compute loss
        feats_hat, _, _, _, _ = self(feats)
        recon_loss = self.reconstruction_loss(feats, feats_hat)
        self.log("test_recon_loss", recon_loss)

        self.log_reconstructions(feats[:20])
        self.log_generations_and_fid(feats)

    def log_reconstructions(self, x: torch.Tensor):
        # TODO This is mostly duplicate code from VAE class
        
        x_hat, _, _, _, _ = self(x)

        reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
        samples = torch.argmax(x, dim=1).unsqueeze(1)

        # Interleave samples and reconstructions
        batch_size, num_channels, width, height = samples.shape
        assert width == height
        samples_and_reconstructions = torch.empty(batch_size * 2, num_channels, width, height)
        
        for i in range(samples.shape[0]):
            samples_and_reconstructions[i * 2] = samples[i]
            samples_and_reconstructions[i * 2 + 1] = reconstructions[i]
        
        show_samples(samples_and_reconstructions, rgb=False, nrow=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())

    def log_generations_and_fid(self, feats: torch.Tensor):
        num_samples, _, _, _ = feats.shape
        
        # Generate probabilistic segmentation maps
        x_fake = self.decoder.generate(num_samples, device=feats.device)
        feats_fake = self.conditional_coder(x_fake)

        # Discretise probabilistic map then view generations
        generations = torch.argmax(x_fake[:20], dim=1).unsqueeze(1)
        show_samples(generations, rgb=False, nrow=10, figsize=(10, 2), display=False)
        self.logger.experiment.add_figure("img/generations", plt.gcf())

        # TODO FID
