import math
import numpy as np
import torch
import torch.nn as nn

from arch.nvae.decoder import DecoderResidualCell
from arch.nvae.distribution import Normal
from arch.nvae.encoder import EncoderCombinerCell

class DecoderCombinerCell(nn.Module):
    """
    Decoder combiner cell.
    
    Following the official NVAE implementation from class DecCombinerCell in
    neural_operations.py.
    """
    
    def __init__(self, in_channels_x1: int, in_channels_x2: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Conv2d(in_channels_x1 + in_channels_x2, out_channels, kernel_size=1)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x1, x2], dim=1))

class Decoder(nn.Module):
    """
    Conditional NVAE Decoder.
    """
    
    def __init__(
        self,
        # This must be the reverse of the encoder, otherwise the shapes of
        # samples drawn from the encoder samplers will not match
        num_groups_per_layer: list[int],
        # 1-to-1 map with num_groups_per_layer
        is_layer_shared: list[bool],
        initial_channels: int=256,
        min_channels: int=16,
        top_latent_shape: tuple[int, int]=(4, 4),
        z_channels: int=20,
        # This must match initial_downsample_factor of Encoder
        final_upsample_factor: int=2,
    ):
        super().__init__()
        
        assert len(num_groups_per_layer) == len(is_layer_shared)
        
        self.min_channels = min_channels
        self.num_latent_layers = len(num_groups_per_layer)
        self.z_channels = z_channels
        self.top_latent_shape = top_latent_shape
        
        # Get end indices for each latent layer
        self.cumulative_groups_per_layer = np.array(num_groups_per_layer).cumsum()

        # Size of the topmost prior: [top_channels, width, height]
        top_channels = self._num_channels(initial_channels)
        self.top_prior = nn.Parameter(
            torch.rand(size=(top_channels, *self.top_latent_shape)),
            requires_grad=True,
        )
        
        # TODO This should technically be part of the Encoder
        self.top_combiner_cell = EncoderCombinerCell(top_channels, top_channels)
        self.top_sampler = nn.Conv2d(
            top_channels,
            2 * z_channels,
            kernel_size=3,
            padding=1,
        )
        
        # Build tower
        
        self.tower = nn.ModuleList()
        self.samplers = nn.ModuleList()
        
        num_channels = initial_channels
        
        for s in range(self.num_latent_layers):
            for g in range(num_groups_per_layer[s]):
                true_num_channels = self._num_channels(num_channels)
                
                if not (s == 0 and g == 0):
                    # Residual cells
                    # Official implementation uses mconv_e6k5g0
                    self.tower.append(DecoderResidualCell(true_num_channels))
                    self.tower.append(DecoderResidualCell(true_num_channels))
                    
                    if is_layer_shared[s]:
                        # Add sampler
                        self.samplers.append(
                            nn.Sequential(
                                nn.ELU(),
                                nn.Conv2d(
                                    true_num_channels,
                                    2 * z_channels,
                                    kernel_size=1,
                                ),
                            ),
                        )

                if is_layer_shared[s]:
                    # Add dec combiner
                    self.tower.append(
                        DecoderCombinerCell(
                            true_num_channels,
                            z_channels,
                            true_num_channels,
                        ),
                    )
            
            if s < self.num_latent_layers - 1:
                # Upsample
                # Official implementation uses mconv_e6k5g0 with kernel size 5
                self.tower.append(
                    nn.ConvTranspose2d(
                        true_num_channels,
                        self._num_channels(num_channels // 2),
                        kernel_size=5,
                        stride=2,
                        padding=2,
                        output_padding=1,
                        bias=False,
                    ),
                )
                
                num_channels //= 2
    
        # Build postprocessing modules
        
        postprocess_modules = []
        num_postprocess_layers = int(math.log2(final_upsample_factor))
        
        for _ in range(num_postprocess_layers):
            true_num_channels_halved = self._num_channels(num_channels // 2)
            
            postprocess_modules.append(
                nn.ConvTranspose2d(
                    self._num_channels(num_channels),
                    true_num_channels_halved,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )
            )
            postprocess_modules.append(DecoderResidualCell(true_num_channels_halved))
            postprocess_modules.append(DecoderResidualCell(true_num_channels_halved))
            
            num_channels //= 2
        
        self.postprocess = nn.Sequential(*postprocess_modules)
    
    def _num_channels(self, num_channels: int) -> int:
        return max(num_channels, self.min_channels)
    
    def get_top_latent_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        return (batch_size, self.z_channels, *self.top_latent_shape)

    def forward(
        self,
        x: torch.Tensor,
        xs: list[torch.Tensor],
        y: torch.Tensor,
        ys: list[torch.Tensor],
        img_enc_combiner_cells: list[nn.Module],
        img_enc_samplers: list[nn.Module],
        mask_enc_combiner_cells: list[nn.Module],
        mask_enc_samplers: list[nn.Module],
        return_latents: bool=False,
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass: CNVAE Decoder. This method should be called during
        training (and validation) only. For inference, use the inference()
        method.
        
        Args:
            x (torch.Tensor): Top-level image encoding.
            xs (list[torch.Tensor]): Non-top-level image encodings.
            y (torch.Tensor): Top-level mask encoding.
            ys (list[torch.Tensor]): Non-top-level mask encodings.
            img_enc_combiner_cells (list[nn.Module]): Image encoder combiner
                cells.
            img_enc_samplers (list[nn.Module]): Image encoder samplers.
            mask_enc_combiner_cells (list[nn.Module]): Mask encoder combiner
                cells.
            mask_enc_samplers (list[nn.Module]): Mask encoder samplers.
        
        Returns:
            x (torch.Tensor): Output logits before passing through the
                conditional coder.
            qs (list[Normal]): Approximate posterior distributions.
            cps (list[Normal]): Conditional prior distributions.
            ps (list[Normal]): Prior distributions.
            log_qs (list[torch.Tensor]): Log probabilities of samples drawn 
                from qs.
            log_cps (list[torch.Tensor]): Log probabilities of samples drawn
                from cps.
            log_ps (list[torch.Tensor]): Log probabilities of samples drawn from
                ps.
        """
        assert x.shape[0] == y.shape[0]
        batch_size = x.shape[0]
        
        # Top-level prior: Sample mu, logsig of the topmost latent layer
        latent_repr_x = img_enc_samplers[0](x)
        # [8, 20, 4, 4]
        mu_p, logsig_p = torch.chunk(latent_repr_x, 2, dim=1)
        
        # Top-level prior should draw information from mask
        [8, 20, 4, 4]
        comb_feats = self.top_combiner_cell(y, x)
        latent_repr_y = self.top_sampler(comb_feats)
        # [8, 20, _, _]
        dmu_q, dlogsig_q = torch.chunk(latent_repr_y, 2, dim=1)
        
        # TODO Revert
        dmu_q = torch.zeros_like(mu_p)
        dlogsig_q = torch.zeros_like(logsig_p)
        
        # Top-level approximate posterior
        distr = Normal(mu_p + dmu_q, logsig_p + dlogsig_q)
        # [8, 20, 4, 4]
        z = distr.sample()
        
        if return_latents:
            zs = [z]
        
        qs = [distr]
        log_qs = [distr.log_p(z)]
        
        # Record conditional prior for top-level z
        distr = Normal(mu=mu_p, logsig=logsig_p)
        cps = [distr]
        log_cps = [distr.log_p(z)]
        
        # Record unconditional prior for top-level z (no mask information)
        distr = Normal(mu=torch.zeros_like(z), logsig=torch.zeros_like(z))
        ps = [distr]
        log_ps = [distr.log_p(z)]
        
        # Latent group index
        idx_dec = 0
        # [1, top_channels, width, height]
        y_hat = self.top_prior.unsqueeze(0)
        # [batch_size, top_channels, width, height]
        # [8, 128, 4, 4]
        y_hat = y_hat.expand(batch_size, -1, -1, -1)
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Sample prior
                    latent_repr_p = self.samplers[idx_dec - 1](y_hat)
                    # [8, 20, 4, 4] for topmost layer
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    if idx_dec < self.cumulative_groups_per_layer[self.num_latent_layers - 1]:
                        # Prior from image encoder
                        comb_feats_x = img_enc_combiner_cells[idx_dec - 1](
                            xs[idx_dec - 1],
                            y_hat,
                        )
                        latent_repr_x = img_enc_samplers[idx_dec](comb_feats_x)
                        # [8, 20, _, _]
                        dmu_p, dlogsig_p = torch.chunk(latent_repr_x, 2, dim=1)
                        
                        # Variational posterior from mask encoder
                        comb_feats_x_y = mask_enc_combiner_cells[idx_dec - 1](
                            ys[idx_dec - 1],
                            xs[idx_dec - 1],
                            y_hat,
                        )
                        latent_repr_x_y = mask_enc_samplers[idx_dec](comb_feats_x_y)
                        # [8, 20, _, _]
                        dmu_q, dlogsig_q = torch.chunk(latent_repr_x_y, 2, dim=1)
                    else:
                        dmu_p = torch.zeros_like(mu_p)
                        dlogsig_p = torch.zeros_like(logsig_p)
                        dmu_q = torch.zeros_like(mu_p)
                        dlogsig_q = torch.zeros_like(logsig_p)
                    
                    # TODO Revert
                    dmu_q = torch.zeros_like(mu_p)
                    dlogsig_q = torch.zeros_like(logsig_p)

                    # Residual distribution i.e. approximate posterior
                    distr = Normal(
                        mu_p + dmu_p + dmu_q,
                        logsig_p + dlogsig_p + dlogsig_q,
                    )
                    z = distr.sample()
                    qs.append(distr)
                    log_qs.append(distr.log_p(z))
                    
                    if return_latents:
                        zs.append(z)
                    
                    # Record conditional prior
                    distr = Normal(mu_p + dmu_p, logsig_p + dlogsig_p)
                    cps.append(distr)
                    log_cps.append(distr.log_p(z))
                    
                    # Record unconditional prior
                    distr = Normal(mu_p, logsig_p)
                    ps.append(distr)
                    log_ps.append(distr.log_p(z))

                y_hat = cell(y_hat, z)
                
                idx_dec += 1
            else:
                y_hat = cell(y_hat)

        y_hat = self.postprocess(y_hat)
        
        if return_latents:
            return y_hat, qs, cps, ps, log_qs, log_cps, log_ps, zs
        
        return y_hat, qs, cps, ps, log_qs, log_cps, log_ps
    
    def inference(
        self,
        x: torch.Tensor,
        xs: list[torch.Tensor],
        img_enc_combiner_cells: list[nn.Module],
        img_enc_samplers: list[nn.Module],
        return_latents: bool=False,
    ) -> torch.Tensor:
        """
        Forward pass at test time, when the ground truth mask is not available.
        
        Args:
            x (torch.Tensor): Top-level image encoding.
            xs (list[torch.Tensor]): Non-top-level image encodings.
            img_enc_combiner_cells (list[nn.Module]): Image encoder combiner
                cells.
            img_enc_samplers (list[nn.Module]): Image encoder samplers.
        
        Returns:
            x (torch.Tensor): Output logits before passing through the
                conditional coder.
        """
        batch_size = x.shape[0]
        
        # Top-level prior: Sample mu, logsig of the topmost latent layer
        latent_repr_x = img_enc_samplers[0](x)
        # [batch_size, 20, 4, 4])
        mu_p, logsig_p = torch.chunk(latent_repr_x, 2, dim=1)
        
        # Top-level conditional prior
        distr = Normal(mu_p, logsig_p)
        # [batch_size, 20, 4, 4]
        z = distr.sample(deterministic=True)
        
        if return_latents:
            zs = [z]
        
        # Latent group index
        idx_dec = 0
        # [1, top_channels, width, height]
        y_hat = self.top_prior.unsqueeze(0)
        # [batch_size, top_channels, width, height]
        # [batch_size, 128, 4, 4]
        y_hat = y_hat.expand(batch_size, -1, -1, -1)
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Sample prior
                    latent_repr_p = self.samplers[idx_dec - 1](y_hat)
                    # [batch_size, 20, 4, 4] for topmost layer
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    if idx_dec < self.cumulative_groups_per_layer[self.num_latent_layers - 1]:
                        # Prior from image encoder
                        comb_feats_x = img_enc_combiner_cells[idx_dec - 1](
                            xs[idx_dec - 1],
                            y_hat,
                        )
                        latent_repr_x = img_enc_samplers[idx_dec](comb_feats_x)
                        # [8, 20, _, _]
                        dmu_p, dlogsig_p = torch.chunk(latent_repr_x, 2, dim=1)
                    else:
                        dmu_p = torch.zeros_like(mu_p)
                        dlogsig_p = torch.zeros_like(logsig_p)

                    # Residual distribution i.e. approximate posterior
                    distr = Normal(mu_p + dmu_p, logsig_p + dlogsig_p)
                    z = distr.sample(deterministic=True)
                    
                    if return_latents:
                        zs.append(z)

                y_hat = cell(y_hat, z)
                
                idx_dec += 1
            else:
                y_hat = cell(y_hat)

        y_hat = self.postprocess(y_hat)
        
        if return_latents:
            return y_hat, zs
        
        return y_hat
