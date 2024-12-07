import math
import numpy as np
import torch
import torch.nn as nn

from arch.nvae.decoder import DecoderResidualCell
from arch.nvae.distribution import Normal

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
        self.top_prior = nn.Parameter(
            torch.rand(size=(self._num_channels(initial_channels), *self.top_latent_shape)),
            requires_grad=True,
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
        test: bool=False,
        num_shared_layers: int=-1,
        return_latents: bool=False,
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass: NVAE Decoder.
        
        Args:
            x (torch.Tensor): Top-level image encoding.
            xs (list[torch.Tensor]): Non-top-level image encodings.
            y (torch.Tensor): Top-level mask encoding.
            ys (list[torch.Tensor]): Non-top-level mask encodings.
            img_enc_combiner_cells (list[nn.Module]): Image encoder combiner cells.
            img_enc_samplers (list[nn.Module]): Image encoder samplers.
            mask_enc_combiner_cells (list[nn.Module]): Mask encoder combiner cells.
            mask_enc_samplers (list[nn.Module]): Mask encoder samplers.
            test (bool): Indicates whether test mode is enabled (compared to
                train or validation mode). If True, use deterministic sampling
                for all non-topmost latent layers, that is, take the mean of the
                residual distribution instead of sampling from it. Default:
                False.
            num_shared_layers (int): Number of latent layers shared with the
                decoder from the topmost layer. For example, if
                @num_shared_layers is 2, only the topmost and its immediate
                subsequent latent layer are shared. If a latent layer is not
                shared, the decoder does not draw information from the encoder.
                That is, the residual distribution only consists of the prior
                and not the variational posterior. If -1, all latent layers are
                shared. Useful for ablation study and checking collapsed layers.
                Default: -1.
        
        Returns:
            x (torch.Tensor): Output logits before passing through the
                conditional coder.
            qs (list[Normal]): Approximate posterior distributions.
            ps (list[Normal]): Prior distributions.
            log_qs (list[torch.Tensor]): Log probabilities of samples drawn from
                the residual distribution with respect to the approximate
                posterior.
            log_ps (list[torch.Tensor]): Log probabilities of samples drawn from
                the residual distribution with respect to the prior.
        """
        
        if num_shared_layers == 0:
            raise ValueError("Cannot have 0 shared layers")
        
        if num_shared_layers == -1:
            num_shared_layers = self.num_latent_layers
        else:
            assert test
            assert num_shared_layers <= self.num_latent_layers
        
        assert x.shape[0] == y.shape[0]
        batch_size = x.shape[0]
        
        # Top-level prior: Sample mu, logsig of the topmost latent layer
        latent_repr_x = img_enc_samplers[0](x)
        # [8, 20, 4, 4]
        mu_q, logsig_q = torch.chunk(latent_repr_x, 2, dim=1)
        
        # TODO Top-level prior should draw information from mask
        # comb_feats = combiner_cells...
        # latent_repr_y = mask_enc_samplers[0](comb_feats)
        [8, 20, 4, 4]
        # dmu_q, dlogsig_q = torch.chunk(latent_repr_y, 2, dim=1)
        
        print(f"mu_q: {mu_q.shape}")
        print(f"logsig_q: {logsig_q.shape}")
        
        # Top-level approximate posterior
        # distr = Normal(mu_q + dmu_q, logsig_q + dlogsig_q)
        distr = Normal(mu_q, logsig_q)
        # [8, 40, 4, 4]
        z = distr.sample(test)
        
        print(f"z: {z.shape}")
        
        if return_latents:
            zs = [z]
        
        qs = [distr]
        log_qs = [distr.log_p(z)]
        
        # Prior for top-level z
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
        
        print(f"h: {y_hat.shape}")
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Sample prior
                    latent_repr_p = self.samplers[idx_dec - 1](y_hat)
                    # [8, 20, 4, 4]
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    print(f"mu_p: {mu_p.shape}")
                    print(f"logsig_p: {logsig_p.shape}")
                    
                    if idx_dec < self.cumulative_groups_per_layer[num_shared_layers - 1]:
                        # Prior from image encoder
                        comb_feats_x = img_enc_combiner_cells[idx_dec - 1](xs[idx_dec - 1], y_hat)
                        latent_repr_x = img_enc_samplers[idx_dec](comb_feats_x)
                        # [8, 20, _, _]
                        dmu_p, dlogsig_p = torch.chunk(latent_repr_x, 2, dim=1)
                        
                        print(f"Image encoder prior dmu_p: {dmu_p.shape}")
                        print(f"Image encoder prior dlogsig_p: {dlogsig_p.shape}")
                        
                        import sys
                        sys.exit()
                        
                        # Variational posterior from mask encoder
                        comb_feats_x_y = mask_enc_combiner_cells[idx_dec - 1](ys[idx_dec - 1], y_hat)
                        latent_repr_x_y = mask_enc_samplers[idx_dec](comb_feats_x_y)
                        # [8, 20, _, _]
                        dmu_q, dlogsig_q = torch.chunk(latent_repr_x_y, 2, dim=1)

                        print(f"Mask encoder prior mu_q: {dmu_q.shape}")
                        print(f"Mask encoder prior logsig_q: {dlogsig_q.shape}")

                    else:
                        dmu_p = torch.zeros_like(mu_p)
                        dlogsig_p = torch.zeros_like(logsig_p)
                        dmu_q = torch.zeros_like(mu_p)
                        dlogsig_q = torch.zeros_like(logsig_p)

                    # Residual distribution i.e. approximate posterior
                    distr = Normal(
                        mu_p + dmu_p + dmu_q,
                        logsig_p + dlogsig_p + dlogsig_q,
                    )
                    z = distr.sample(deterministic=test)
                    qs.append(distr)
                    log_qs.append(distr.log_p(z))
                    
                    if return_latents:
                        zs.append(z)
                    
                    # Use prior
                    distr = Normal(mu_p, logsig_p)
                    ps.append(distr)
                    log_ps.append(distr.log_p(z))

                y_hat = cell(y_hat, z)
                
                idx_dec += 1
            else:
                y_hat = cell(y_hat)

        y_hat = self.postprocess(y_hat)
        
        if return_latents:
            return y_hat, qs, ps, log_qs, log_ps, zs
        
        return y_hat, qs, ps, log_qs, log_ps

    def generate(
        self,
        num_samples: int,
        device: torch.device,
        temp: float=1.0,
        num_sample_layers: int=-1,
        z: torch.Tensor=None,
    ) -> torch.Tensor:
        """
        Generate samples from a Gaussian prior.
        
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device used for Torch operations.
            temp (float): Sample temperature, i.e. standard deviation of the
                prior. A lower temperature is often used for generations as it
                allows the model to focus on the high probability region[1]. It
                results in less diverse samples. Default: 1.0.
            num_sample_layers (int): Number of latent layers from the topmost
                layer to sample from. For example, if @num_sample_layers is 2,
                only the topmost and its immediate subsequent latent layer are
                sampled from. All other subsequent latent layers use
                deterministic sampling, that is, take the mean of the prior
                distribution instead of sampling from it. If -1, sample from all
                latent layers. Useful for ablation study and checking collapsed
                layers. Default: -1.
            z (torch.Tensor): Custom fixed latent representation. If provided,
                the model will use this as the topmost latent variable instead
                of sampling from a Gaussian prior. Default: None.
        
        Returns:
            x (torch.Tensor): Generated samples.
            
        [1]: Kingma DP, Dhariwal P. Glow: Generative flow with invertible 1x1
        convolutions. Advances in neural information processing systems.
        2018;31.
        """
        if num_sample_layers == -1:
            num_sample_layers = self.num_latent_layers
        else:
            assert num_sample_layers <= self.num_latent_layers
        
        # Form posterior for top-level assuming Gaussian prior
        if z is None:
            top_latent_shape = (num_samples, self.z_channels, *self.top_latent_shape)
            distr = Normal(
                mu=torch.zeros(top_latent_shape).to(device),
                logsig=torch.zeros(top_latent_shape).to(device),
                temp=temp,
            )

            z = distr.sample()
        
        # Latent group index
        idx_dec = 0
        
        # [1, top_channels, width, height]
        x = self.top_prior.unsqueeze(0)
        # [num_samples, top_channels, width, height]
        x = x.expand(num_samples, -1, -1, -1)
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Form prior
                    latent_repr_p = self.samplers[idx_dec - 1](x)
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    sample_deterministic = idx_dec >= self.cumulative_groups_per_layer[num_sample_layers - 1]
                    distr = Normal(mu_p, logsig_p, temp=temp)
                    z = distr.sample(sample_deterministic)

                x = cell(x, z)
                
                idx_dec += 1
            else:
                x = cell(x)
        
        x = self.postprocess(x)
        
        return x
