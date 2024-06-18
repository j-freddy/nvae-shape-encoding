import math
import numpy as np
import torch
import torch.nn as nn
import torchvision.ops as ops

from arch.nvae.distribution import Normal

class DecoderResidualCell(nn.Module):
    """
    Decoder residual cell.
    
    Implementation as described by Fig. 3a in the NVAE paper. BatchNorm momentum
    is set to 0.05 in official NVAE implementation. Expand factor for depthwise
    convolution is set to either 6 (MobileNetV2) or 3. The latter reduces
    memory.
    """

    def __init__(self, num_channels: int, expand_factor: int=6):
        super().__init__()
        
        self.net = nn.Sequential(
            # BN
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            # Conv 1x1 (expand)
            nn.Conv2d(num_channels, num_channels * expand_factor, kernel_size=1, bias=False),
            # BN + Swish
            nn.BatchNorm2d(num_features=num_channels * expand_factor, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            # Depthwise convolution
            nn.Conv2d(
                num_channels * expand_factor,
                num_channels * expand_factor,
                # Following the official NVAE implementation from OPS in neural_operations.py
                kernel_size=5,
                padding=2,
                groups=num_channels * expand_factor,
                bias=False,
            ),
            # BN + Swish
            nn.BatchNorm2d(num_features=num_channels * expand_factor, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            # Conv 1x1 (revert)
            nn.Conv2d(num_channels * expand_factor, num_channels, kernel_size=1, bias=False),
            # BN
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            # SE
            ops.SqueezeExcitation(
                num_channels,
                # Following the official NVAE implementation from class SE in
                # neural_operations.py
                squeeze_channels=max(num_channels // 16, 4),
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

class DecoderCombinerCell(nn.Module):
    """
    Decoder combiner cell.
    
    Following the official NVAE implementation from class DecCombinerCell in
    neural_operations.py.
    """
    
    def __init__(self, in_channels_x1: int, in_channels_x2: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Conv2d(in_channels_x1 + in_channels_x2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        return self.net(torch.cat([x1, x2], dim=1))

class Decoder(nn.Module):
    """
    NVAE Decoder.
    
    Implementation is adapted from this diagram:
    - https://github.com/NVlabs/NVAE/blob/master/img/model_diagram.png
    
    Also see init_decoder_tower() in official NVAE code.
    """
    
    def __init__(
        self,
        num_latent_layers: int,
        # This must be the reverse of the encoder, otherwise the shapes of
        # samples drawn from the encoder samplers will not match
        num_groups_per_layer: list[int],
        initial_channels: int=256,
        top_latent_shape: tuple[int, int]=(4, 4),
        z_channels: int=20,
        # This must match initial_downsample_factor of Encoder
        final_upsample_factor: int=2,
    ):
        super().__init__()
        
        assert len(num_groups_per_layer) == num_latent_layers
        
        self.num_latent_layers = num_latent_layers
        self.z_channels = z_channels
        self.top_latent_shape = top_latent_shape
        
        # Get end indices for each latent layer
        self.cumulative_groups_per_layer = np.array(num_groups_per_layer).cumsum()

        # Size of the topmost prior: [top_channels, width, height]
        self.top_prior = nn.Parameter(
            torch.rand(size=(initial_channels, *self.top_latent_shape)),
            requires_grad=True,
        )
        
        # Build tower
        
        self.tower = nn.ModuleList()
        self.samplers = nn.ModuleList()
        
        num_channels = initial_channels
        
        for s in range(num_latent_layers):
            for g in range(num_groups_per_layer[s]):
                if not (s == 0 and g == 0):
                    # Residual cells
                    # Official implementation uses mconv_e6k5g0
                    self.tower.append(DecoderResidualCell(num_channels))
                    self.tower.append(DecoderResidualCell(num_channels))
                    
                    # Add sampler
                    self.samplers.append(
                        nn.Sequential(
                            nn.ELU(),
                            nn.Conv2d(
                                num_channels,
                                2 * z_channels,
                                kernel_size=1,
                            ),
                        ),
                    )

                # Add dec combiner
                self.tower.append(
                    DecoderCombinerCell(
                        num_channels,
                        z_channels,
                        num_channels,
                    ),
                )
            
            if s < num_latent_layers - 1:
                # Upsample
                # Official implementation uses mconv_e6k5g0 with kernel size 5
                self.tower.append(
                    nn.ConvTranspose2d(
                        num_channels,
                        num_channels // 2,
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
            postprocess_modules.append(
                nn.ConvTranspose2d(
                    num_channels,
                    num_channels // 2,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    output_padding=1,
                    bias=False,
                )
            )
            postprocess_modules.append(DecoderResidualCell(num_channels // 2))
            postprocess_modules.append(DecoderResidualCell(num_channels // 2))
            
            num_channels //= 2
        
        self.postprocess = nn.Sequential(*postprocess_modules)

    def forward(
        self,
        x: torch.Tensor,
        xs: torch.Tensor,
        enc_combiner_cells: list[nn.Module],
        enc_samplers: list[nn.Module],
        test: bool=False,
        num_shared_layers: int=-1,
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass: NVAE Decoder.
        
        Args:
            x (torch.Tensor): Top-level encoding.
            xs (torch.Tensor): Non-top-level encodings.
            enc_combiner_cells (list[nn.Module]): Encoder combiner cells.
            enc_samplers (list[nn.Module]): Encoder samplers.
            test (bool): Indicates whether test mode is enabled (compared to
                train or validation mode). If True, use deterministic sampling
                for all non-topmost latent layers, that is, take the mean of the
                residual distribution instead of sampling from it. Default:
                False.
            num_shared_layers (int): Number of latent layers shared with the
                decoder from the topmost layer. For example, if
                @num_shared_layers is 2, only the topmost and its immediate
                subsequent layer are shared. If a layer is not shared, the
                decoder does not draw information from the encoder. That is, the
                residual distribution only consists of the prior and not the
                approximate posterior. If -1, all layers are shared. Useful for
                ablation study and checking collapsed layers. Default: -1.
        
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
        if num_shared_layers == -1:
            num_shared_layers = self.num_latent_layers
        else:
            assert test
            assert num_shared_layers <= self.num_latent_layers
        
        batch_size, _, _, _ = x.shape
        
        # Sample mu, logsig of the topmost latent layer
        latent_repr_q = enc_samplers[0](x)
        mu_q, logsig_q = torch.chunk(latent_repr_q, 2, dim=1)
        # Approximate posterior for top-level
        distr = Normal(mu_q, logsig_q)
        z = distr.sample(test)
        
        qs = [distr]
        log_qs = [distr.log_p(z)]
        
        # TODO Normalising flows skipped
        
        # Prior for top-level z
        distr = Normal(mu=torch.zeros_like(z), logsig=torch.zeros_like(z))
        ps = [distr]
        log_ps = [distr.log_p(z)]
        
        idx_dec = 0
        # [1, top_channels, width, height]
        x = self.top_prior.unsqueeze(0)
        # [batch_size, top_channels, width, height]
        x = x.expand(batch_size, -1, -1, -1)
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Sample prior
                    latent_repr_p = self.samplers[idx_dec - 1](x)
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    # Approximate posterior
                    if idx_dec < self.cumulative_groups_per_layer[num_shared_layers - 1]:
                        comb_feats = enc_combiner_cells[idx_dec - 1](xs[idx_dec - 1], x)
                        latent_repr_q = enc_samplers[idx_dec](comb_feats)
                        mu_q, logsig_q = torch.chunk(latent_repr_q, 2, dim=1)
                    else:
                        mu_q = torch.zeros_like(mu_p)
                        logsig_q = torch.zeros_like(logsig_p)

                    # Residual distribution
                    distr = Normal(mu_p + mu_q, logsig_p + logsig_q)
                    z = distr.sample(deterministic=test)
                    qs.append(distr)
                    log_qs.append(distr.log_p(z))
                    
                    # Use prior
                    distr = Normal(mu_p, logsig_p)
                    ps.append(distr)
                    log_ps.append(distr.log_p(z))

                x = cell(x, z)
                
                idx_dec += 1
            else:
                x = cell(x)

        x = self.postprocess(x)
        
        return x, qs, ps, log_qs, log_ps

    def get_top_latent_shape(self, batch_size: int) -> tuple[int, int, int, int]:
        return (batch_size, self.z_channels, *self.top_latent_shape)

    def generate(
        self,
        num_samples: int,
        device: torch.device,
        num_sample_layers: int=-1,
        z: torch.Tensor=None,
    ) -> torch.Tensor:
        """
        Generate samples from a Gaussian prior.
        
        Args:
            num_samples (int): Number of samples to generate.
            device (torch.device): Device used for Torch operations.
            num_sample_layers (int): Number of latent layers from the topmost
                layer to sample from. For example, if @num_sample_layers is 2,
                only the topmost and its immediate subsequent layer are sampled
                from. All other subsequent layers use deterministic sampling,
                that is, take the mean of the prior distribution instead of
                sampling from it. If -1, sample from all layers. Useful for
                ablation study and checking collapsed layers. Default: -1.
            z (torch.Tensor): Custom fixed latent representation. If provided,
                the model will use this as the topmost latent variable instead
                of sampling from a Gaussian prior. Default: None.
        
        Returns:
            x (torch.Tensor): Generated samples.
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
            )

            z = distr.sample()
        
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
                    distr = Normal(mu_p, logsig_p)
                    z = distr.sample(sample_deterministic)

                x = cell(x, z)
                
                idx_dec += 1
            else:
                x = cell(x)
        
        x = self.postprocess(x)
        
        return x
