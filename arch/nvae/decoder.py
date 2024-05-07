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
    neural_operations.py
    """
    
    def __init__(self, in_channels_x1: int, in_channels_x2: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Conv2d(in_channels_x1 + in_channels_x2, out_channels, kernel_size=1)

    def forward(self, x1, x2):
        return self.net(torch.cat([x1, x2], dim=1))

class Decoder(nn.Module):
    """
    NVAE Decoder.
    
    Implementation as described by the diagram: -
    https://github.com/NVlabs/NVAE/blob/master/img/model_diagram.png
    
    Also see init_decoder_tower() in official NVAE code.
    """
    
    def __init__(self, initial_channels: int=256, z_channels: int=20):
        super().__init__()
        
        # TODO Do not hardcode the size
        # This is just the size of the topmost prior
        # To work it out, just use print(x.shape) after self.encoder(x) in
        # nvae.py
        self.top_prior = nn.Parameter(torch.rand(size=(32, 32, 32)), requires_grad=True)
        
        # Build tower
        
        self.tower = nn.ModuleList()
        self.samplers = nn.ModuleList()
        
        # TODO This must match Encoder, but num_groups_per_scale must be
        # reversed
        num_latent_scales = 3
        num_groups_per_scale = [4, 2, 1]
        num_groups_per_scale = num_groups_per_scale[::-1]
        
        num_channels = initial_channels
        
        for s in range(num_latent_scales):
            for g in range(num_groups_per_scale[s]):
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
            
            if s < num_latent_scales - 1:
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
    
        # Build postprocessing layers
        
        self.postprocess = nn.Sequential(
            DecoderResidualCell(num_channels),
            DecoderResidualCell(num_channels),
        )

    def forward(
        self,
        x: torch.Tensor,
        xs: torch.Tensor,
        enc_combiner_cells: list[nn.Module],
        enc_samplers: list[nn.Module],
    ) -> tuple[torch.Tensor, list[Normal], list[Normal], list[torch.Tensor], list[torch.Tensor]]:
        batch_size, _, _, _ = x.shape
        
        # Sample mu, logsig of the topmost latent scale
        latent_repr_q = enc_samplers[0](x)
        mu_q, logsig_q = torch.chunk(latent_repr_q, 2, dim=1)
        # Approximate posterior for top-level
        distr = Normal(mu_q, logsig_q)
        z = distr.sample()
        qs = [distr]
        log_qs = [distr.log_p(z)]
        
        # TODO Normalising flows skipped
        
        # Prior for top-level z
        distr = Normal(mu=torch.zeros_like(z), logsig=torch.zeros_like(z))
        ps = [distr]
        log_ps = [distr.log_p(z)]
        
        idx_dec = 0
        # [1, width, height]
        x = self.top_prior.unsqueeze(0)
        # [batch_size, 1, width, height]
        x = x.expand(batch_size, -1, -1, -1)
        
        for cell in self.tower:
            if isinstance(cell, DecoderCombinerCell):
                if idx_dec > 0:
                    # Sample prior
                    latent_repr_p = self.samplers[idx_dec - 1](x)
                    mu_p, logsig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    # Approximate posterior
                    comb_feats = enc_combiner_cells[idx_dec - 1](xs[idx_dec - 1], x)
                    latent_repr_q = enc_samplers[idx_dec](comb_feats)
                    mu_q, logsig_q = torch.chunk(latent_repr_q, 2, dim=1)
                    # Residual distribution
                    distr = Normal(mu_p + mu_q, logsig_p + logsig_q)
                    z = distr.sample()
                    
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
