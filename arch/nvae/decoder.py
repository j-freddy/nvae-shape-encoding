import torch
import torch.nn as nn
import torchvision.ops as ops

from arch.nvae.distribution import Normal
from utils import soft_clamp

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
        self.top_prior = nn.Parameter(torch.rand(size=(256, 8, 8)), requires_grad=True)
        
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

    def forward(
        self,
        x: torch.Tensor,
        xs: torch.Tensor,
        enc_combiner_cells: list[nn.Module],
        enc_samplers: list[nn.Module],
    ) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        
        # Sample mu, logsig of the topmost latent scale
        latent_repr_q = enc_samplers[0](x)
        mu_q, logsig_q = torch.chunk(latent_repr_q, 2, dim=1)
        mu_q = soft_clamp(mu_q)
        logsig_q = soft_clamp(logsig_q)
        
        # Approximate posterior for top-level
        distr = Normal(mu_q, logsig_q)
        z = distr.sample()
        log_qs = [distr.log_p(z)]
        
        # TODO Normalising flows skipped
        
        # Prior for top-level z
        distr = Normal(mu=torch.zeros_like(z), logsig=torch.zeros_like(z))
        
        distrs = [distr]
        log_ps = [distr.log_p(z)]
        
        idx_dec = 0
        # [1, width, height]
        x = self.top_prior.unsqueeze(0)
        # [batch_size, 1, width, height]
        x = x.expand(batch_size, -1, -1, -1)
        
        for cell in self.tower:
            print(cell)
            
            if isinstance(cell, DecoderCombinerCell):
                print("Foo")
                
                if idx_dec > 0:
                    print("Bar")
                    
                    # Prior
                    latent_repr_p = self.dec_samplers[idx_dec - 1](x)
                    mu_p, log_sig_p = torch.chunk(latent_repr_p, 2, dim=1)
                    
                    # TODO
                    
                    # Encoder
                    # ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
                    # param = self.enc_sampler[idx_dec](ftr)
                    # mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
                    # dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
                    # z, _ = dist.sample()
                    # log_q_conv = dist.log_p(z)
                    
                    # all_log_q.append(log_q_conv)
                    # all_q.append(dist)
                    
                    # Evaluate log_p(z)
                    # dist = Normal(mu_p, log_sig_p)
                    # log_p_conv = dist.log_p(z)
                    # all_p.append(dist)
                    # all_log_p.append(log_p_conv)

                print(x.shape)
                print(z.shape)

                x = cell(x, z)
                
                print(x.shape)
                
                idx_dec += 1
            else:
                x = cell(x)
                print("Baz")
                print(x.shape)
                
                import sys
                sys.exit()
        
        # TODO From line 424 in model.py, official NVAE implementation
        
        return NotImplemented
