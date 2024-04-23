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

class Decoder(nn.Module):
    """
    NVAE Decoder.
    
    Implementation as described by the diagram: -
    https://github.com/NVlabs/NVAE/blob/master/img/model_diagram.png
    
    Also see init_decoder_tower() in official NVAE code.
    """
    
    def __init__(self, initial_channels: int=256):
        super().__init__()
        
        # Build tower
        
        self.tower = nn.ModuleList()
        
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
                    self.tower.append(DecoderResidualCell(num_channels))
                    self.tower.append(DecoderResidualCell(num_channels))
                    
                # TODO DecoderCombinerCell
                
                # TODO Add samplers
                # These are just Conv2D with output features = 20 * 2
                # where 20 is #channels in z (see Table 6) and 20 * 2 allows to
                # encode mean and logvar
            
            if s < num_latent_scales - 1:
                # Upsample
                self.tower.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels // 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
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
        # Sample mu, logsig of the topmost latent scale
        latent_repr = enc_samplers[0](x)
        mu, logsig = torch.chunk(latent_repr, 2, dim=1)
        mu = soft_clamp(mu)
        logsig = soft_clamp(logsig)
        
        # Approximate posterior for top-level
        distr = Normal(mu, logsig)
        z = distr.sample()
        log_qs = [distr.log_p(z)]
        
        # TODO Normalising flows skipped
        
        # Prior for top-level z
        distr = Normal(mu=torch.zeros_like(z), logsig=torch.zeros_like(z))
        
        distributions = [distr]
        log_ps = [distr.log_p(z)]
        
        import sys
        sys.exit()
        
        # # To make sure we do not pass any deterministic features from x to decoder.
        # s = 0

        # idx_dec = 0
        # s = self.prior_ftr0.unsqueeze(0)
        # batch_size = z.size(0)
        # s = s.expand(batch_size, -1, -1, -1)
        # for cell in self.dec_tower:
        #     if cell.cell_type == 'combiner_dec':
        #         if idx_dec > 0:
        #             # form prior
        #             param = self.dec_sampler[idx_dec - 1](s)
        #             mu_p, log_sig_p = torch.chunk(param, 2, dim=1)

        #             # form encoder
        #             ftr = combiner_cells_enc[idx_dec - 1](combiner_cells_s[idx_dec - 1], s)
        #             param = self.enc_sampler[idx_dec](ftr)
        #             mu_q, log_sig_q = torch.chunk(param, 2, dim=1)
        #             dist = Normal(mu_p + mu_q, log_sig_p + log_sig_q) if self.res_dist else Normal(mu_q, log_sig_q)
        #             z, _ = dist.sample()
        #             log_q_conv = dist.log_p(z)
        #             # apply NF
        #             for n in range(self.num_flows):
        #                 z, log_det = self.nf_cells[nf_offset + n](z, ftr)
        #                 log_q_conv -= log_det
        #             nf_offset += self.num_flows
        #             all_log_q.append(log_q_conv)
        #             all_q.append(dist)

        #             # evaluate log_p(z)
        #             dist = Normal(mu_p, log_sig_p)
        #             log_p_conv = dist.log_p(z)
        #             all_p.append(dist)
        #             all_log_p.append(log_p_conv)

        #         # 'combiner_dec'
        #         s = cell(s, z)
        #         idx_dec += 1
        #     else:
        #         s = cell(s)
