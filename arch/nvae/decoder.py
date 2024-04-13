import torch
import torch.nn as nn
import torchvision.ops as ops

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
    ) -> torch.Tensor:
        NotImplemented
