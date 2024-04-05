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
    NotImplemented
