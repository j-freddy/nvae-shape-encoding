import torch
import torch.nn as nn
import torchvision.ops as ops

class EncoderResidualCell(nn.Module):
    """
    Encoder residual cell.
    
    Implementation as described by Fig. 3b in the NVAE paper. BatchNorm momentum
    is set to 0.05 in official NVAE implementation.
    """
    
    def __init__(self, num_channels: int):
        super().__init__()
        
        self.net = nn.Sequential(
            # BN + Swish
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            # Conv 3x3
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
            # BN + Swish
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            # Conv 3x3
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1, bias=False),
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

class EncoderCombinerCell(nn.Module):
    """
    Encoder combiner cell.
    
    Following the official NVAE implementation from class EncCombinerCell in
    neural_operations.py
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        
        self.net = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return x1 + self.net(x2)

class Encoder(nn.Module):
    """
    NVAE Encoder.
    
    Implementation as described by the diagram: -
    https://github.com/NVlabs/NVAE/blob/master/img/model_diagram.png
    
    For alignment, the 2nd InvertedResidual (i.e. EncoderResidualCell) in
    preprocess shown in the diagram is lifted to the start of the tower.
    """
    
    def __init__(self, num_channels: int=20):
        super().__init__()
        
        self.preprocess = EncoderResidualCell(3)
        
        # NVAE paper: Table 6
        self.tower = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(
                    3 if i == 0 else num_channels,
                    num_channels,
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                ),
                EncoderResidualCell(num_channels),
                EncoderResidualCell(num_channels),
            )
            # TODO I am using only 1 group per scale. The official paper uses
            # more (see # groups in each scale) in Table 6.
            for i in range(3)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        
        xs = []
        
        for layer in self.tower:
            x = layer(x)
            xs.append(x)
        
        return xs
