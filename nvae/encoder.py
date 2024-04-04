import torch
import torch.nn as nn
import torchvision.ops as ops

class EncoderResidualCell(nn.Module):
    """
    Encoder residual cell.
    
    Implementation as described by Fig. 3b in the NVAE paper.
    """
    
    def __init__(self, num_channels: int):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=num_channels, eps=1e-5, momentum=0.05),
            nn.SiLU(),
            nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1),
            ops.SqueezeExcitation(
                num_channels,
                # Following the official NVAE implementation from class SE in
                # neural_operations.py
                squeeze_channels=max(num_channels // 16, 4),
            ),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # TODO hparams
        return x + 0.1 * self.net(x)

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
    def __init__(self):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            # TODO num_channels
            EncoderResidualCell(3),
            EncoderResidualCell(3),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(x)
        
        print(x.shape)
        
        import sys
        sys.exit()
        
        return NotImplemented
