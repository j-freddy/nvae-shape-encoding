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
    """
    
    def __init__(self, initial_channels: int=64):
        super().__init__()
        
        self.preprocess = nn.Sequential(
            EncoderResidualCell(initial_channels),
            EncoderResidualCell(initial_channels),
        )
        
        # NVAE paper: Table 6
        num_latent_scales = 3
        # num_groups_per_scale = [20, 10, 5]
        num_groups_per_scale = [4, 2, 1]
        
        self.tower = nn.ModuleList()
        
        num_channels = initial_channels
        
        for s in range(num_latent_scales):
            for g in range(num_groups_per_scale[s]):
                # Inverted residual cells
                self.tower.append(EncoderResidualCell(num_channels))
                self.tower.append(EncoderResidualCell(num_channels))
                
                # Add enc combiner if not last group in last scale
                # TODO Note that 2nd arg is num_channels_dec * multiplier but it
                # is always equal to num_channels_enc
                if not (s == num_latent_scales - 1 and g == num_groups_per_scale[s] - 1):
                    self.tower.append(EncoderCombinerCell(num_channels, num_channels))
        
            if s < num_latent_scales - 1:
                # Downsample
                self.tower.append(
                    nn.Conv2d(
                        num_channels,
                        num_channels * 2,
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                )
                
                num_channels *= 2
    
    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[EncoderCombinerCell]]:
        x = self.preprocess(x)
        
        xs = []
        combiner_cells = []
        
        # Go through the tower and checkpoint combiner cells as it requires
        # sampled variables in the decoder pass
        for cell in self.tower:
            if isinstance(cell, EncoderCombinerCell):
                xs.append(x)
                combiner_cells.append(cell)
            else:
                x = cell(x)
        
        # Final x is not added as last group in last scale does not have a
        # combiner cell

        return x, xs, combiner_cells
