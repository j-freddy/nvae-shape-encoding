import math
import torch
import torch.nn as nn
import torchvision.ops as ops

class EncoderResidualCell(nn.Module):
    """
    Encoder residual cell.
    
    Implementation as described by Fig. 3b in the NVAE paper. BatchNorm
    momentum is set to 0.05 in official NVAE implementation.
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

class ExtendedEncoderCombinerCell(nn.Module):
    """
    Extended encoder combiner cell.
    
    This class has 2 convnets and allows 3 tensors to be combined.
    """
    
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.net1 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.net2 = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor, x3: torch.Tensor) -> torch.Tensor:
        return x1 + self.net1(x2) + self.net2(x3)

class Encoder(nn.Module):
    """
    NVAE Encoder.
    
    Implementation is adapted from this diagram:
    - https://github.com/NVlabs/NVAE/blob/master/img/model_diagram.png
    
    Also see init_encoder_tower() in official NVAE code.
    
    Extended to support ExtendedEncoderCombinerCell for Conditional NVAE. See
    CNVAE class.
    """
    
    def __init__(
        self,
        num_groups_per_layer: list[int],
        # 1-to-1 map with num_groups_per_layer
        is_layer_shared: list[bool],
        initial_channels: int=64,
        min_channels: int=16,
        z_channels: int=20,
        initial_downsample_factor: int=2,
        use_extended_combiner: bool=False,
    ):
        super().__init__()
        
        assert len(num_groups_per_layer) == len(is_layer_shared)
        
        self.min_channels = min_channels
        self.num_latent_layers = len(num_groups_per_layer)
        
        CombinerCell = ExtendedEncoderCombinerCell \
            if use_extended_combiner \
            else EncoderCombinerCell
        
        # Build preprocessing modules
        
        # In official NVAE implementation, by default arch_type is 'res_mbconv'
        # and so 'down_pre' is ['res_bnswish', 'res_bnswish'] with 2 preprocess
        # cells, 1 preprocess block and channel multiplier of 1.
        
        preprocess_modules = []
        num_preprocess_layers = int(math.log2(initial_downsample_factor))
        
        num_channels = initial_channels
        
        for _ in range(num_preprocess_layers):
            true_num_channels = self._num_channels(num_channels)
            
            # Inverted residual cells
            preprocess_modules.append(EncoderResidualCell(true_num_channels))
            preprocess_modules.append(EncoderResidualCell(true_num_channels))
            
            # Downsample
            preprocess_modules.append(
                nn.Conv2d(
                    true_num_channels,
                    self._num_channels(num_channels * 2),
                    kernel_size=3,
                    stride=2,
                    padding=1,
                    bias=False,
                )
            )
            
            num_channels *= 2
        
        self.preprocess = nn.Sequential(*preprocess_modules)
        
        # Build tower
        
        self.tower = nn.ModuleList()
        self.samplers = nn.ModuleList()
        
        for s in range(self.num_latent_layers):
            for g in range(num_groups_per_layer[s]):
                true_num_channels = self._num_channels(num_channels)
                
                # Inverted residual cells
                self.tower.append(EncoderResidualCell(true_num_channels))
                self.tower.append(EncoderResidualCell(true_num_channels))
                
                if is_layer_shared[s]:
                    # Add sampler
                    self.samplers.append(
                        nn.Conv2d(
                            true_num_channels,
                            2 * z_channels,
                            kernel_size=3,
                            padding=1,
                        )
                    )
                    
                    # Add enc combiner if not last group in last layer
                    if not (s == self.num_latent_layers - 1 and g == num_groups_per_layer[s] - 1):
                        self.tower.append(CombinerCell(true_num_channels, true_num_channels))
        
            if s < self.num_latent_layers - 1:
                # Downsample
                self.tower.append(
                    nn.Conv2d(
                        true_num_channels,
                        self._num_channels(num_channels * 2),
                        kernel_size=3,
                        stride=2,
                        padding=1,
                        bias=False,
                    ),
                )
                
                num_channels *= 2
      
        # Build compressor
        true_num_channels = self._num_channels(num_channels)
        
        self.compressor = nn.Sequential(
            nn.ELU(),
            nn.Conv2d(true_num_channels, true_num_channels, kernel_size=1, bias=True),
            nn.ELU(),
        )
    
    def _num_channels(self, num_channels: int) -> int:
        return max(num_channels, self.min_channels)
    
    def forward(
        self,
        x: torch.Tensor,
        print_logs: bool=False,
    ) -> tuple[torch.Tensor, torch.Tensor, list[torch.Tensor], list[EncoderCombinerCell | ExtendedEncoderCombinerCell]]:
        x = self.preprocess(x)
        if print_logs:
            print(x.shape)

        xs = []
        combiner_cells = []
        
        if print_logs:
            print(self.tower)
        
        # Go through the tower and checkpoint combiner cells as it requires
        # sampled variables in the decoder pass
        for cell in self.tower:
            if isinstance(cell, EncoderCombinerCell) or isinstance(cell, ExtendedEncoderCombinerCell):
                xs.append(x)
                combiner_cells.append(cell)
            else:
                x = cell(x)

        x = self.compressor(x)
        
        # Final x is not added as last group in last layer does not have a
        # combiner cell
        
        if print_logs:
            print("Printing xs and final x...")

            for x_buf in xs:
                print(x_buf.shape)
            print(x.shape)

            print("End of encoder.")
        
        return x, xs, combiner_cells
