import lightning as L
import torch
import torch.nn as nn

from nvae.decoder import Decoder
from nvae.encoder import Encoder

class NVAE(L.LightningModule):
    """
    arch_cells = dict()
    arch_cells['normal_enc'] = ['res_bnswish', 'res_bnswish']
    arch_cells['down_enc'] = ['res_bnswish', 'res_bnswish']
    arch_cells['normal_dec'] = ['mconv_e6k5g0']
    arch_cells['up_dec'] = ['mconv_e6k5g0']
    arch_cells['normal_pre'] = ['res_bnswish', 'res_bnswish']
    arch_cells['down_pre'] = ['res_bnswish', 'res_bnswish']
    arch_cells['normal_post'] = ['mconv_e3k5g0']
    arch_cells['up_post'] = ['mconv_e3k5g0']
    arch_cells['ar_nn'] = ['']
    """
    
    def __init__(self, initial_channels: int=64):
        super().__init__()
        
        # Table 6: # initial channels in enc. (NVAE paper)
        self.stem = nn.Conv2d(3, initial_channels, kernel_size=3, padding=1, bias=True)
        
        self.encoder = Encoder(initial_channels=initial_channels)
        # TODO In encoder.py I use num_latent_scales = 3
        # In general, initial_channels = initial_channels * (2 ** (num_latent_scales - 1))
        self.decoder = Decoder(initial_channels=initial_channels * 4)
    
    def configure_optimizers(self):
        NotImplemented
    
    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        # TODO Official NVAE implementation uses s = self.stem(2 * x - 1.0)
        x = self.stem(feats)
        
        # Pass through encoder
        x, xs, enc_combiner_cells = self.encoder(x)
        
        # Reverse buffers for decoder
        xs = xs[::-1]
        enc_combiner_cells = enc_combiner_cells[::-1]
        
        print(x.shape)
        
        for x in xs:
            print(x.shape)
        
        import sys
        sys.exit()
        
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        feats, labels = batch
        self(feats)
        
        NotImplemented
    
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        NotImplemented
