import lightning as L
import torch

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
    
    def __init__(self):
        super().__init__()
        
        self.encoder = Encoder()
        self.decoder = Decoder()
    
    def configure_optimizers(self):
        NotImplemented
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        NotImplemented
        
    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        feats, labels = batch

        print(feats.shape)
        print(labels.shape)
        
        import sys
        sys.exit()
        
        NotImplemented
    
    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_idx: int,
    ) -> torch.Tensor:
        NotImplemented
