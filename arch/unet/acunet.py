import torch
from arch.nvae.nvae import NVAE
from arch.unet.unet import UNet
from const import NVAE_MODEL_PATH

import torch.nn.functional as F

class ACUNet(UNet):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        loss_reg: str="shape_prior",
        nvae_path: str=NVAE_MODEL_PATH
    ):
        super().__init__(in_channels, out_channels, loss_reg)
        
        self.save_hyperparameters()
        
        self.nvae = NVAE.load_from_checkpoint(nvae_path)
        self.nvae.freeze()
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-2, weight_decay=0)
    
    def loss(
        self,
        y: torch.Tensor,
        y_hat_logits: torch.Tensor,
    ):
        # recon_loss = self.reconstruction_loss(y, y_hat_logits)
        recon_loss = 0
        
        zs = self.nvae.get_latent(y)
        zs_hat = self.nvae.get_latent(y_hat_logits)

        for z, z_hat in zip(zs, zs_hat):
            recon_loss += F.mse_loss(z, z_hat)

        return recon_loss
