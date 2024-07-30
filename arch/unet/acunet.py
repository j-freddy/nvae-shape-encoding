import torch
from arch.nvae.nvae import NVAE
from arch.unet.unet import UNet
from utils.const import ACDC, NVAE_MODEL_PATH

import torch.nn.functional as F

class ACUNet(UNet):
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        loss_reg: str="shape_prior",
        nvae_path: str=NVAE_MODEL_PATH,
        alpha: float=1.0,
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
        log_components: bool=True,
    ):
        recon_loss = self.reconstruction_loss(y, y_hat_logits)
        
        l2_losses = []
        
        zs = self.nvae.get_latent(y)
        zs_hat = self.nvae.get_latent(y_hat_logits)
        
        print(zs[3:7].shape)
        
        
        print(zs[0].shape)
        print(zs[1].shape)
        print(zs[2].shape)
        print(zs[3].shape)
        print(zs[4].shape)
        
        import sys
        sys.exit()
        
        import sys
        sys.exit()

        for z, z_hat in zip(zs, zs_hat):
            assert z.shape == z_hat.shape
            
            batch_size, z_channels, w, h = z.shape
            # latent_size = batch_size * z_channels * w * h
            latent_size = batch_size
            
            # Since CE is averaged over batch but not mask size, let's multiply
            # by number of pixels in mask
            mask_size = ACDC.WIDTH * ACDC.WIDTH
            
            l2_loss = F.mse_loss(z, z_hat) / latent_size * mask_size
            l2_losses.append(l2_loss)
        
        shape_prior_loss = sum(l2_losses)
        
        print(f"Reconstruction loss: {recon_loss}")
        print(f"Shape prior loss: {shape_prior_loss}")
        
        if log_components:
            self.log("loss/recon", recon_loss)
            self.log("loss/shape_prior", shape_prior_loss)
            
            num_groups = len(l2_losses)
            
            for i, l2_loss in enumerate(l2_losses):
                # Shallowest group is 0
                # Topmost group is 6
                group_idx = num_groups - i - 1
                self.log(f"loss/shape_prior/dim_{group_idx}", l2_loss)

        return self.hparams.alpha * recon_loss + shape_prior_loss
