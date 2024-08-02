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
        
        # We train 2 models
        # 1. Baseline U-Net with cross-entropy loss
        # 2. ACUNet with shape prior loss term only (alpha=0)
        # The shape prior factor constant is how much larger the cross-entropy
        # loss is compared to the L2 shape prior loss, averaged over the
        # entirety of training time
        self.shape_prior_factor = 974970
    
    def loss(
        self,
        y: torch.Tensor,
        y_hat_logits: torch.Tensor,
        log_components: bool=True,
    ):
        recon_loss = self.reconstruction_loss(y, y_hat_logits)
        
        l2_losses = []
        
        zs = self.nvae.get_latent(y)
        zs_hat = self.nvae.get_latent(F.softmax(y_hat_logits, dim=1))

        for z, z_hat in zip(zs, zs_hat):
            assert z.shape == z_hat.shape
            l2_loss = F.mse_loss(z, z_hat)
            l2_losses.append(l2_loss)
        
        l2_losses = torch.stack(l2_losses)
        shape_prior_loss = l2_losses.mean()
        
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
        
        return self.hparams.alpha * recon_loss + self.shape_prior_factor * shape_prior_loss
