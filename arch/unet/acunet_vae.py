import torch
from arch.unet.unet import UNet
from arch.vae.vae import VAE
from utils.const import VAE_MODEL_PATH

import torch.nn.functional as F

class ACVAEUNet(UNet):
    """
    ACU-Net baseline: Use VAE instead of NVAE
    
    See ACUNet docstring for more information.
    """
    
    def __init__(
        self,
        in_channels: int=1,
        out_channels: int=4,
        loss_reg: str="shape_prior",
        vae_path: str=VAE_MODEL_PATH,
        alpha: float=1.0,
        model_type: str="acunet_vae",
    ):
        super().__init__(in_channels, out_channels, loss_reg)
        
        self.save_hyperparameters()
        
        self.hparams.update({"model_type": "unet"})
        
        self.vae = VAE.load_from_checkpoint(vae_path)
        self.vae.freeze()
        
        # TODO
        self.shape_prior_factor = 1
    
    def loss(
        self,
        y: torch.Tensor,
        y_hat_logits: torch.Tensor,
        log_components: bool=True,
    ):
        recon_loss = self.reconstruction_loss(y, y_hat_logits)
        
        l2_losses = []
        
        zs = self.vae.get_latent(y)
        zs_hat = self.vae.get_latent(F.softmax(y_hat_logits, dim=1))

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
