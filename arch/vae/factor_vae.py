import torch
import torch.nn as nn
import torch.nn.functional as F

from arch.vae.vae import VAE

class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = 512
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.hidden_dim, 2),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
    
# https://github.com/AliLotfi92/Disentangling_by_Factorising/blob/master/FactorizedVAE.py

class FactorVAE(VAE):
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        beta: float=1.0,
    ):
        super().__init__(in_channels, latent_dim, beta)
        
        # "For advanced research topics like reinforcement learning, sparse
        # coding, or GAN research, it may be desirable to manually manage the
        # optimization process, especially when dealing with multiple optimizers
        # at the same time."
        # 
        # https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html#manual-optimization
        self.automatic_optimization = False
        
        self.discriminator = Discriminator(latent_dim)
        
    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        kl_div = self._kl_divergence(mu, logvar)
        
        pred: torch.Tensor = self.discriminator(z)
        tc = (pred[:, :1] - pred[:, 1:]).mean()
        
        return recon_loss + kl_div + (1 - self.hparams.beta) * tc

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        # TODO Calculate the loss to train the Discriminator
        # Check that self.manual_backward and opt.step does not update the
        # Discriminator
        
        opt = self.optimizers()
        opt.zero_grad()
        
        mu, logvar, z, x_hat = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat)
        self.log("train_loss", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")
    
        self.manual_backward(loss)
        opt.step()

        return loss
