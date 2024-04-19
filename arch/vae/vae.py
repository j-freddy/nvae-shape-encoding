import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arch.vae.decoder import Decoder
from arch.vae.encoder import Encoder

class VAE(L.LightningModule):
    """
    Variational Autoencoder (VAE) for the ACDC dataset. Following the
    architecture proposed in [1]. Note that [1] uses segmentation maps of size
    256x256 instead of 128x128 and a 32-dim latent space.
    
    [1]: Painchaud N, Skandarani Y, Judge T, Bernard O, Lalande A, Jodoin PM.
    Cardiac segmentation with strong anatomical guarantees. IEEE transactions on
    medical imaging. 2020 Jun 17;39(11):3703-13.
    """
    
    def __init__(self, latent_dim: int=2, beta: float=1.0):
        super().__init__()
        
        self.save_hyperparameters()
        
        # TODO noqa
        # 3 segmentation classes + background
        in_channels = 4
        
        self.encoder = Encoder(in_channels, self.hparams.latent_dim)
        self.decoder = Decoder(in_channels, self.hparams.latent_dim)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=6e-5, weight_decay=1e-2)
    
    def loss(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        x: torch.Tensor,
        x_hat: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        
        kl_div = -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()
        
        return recon_loss + self.hparams.beta * kl_div
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input tensor, return its latent representation z by passing it
        through the encoder.
        """
        mu, logvar = self.encoder(x)
        return self.reparameterise(mu, logvar)
    
    def discretise(self, x_hat: torch.Tensor) -> torch.Tensor:
        """
        Given a probablistic segmentation map, round each pixel to the nearest
        class and return the non-probablistic map.
        """
        x_hat_argmax = torch.argmax(x_hat, dim=1)
        x_hat_hard = F.one_hot(
            x_hat_argmax.long(),
            num_classes=len(x_hat_argmax.unique())
        ).permute(0, 3, 1, 2)
        
        return x_hat_hard
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        x_hat = self.decoder(mu, logvar)
        return mu, logvar, x_hat
    
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, x_hat = self(x)
        
        # Compute loss
        loss = self.loss(mu, logvar, x, x_hat)
        self.log("train_loss", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, x_hat = self(x)
        
        # Compute loss
        loss = self.loss(mu, logvar, x, x_hat)
        self.log("val_loss", loss)
        
        print(f"Val loss: {loss}")
