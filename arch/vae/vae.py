import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class VAE(L.LightningModule):
    """
    Variational Autoencoder (VAE) for the ACDC dataset. Following the
    architecture proposed in [1]. Note that [1] uses segmentation maps of size
    256x256 instead of 128x128 and a 32-dim latent space.
    
    [1]: Painchaud N, Skandarani Y, Judge T, Bernard O, Lalande A, Jodoin PM.
    Cardiac segmentation with strong anatomical guarantees. IEEE transactions on
    medical imaging. 2020 Jun 17;39(11):3703-13.
    """
    
    def __init__(self, latent_dim: int=2):
        super().__init__()
        
        self.save_hyperparameters()
        
        # TODO noqa
        # 3 segmentation classes + background
        in_channels = 4
        
        self.encoder = nn.Sequential(
            # 3x128x128 -> 48x64x64
            nn.Conv2d(in_channels, 48, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 48x64x64 -> 48x64x64
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 48x64x64 -> 96x32x32
            nn.Conv2d(48, 96, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 96x32x32 -> 96x32x32
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 96x32x32 -> 192x16x16
            nn.Conv2d(96, 192, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 192x16x16 -> 192x16x16
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 192x16x16 -> 384x8x8
            nn.Conv2d(192, 384, kernel_size=3, stride=2, padding=1),
            nn.ELU(),
            # 384x8x8 -> 384x8x8
            nn.Conv2d(384, 384, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            nn.Flatten(),
            nn.Linear(384*8*8, self.hparams.latent_dim * 2),
        )
        
        self.decoder = nn.Sequential(
            # latent_dim -> 384x8x8
            nn.Linear(self.hparams.latent_dim, 384*8*8),
            nn.Unflatten(1, (384, 8, 8)),
            # 384x8x8 -> 192x16x16
            nn.ConvTranspose2d(384, 192, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 192x16x16 -> 192x16x16
            nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 192x16x16 -> 96x32x32
            nn.ConvTranspose2d(192, 96, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 96x32x32 -> 96x32x32
            nn.Conv2d(96, 96, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 96x32x32 -> 48x64x64
            nn.ConvTranspose2d(96, 48, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 48x64x64 -> 48x64x64
            nn.Conv2d(48, 48, kernel_size=3, stride=1, padding=1),
            nn.ELU(),
            # 48x64x64 -> 1x128x128
            nn.ConvTranspose2d(48, in_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ELU(),
            # 3x128x128 -> 3x128x128
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1),
            nn.Softmax(dim=1),
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=6e-5, weight_decay=1e-2)
        
    def _reparametrise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z_m = mu(x_m) + sigma(x_m) * epsilon
        # epsilon ~ N(0, 1)
        
        eps = torch.randn_like(logvar)
        return mu + torch.exp(0.5 * logvar) * eps
    
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
        
        return recon_loss + kl_div
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent_repr: torch.Tensor = self.encoder(x)
        mu, logvar = torch.chunk(latent_repr, 2, dim=1)
        
        z = self._reparametrise(mu, logvar)
        x_hat = self.decoder(z)
        
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
