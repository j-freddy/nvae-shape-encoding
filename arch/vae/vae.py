import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from arch.vae.decoder import Decoder
from arch.vae.encoder import Encoder
from utils import frechet_inception_distance, show_samples

class VAE(L.LightningModule):
    """
    Variational Autoencoder (VAE) for the ACDC dataset. Following the
    architecture proposed in [1]. Note that [1] uses segmentation maps of size
    256x256 instead of 128x128 and a 32-dim latent space.
    
    [1]: Painchaud N, Skandarani Y, Judge T, Bernard O, Lalande A, Jodoin PM.
    Cardiac segmentation with strong anatomical guarantees. IEEE transactions on
    medical imaging. 2020 Jun 17;39(11):3703-13.
    """
    
    def __init__(self, in_channels: int=4, latent_dim: int=2, beta: float=1.0):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = Encoder(self.hparams.in_channels, self.hparams.latent_dim)
        self.decoder = Decoder(self.hparams.in_channels, self.hparams.latent_dim)

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
        return self.decoder.reparameterise(mu, logvar)
    
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
    
    def test_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        assert batch_idx == 0, "Only 1 batch allowed"

        # Compute loss
        mu, logvar, x_hat = self(x)
        loss = self.loss(mu, logvar, x, x_hat)
        self.log("test_loss", loss)

        self.log_reconstructions(x[:20])
        self.log_generations_and_fid(x)
        self.log_lerp(x[:20])
    
    def log_reconstructions(self, x: torch.Tensor):
        _, _, x_hat = self(x)

        reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)
        samples = torch.argmax(x, dim=1).unsqueeze(1)

        # Interleave samples and reconstructions
        batch_size, num_channels, width, height = samples.shape
        assert width == height
        samples_and_reconstructions = torch.empty(batch_size * 2, num_channels, width, height)
        
        for i in range(samples.shape[0]):
            samples_and_reconstructions[i * 2] = samples[i]
            samples_and_reconstructions[i * 2 + 1] = reconstructions[i]
        
        show_samples(samples_and_reconstructions, rgb=False, nrow=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())
    
    def log_generations_and_fid(self, x: torch.Tensor):
        num_samples, _, _, _ = x.shape
            
        # Sample from latent space
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        
        # Generate prbabilistic segmentation maps from latent variables
        x_fake: torch.Tensor = self.decoder.net(z)

        # Discretise prbabilistic map then view generations
        generations = torch.argmax(x_fake[:40], dim=1).unsqueeze(1)
        show_samples(generations, rgb=False, nrow=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/generations", plt.gcf())
        
        fid_value = frechet_inception_distance(x, self.discretise(x_fake))
        self.log("fid", fid_value)
    
    def log_lerp(self, x: torch.Tensor):
        """
        Linearly interpolate between the latent representations of two samples,
        then visualise the reconstructions.
        """
        z = self.get_latent(x)
        
        # TODO noqa
        # Hand pick 2 masks that look different
        z1, z2 = z[1], z[19]
        
        # Linear interpolation between z1 and z2
        z_lerps = []

        for i in range(10):
            z_lerps.append(torch.lerp(z1, z2, i / 9))
        
        z_lerps = torch.stack(z_lerps)
        
        # Pass through decoder
        x_hat: torch.Tensor = self.decoder.net(z_lerps)
        
        reconstructions = torch.argmax(x_hat, dim=1).unsqueeze(1)

        show_samples(reconstructions, rgb=False, nrow=10, figsize=(10, 1), display=False)
        self.logger.experiment.add_figure("img/lerp", plt.gcf())
