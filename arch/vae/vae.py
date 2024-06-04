import lightning as L
from matplotlib import pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim

from arch.vae.decoder import Decoder
from arch.vae.encoder import Encoder
from utils.eval import frds, get_samples_and_reconstructions
from utils.utils import discretise, show_samples

class VAE(L.LightningModule):
    """
    Variational Autoencoder (VAE) for the ACDC dataset. Following the
    architecture proposed in [1]. Note that [1] uses segmentation maps of size
    256x256 instead of 128x128 and a 32-dim latent space.
    
    This class implements the beta-VAE regulariser term based on [2].
    
    [1]: Painchaud N, Skandarani Y, Judge T, Bernard O, Lalande A, Jodoin PM.
    Cardiac segmentation with strong anatomical guarantees. IEEE transactions on
    medical imaging. 2020 Jun 17;39(11):3703-13.
    
    [2]: Higgins I, Matthey L, Pal A, Burgess CP, Glorot X, Botvinick MM,
    Mohamed S, Lerchner A. beta-vae: Learning basic visual concepts with a
    constrained variational framework. ICLR (Poster). 2017 Apr 24;3.
    """
    
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="beta_vae",
        beta: float=1.0,
        gamma: float=1.0,
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.encoder = Encoder(self.hparams.in_channels, self.hparams.latent_dim)
        self.decoder = Decoder(self.hparams.in_channels, self.hparams.latent_dim)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=6e-5, weight_decay=1e-2)
    
    def _kl_divergence(
        self,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        marginal: bool=False,
    ) -> torch.Tensor:
        if marginal:
            return -0.5 * torch.mean(
                1 + logvar - mu.pow(2) - logvar.exp(),
                dim=0,
            )
        
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()
        
    def reconstruction_loss(self, x: torch.Tensor, x_hat_logits: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        return F.cross_entropy(x_hat_logits, x, reduction="sum") / batch_size
    
    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat_logits: torch.Tensor,
        log_components: bool=True,
    ) -> torch.Tensor:
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        kl_div = self._kl_divergence(mu, logvar)

        weighted_kl_div = self.hparams.beta * kl_div
        
        if log_components:
            marginal_kl_div = self._kl_divergence(mu, logvar, marginal=True)
            
            self.log("recon_loss", recon_loss)
            self.log("kl_div", weighted_kl_div)
            for i, marginal_kl in enumerate(marginal_kl_div):
                self.log(f"marginal_kl_div/dim_{i}", marginal_kl)
        
        return recon_loss + weighted_kl_div
    
    def _reparameterise(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # z_m = mu(x_m) + sigma(x_m) * epsilon
        # epsilon ~ N(0, 1)
        
        eps = torch.randn_like(logvar)
        return mu + torch.exp(0.5 * logvar) * eps
    
    def get_latent(self, x: torch.Tensor) -> torch.Tensor:
        """
        Given an input tensor, return its latent representation z by passing it
        through the encoder.
        """
        mu, logvar = self.encoder(x)
        return self._reparameterise(mu, logvar)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar = self.encoder(x)
        z = self._reparameterise(mu, logvar)
        x_hat_logits = self.decoder(z)
        return mu, logvar, z, x_hat_logits
    
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits)
        self.log("train_loss", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits, log_components=False)
        self.log("val_loss", loss)
        
        print(f"Val loss: {loss}")
    
    def test_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        assert batch_idx == 0, "Only 1 batch allowed"

        # Compute loss
        _, _, _, x_hat_logits = self(x)
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        self.log("test_recon_loss", recon_loss)

        self.log_reconstructions(x[:20])
        self.log_generations_and_frds(x)
        self.log_lerp(x[:20])
    
    def log_reconstructions(self, x: torch.Tensor):
        _, _, _, x_hat_logits = self(x)

        samples_and_reconstructions = get_samples_and_reconstructions(x, x_hat_logits)
        show_samples(samples_and_reconstructions, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/reconstructions", plt.gcf())
    
    def log_generations_and_frds(self, x: torch.Tensor):
        num_samples, _, _, _ = x.shape

        # Sample from latent space
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        
        # Generate probabilistic segmentation maps from latent variables
        x_fake_logits: torch.Tensor = self.decoder(z)

        # Discretise probabilistic map then view generations
        generations = torch.argmax(x_fake_logits[:40], dim=1).unsqueeze(1)
        show_samples(generations, rgb=False, ncol=10, figsize=(10, 4), display=False)
        self.logger.experiment.add_figure("img/generations", plt.gcf())
        
        frds_value = frds(
            x,
            discretise(x_fake_logits),
            device=self.device,
        )

        self.log("frds", frds_value)
    
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
        x_hat_logits: torch.Tensor = self.decoder(z_lerps)
        
        reconstructions = torch.argmax(x_hat_logits, dim=1).unsqueeze(1)

        show_samples(reconstructions, rgb=False, ncol=10, figsize=(10, 1), display=False)
        self.logger.experiment.add_figure("img/lerp", plt.gcf())
