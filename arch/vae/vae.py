import lightning as L
from matplotlib import pyplot as plt
from monai.losses.dice import DiceLoss
import torch
import torch.nn.functional as F
import torch.optim as optim

from arch.vae.decoder import Decoder
from arch.vae.encoder import Encoder
from const import FRDS_MODEL_PATH
from utils.anatomical_validity_checker import AnatomicalValidityChecker
from utils.eval import compute_frds, get_samples_and_reconstructions_pixel_diff
from utils.utils import discretise, show_samples

class VAE(L.LightningModule):
    """
    Single-layer variational autoencoder (VAE) for the ACDC dataset. Encoder and
    decoder architecture is adapted from the architecture proposed by [1]. This
    class implements the beta-VAE regulariser term proposed by [2]. Standard
    Gaussian prior is assumed.
    
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
        
        # To keep track of test set and generated samples during test time, to
        # compute FRDS
        self.x_buffer: list[torch.Tensor] = []
        self.discretised_x_fakes_buffer: list[torch.Tensor] = []

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
        """
        Compute the reconstruction loss using cross-entropy.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input
                (output of decoder).
        
        Returns:
            recon_loss (torch.Tensor): Reconstruction loss.
        """
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
        """
        Compute the beta-VAE loss: sum of reconstruction loss and beta-weighted
        KL divergence regulariser term.
        """
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        kl_div = self._kl_divergence(mu, logvar)

        weighted_kl_div = self.hparams.beta * kl_div
        
        if log_components:
            marginal_kl_div = self._kl_divergence(mu, logvar, marginal=True)
            
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", weighted_kl_div)
            for i, marginal_kl in enumerate(marginal_kl_div):
                self.log(f"loss/marginal_kl_div/dim_{i}", marginal_kl)
        
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
        mu, logvar = self.encoder(2 * x - 1.0)
        return self._reparameterise(mu, logvar)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass through the VAE encoder and decoder.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
        
        Returns:
            mu (torch.Tensor): Mean of approximate posterior.
            logvar (torch.Tensor): Log-variance of approximate posterior.
            z (torch.Tensor): Latent representation of input.
            x_hat_logits (torch.Tensor): Logits of reconstruction of input.
        """
        mu, logvar = self.encoder(2 * x - 1.0)
        z = self._reparameterise(mu, logvar)
        x_hat_logits = self.decoder(z)
        return mu, logvar, z, x_hat_logits
    
    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits)
        self.log("loss/train", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        return loss
    
    def validation_step(self, x: torch.Tensor) -> torch.Tensor:
        mu, logvar, z, x_hat_logits = self(x)
        
        # Compute loss
        loss = self.loss(x, mu, logvar, z, x_hat_logits, log_components=False)
        self.log("loss/val", loss)
        
        print(f"Val loss: {loss}")
    
    def test_step(self, x: torch.Tensor, batch_idx: int) -> torch.Tensor:
        self.log_reconstruction_metrics(x)
        self.log_generation_metrics(x)
    
    def log_reconstruction_metrics(self, x: torch.Tensor):
        """
        Log reconstruction metrics to TensorBoard. This includes average
        reconstruction loss and Dice score across the batch.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
        """
        _, _, _, x_hat_logits = self(x)
        
        # Compute reconstruction loss
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        self.log("loss/test_recon", recon_loss)
        
        # Compute Dice score
        x_hat = torch.softmax(x_hat_logits, dim=1)
        x_hat_onehot = discretise(x_hat)
        dl = DiceLoss(reduction="mean", include_background=False)
        dice_score = 1 - dl(input=x_hat_onehot, target=x)
        self.log("loss/dsc", dice_score)
    
    def log_generation_metrics(self, x: torch.Tensor):
        """
        Log generation metrics to TensorBoard. This includes the Frechet Resnet
        Distance with SimCLR (FRDS) metric across the batch.
        
        Args:
            x (torch.Tensor): One-hot encoded input segmentations.
        """
        num_samples, _, _, _ = x.shape

        # Sample from latent space
        z = torch.randn(num_samples, self.hparams.latent_dim).to(self.device)
        
        # Generate probabilistic segmentation maps from latent variables
        x_fake_logits: torch.Tensor = self.decoder(z)
        
        discretised_x_fakes = discretise(x_fake_logits)
        
        # Percentage of anatomically valid generations
        num_valid = 0
        
        for discretised_x_fake in discretised_x_fakes:
            AV = AnatomicalValidityChecker(discretised_x_fake)
            if AV.count_violations() == 0:
                num_valid += 1
        
        self.log("gen/anatomically_valid", num_valid / num_samples)
        
        # Keep track of all samples, to compute FRDS
        self.x_buffer.append(x)
        self.discretised_x_fakes_buffer.append(discretised_x_fakes)
    
    def on_test_end(self):
        x = torch.cat(self.x_buffer, dim=0)
        discretised_x_fakes = torch.cat(self.discretised_x_fakes_buffer, dim=0)
        
        frds_value = compute_frds(
            x,
            discretised_x_fakes,
            resnet_path=FRDS_MODEL_PATH,
            device=self.device,
        )

        print(f"FRDS: {frds_value}")
        self.logger.experiment.add_scalar("gen/frds", frds_value, 0)
