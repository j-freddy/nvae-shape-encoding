import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from arch.vae.vae import VAE

class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = 8
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, 1),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class FactorVAE(VAE):
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="factor_vae",
        beta: float=1.0,
        gamma: float=1.0,
    ):
        super().__init__(in_channels, latent_dim, loss_reg, beta)
        
        self.save_hyperparameters()
        
        # "For advanced research topics like reinforcement learning, sparse
        # coding, or GAN research, it may be desirable to manually manage the
        # optimization process, especially when dealing with multiple optimizers
        # at the same time."
        # 
        # https://lightning.ai/docs/pytorch/stable/model/build_model_advanced.html#manual-optimization
        self.automatic_optimization = False
        
        self.discriminator = Discriminator(latent_dim)
    
    def configure_optimizers(self):
        opt_vae = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=6e-5,
            weight_decay=1e-2,
        )
        
        opt_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=6e-2,
            weight_decay=1e-2,
        )
        
        return [opt_vae, opt_discriminator], []
        
    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat: torch.Tensor,
        train: bool=True,
        return_pred: bool=False,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        # KL divergence between q(z|x) and p(z)
        # Forming part of ELBO
        kl_div = self._kl_divergence(mu, logvar)
        
        pred: torch.Tensor = self.discriminator(z)
        # KL divergence between q(z) and p(z)
        kl_qp = pred.mean()
        # kl_qp = torch.abs(pred.mean())
        
        weighted_kl_div = self.hparams.beta * kl_div
        weighted_kl_qp = self.hparams.gamma * kl_qp
        
        if train:
            self.log("recon_loss", recon_loss)
            self.log("kl_div", kl_div)
            self.log("kl_qp", weighted_kl_qp)
        
        # beta acts as gamma
        loss = recon_loss + weighted_kl_div + weighted_kl_qp
        
        if return_pred:
            return pred, loss
        return loss

    def loss_discriminator(
        self,
        pred: torch.Tensor,
        predp: torch.Tensor,
    ) -> torch.Tensor:
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pred, torch.zeros_like(pred)) +
            F.binary_cross_entropy_with_logits(predp, torch.ones_like(predp))
        )
        
        return loss

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        opt_vae, opt_discriminator = self.optimizers()
        
        # VAE step
        
        self.toggle_optimizer(opt_vae)

        mu, logvar, z, x_hat = self(x)
        pred, loss = self.loss(x, mu, logvar, z, x_hat, return_pred=True)
        self.log("train_loss", loss)
        
        print(f"Train loss: {loss}")
        
        if torch.isnan(loss):
            raise ValueError("NaN loss")

        opt_vae.zero_grad()
        self.manual_backward(loss)
        opt_vae.step()
        
        self.untoggle_optimizer(opt_vae)
        
        # Discriminator step
        # See Algorithm 2: FactorVAE
        # https://proceedings.mlr.press/v80/kim18b/kim18b.pdf
        
        # Calculating KL[q(z) || p(z)] instead of TC
        # p(z) is the standard Gaussian prior
        
        self.toggle_optimizer(opt_discriminator)
        
        # TODO noqa: self.manual_backward with retain_graph not working for pred
        # As a temporary fix, just compute pred again
        mu, logvar, z, x_hat = self(x)
        pred, _ = self.loss(x, mu, logvar, z, x_hat, return_pred=True)
        
        # print(z.mean(), z.std())
        
        zp = torch.randn_like(z)
        predp: torch.Tensor = self.discriminator(zp)
        
        discriminator_loss = self.loss_discriminator(pred, predp)
        self.log("discriminator_loss", discriminator_loss)
        print(f"Discriminator loss: {discriminator_loss}")
        
        if torch.isnan(discriminator_loss):
            raise ValueError("NaN discriminator loss")
        
        opt_discriminator.zero_grad()
        self.manual_backward(discriminator_loss)
        opt_discriminator.step()
        
        self.untoggle_optimizer(opt_discriminator)

        # return loss
        return torch.zeros(1)
