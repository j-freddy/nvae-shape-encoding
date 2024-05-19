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

class InfoAdversarialVAE(VAE):
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="info_adversarial_vae",
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
        log_components: bool=True,
        return_pred: bool=False,
    ) -> torch.Tensor:
        recon_loss = self.reconstruction_loss(x, x_hat)
        # KL divergence between q(z|x) and p(z)
        # Forming part of ELBO
        kl_div = self._kl_divergence(mu, logvar)
        
        pred: torch.Tensor = self.discriminator(z)
        # KL divergence between q(z) and p(z)
        kl_qp = pred.mean()
        
        weighted_kl_div = self.hparams.beta * kl_div
        weighted_kl_qp = self.hparams.gamma * kl_qp
        
        if log_components:
            marginal_kl_div = self._kl_divergence(mu, logvar, marginal=True)
            
            self.log("recon_loss", recon_loss)
            self.log("kl_div", kl_div)
            self.log("kl_qp", weighted_kl_qp)
            for i, marginal_kl in enumerate(marginal_kl_div):
                self.log(f"marginal_kl_div/dim_{i}", marginal_kl)

        loss = recon_loss + weighted_kl_div + weighted_kl_qp
        
        if return_pred:
            return pred, loss
        return loss

    def loss_discriminator(
        self,
        pred: torch.Tensor,
        predp: torch.Tensor,
    ) -> torch.Tensor:
        ones = torch.ones_like(pred).to(pred.device)
        zeros = torch.zeros_like(predp).to(predp.device)
        
        loss = 0.5 * (
            F.binary_cross_entropy_with_logits(pred, ones) +
            F.binary_cross_entropy_with_logits(predp, zeros)
        )
        
        return loss

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        opt_vae, opt_discriminator = self.optimizers()
        
        # Discriminator step
        # See Algorithm 2: FactorVAE
        # https://proceedings.mlr.press/v80/kim18b/kim18b.pdf
        
        # Calculating KL[q(z) || p(z)] instead of TC
        # p(z) is the standard Gaussian prior
        
        self.toggle_optimizer(opt_discriminator)

        _, _, z, _ = self(x)
        pred: torch.Tensor = self.discriminator(z)
        
        zp = torch.randn_like(z).to(z.device)
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

        return loss
