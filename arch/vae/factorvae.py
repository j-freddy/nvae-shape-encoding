import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler

from arch.vae.vae import VAE

class Discriminator(nn.Module):
    def __init__(self, latent_dim: int):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = 512
        
        self.net = nn.Sequential(
            nn.Linear(latent_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.LeakyReLU(0.2, inplace=False),
            nn.Linear(self.hidden_dim, 2),
        )
    
    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)

class FactorVAE(VAE):
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="beta_vae",
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
    
    def configure_optimizers(self):
        opt_vae = optim.Adam(
            list(self.encoder.parameters()) + list(self.decoder.parameters()),
            lr=6e-5,
            weight_decay=1e-2,
        )
        
        opt_discriminator = optim.Adam(
            self.discriminator.parameters(),
            lr=6e-5,
            weight_decay=1e-2,
        )
        
        return [opt_vae, opt_discriminator], []

    def _permute(self, z):
        batch_size, _ = z.size()
        perm_z = []
        
        for z_j in z.split(1, 1):
            perm = torch.randperm(batch_size).to(z.device)
            perm_z.append(z_j[perm])

        return torch.cat(perm_z, 1)
        
    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat: torch.Tensor,
        return_pred: bool=False,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        kl_div = self._kl_divergence(mu, logvar)
        
        pred: torch.Tensor = self.discriminator(z)
        tc = (pred[:, :1] - pred[:, 1:]).mean()
        
        loss = recon_loss + kl_div + (1 - self.hparams.beta) * tc
        
        if return_pred:
            return pred, loss
        return loss

    def loss_discriminator(
        self,
        pred: torch.Tensor,
        pred_perm: torch.Tensor,
    ) -> torch.Tensor:
        return 0.5 * (
            F.cross_entropy(pred, torch.zeros_like(pred)) +
            F.cross_entropy(pred_perm, torch.ones_like(pred_perm))
        )

    def training_step(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, _, _ = x.shape
        
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
        self.manual_backward(loss, retain_graph=True)
        opt_vae.step()
        
        self.untoggle_optimizer(opt_vae)
        
        # Discriminator step
        # See Algorithm 2: FactorVAE
        # https://proceedings.mlr.press/v80/kim18b/kim18b.pdf
        
        self.toggle_optimizer(opt_discriminator)
        
        # TODO This shouldn't be done
        pred = pred.detach()
        
        # Select a random batch from the training set
        # This batch should be different from the one used in the VAE step
        
        batch_idx = torch.randint(
            0,
            # Last batch may be smaller so do not use
            len(self.trainer.train_dataloader) - 1,
            (1,),
        ).item()

        sampler = SubsetRandomSampler(
            list(
                range(batch_idx * batch_size, (batch_idx + 1) * batch_size)
            )
        )
        
        xp = next(iter(
            DataLoader(
                self.trainer.train_dataloader.dataset,
                sampler=sampler,
                batch_size=batch_size,
            )
        )).to(self.device)
        
        zp = self.get_latent(xp)
        zp_perm = self._permute(zp).detach()
        
        pred_perm: torch.Tensor = self.discriminator(zp_perm)
        
        tc_loss = self.loss_discriminator(pred, pred_perm)
        self.log("discriminator_loss", tc_loss)
        
        print(f"Discriminator loss: {tc_loss}")
        
        if torch.isnan(tc_loss):
            raise ValueError("NaN discriminator loss")
        
        opt_discriminator.zero_grad()
        self.manual_backward(tc_loss)
        opt_discriminator.step()
        
        self.untoggle_optimizer(opt_discriminator)

        return loss
