import torch
import torch.nn.functional as F

from arch.vae.vae import VAE

class InfoVAE(VAE):
    # TODO Write DOCSTRING
    
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="info_vae",
        beta: float=1.0,
        gamma: float=1.0,
    ):
        super().__init__(in_channels, latent_dim, loss_reg, beta)
        
        self.save_hyperparameters()
 
    def _gaussian_log_density(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        normalisation = torch.log(2 * torch.tensor(torch.pi))
        return -0.5 * (
            (z - mu) ** 2 * torch.exp(-logvar) + logvar + normalisation
        )
    
    def _kl_qp(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Compute log(q(z(x_j) | x_i))
        log_qz_prob = self._gaussian_log_density(
            z.unsqueeze(1),
            mu.unsqueeze(0),
            logvar.unsqueeze(0),
        )

        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j) | x_i))) + const =
        # log(sum_i(prod_l q(z(x_j)_l | x_i))) + const
        log_qz = torch.logsumexp(log_qz_prob.sum(dim=2), dim=1)
        
        # Compute log p i.e. log standard Normal
        log_pz = self._gaussian_log_density(
            z,
            torch.zeros_like(mu).to(z.device),
            torch.zeros_like(logvar).to(z.device),
        ).sum(dim=1)

        return (log_qz - log_pz).mean()

    def loss(
        self,
        x: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
        z: torch.Tensor,
        x_hat: torch.Tensor,
        log_components: bool=True,
    ) -> torch.Tensor:
        batch_size = x.size(0)
        recon_loss = F.binary_cross_entropy(x_hat, x, reduction="sum") / batch_size
        kl_div = self._kl_divergence(mu, logvar)
        kl_qp = self._kl_qp(z, mu, logvar)
        print(f"KL_qp: {kl_qp}")
        
        weighted_kl_div = self.hparams.beta * kl_div
        weighted_kl_qp = self.hparams.gamma * kl_qp
        
        if log_components:
            self.log("recon_loss", recon_loss)
            self.log("kl_div", kl_div)
            self.log("kl_qp", weighted_kl_qp)

        return recon_loss + weighted_kl_div + weighted_kl_qp
