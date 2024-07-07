import torch

from arch.vae.vae import VAE

class InfoVAE(VAE):
    """
    Single-layer VAE that implements the InfoVAE loss proposed by [1]. This
    class estimates KL[q(x) || p(x)] with minibatch sampling. See VAE docstring
    for more information.

    [1]: Zhao S, Song J, Ermon S. Infovae: Balancing learning and inference in
    variational autoencoders. InProceedings of the aaai conference on artificial
    intelligence 2019 Jul 17 (Vol. 33, No. 01, pp. 5885-5892).
    """
    
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
        x_hat_logits: torch.Tensor,
        log_components: bool=True,
    ) -> torch.Tensor:
        """
        Compute the InfoVAE loss: sum of reconstruction loss, beta-weighted
        KL divergence regulariser term and gamma-weighted KL[q(z) || p(z)].
        """
        recon_loss = self.reconstruction_loss(x, x_hat_logits)
        kl_div = self._kl_divergence(mu, logvar)
        kl_qp = self._kl_qp(z, mu, logvar)
        
        weighted_kl_div = self.hparams.beta * kl_div
        weighted_kl_qp = self.hparams.gamma * kl_qp
        
        if log_components:
            marginal_kl_div = self._kl_divergence(mu, logvar, marginal=True)
            
            self.log("loss/recon", recon_loss)
            self.log("loss/kl_div", kl_div)
            self.log("loss/kl_qp", weighted_kl_qp)
            for i, marginal_kl in enumerate(marginal_kl_div):
                self.log(f"loss/marginal_kl_div/dim_{i}", marginal_kl)

        return recon_loss + weighted_kl_div + weighted_kl_qp
