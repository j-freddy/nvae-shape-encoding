import torch
import torch.nn.functional as F

from arch.vae.vae import VAE

# TODO Wrong implementation
# Replace TC with KL(q(z) || p(z))

class TCVAE(VAE):
    """
    See DOCSTRING of VAE class.
    
    This model uses a simplified implementation of the beta-TCVAE regulariser
    term based on [1]. It fixes alpha = gamma = 1 in Eq. (4) of [1].
    
    The code for computing total correlation (i.e. _total_correlation,
    _gaussian_log_density) is lifted and translated from [2], which was
    originally implemented with TensorFlow. Their repo can be found at
    https://github.com/google-research/disentanglement_lib.
    
    [1]: Chen RT, Li X, Grosse RB, Duvenaud DK. Isolating sources of
    disentanglement in variational autoencoders. Advances in neural information
    processing systems. 2018;31.
    
    [2]: Locatello F, Bauer S, Lucic M, Raetsch G, Gelly S, Schölkopf B, Bachem
    O. Challenging common assumptions in the unsupervised learning of
    disentangled representations. Ininternational conference on machine learning
    2019 May 24 (pp. 4114-4124). PMLR.
    """
    
    def __init__(
        self,
        in_channels: int=4,
        latent_dim: int=2,
        loss_reg: str="beta_tcvae",
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
    
    def _total_correlation(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        # Compute log(q(z(x_j)|x_i))
        log_qz_prob = self._gaussian_log_density(
            z.unsqueeze(1),
            mu.unsqueeze(0),
            logvar.unsqueeze(0),
        )
        
        # Compute log prod_l p(z(x_j)_l) = sum_l(log(sum_i(q(z(z_j)_l|x_i)))
        # + const)
        log_qz_product = torch.logsumexp(log_qz_prob, dim=1).sum(dim=1)
        # Compute log(q(z(x_j))) as log(sum_i(q(z(x_j)|x_i))) + const =
        # log(sum_i(prod_l q(z(x_j)_l|x_i))) + const
        log_qz = torch.logsumexp(log_qz_prob.sum(dim=2), dim=1)
        
        print(log_qz_product.mean())
        print(log_qz.mean())
        
        import sys
        sys.exit()

        return (log_qz - log_qz_product).mean()

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
        tc = self._total_correlation(z, mu, logvar)
        
        weighted_kl_div = self.hparams.beta * kl_div # beta = 0.1
        weighted_tc = self.hparams.gamma * tc        # gamma
        
        if log_components:
            self.log("recon_loss", recon_loss)
            self.log("kl_div", kl_div)
            self.log("tc", weighted_tc)
        
        # By fixing alpha = gamma = 1, Eq. (4) of [1] simplifies to:
        #   ELBO + (1 - beta) * TC
        return recon_loss + weighted_kl_div + weighted_tc
