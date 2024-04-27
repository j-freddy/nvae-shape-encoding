import math
import torch

class LossRegulariser:
    """
    The default regulariser for VAE. This implements the beta-VAE regulariser
    term based on [1].
    
    [1]: Higgins I, Matthey L, Pal A, Burgess CP, Glorot X, Botvinick MM,
    Mohamed S, Lerchner A. beta-vae: Learning basic visual concepts with a
    constrained variational framework. ICLR (Poster). 2017 Apr 24;3.
    """
    
    def __init__(self, beta=1.0):
        self.beta = beta
        
    def _kl_divergence(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        return -0.5 * torch.sum(
            1 + logvar - mu.pow(2) - logvar.exp(),
            dim=1,
        ).mean()

    def __call__(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        kl_div = self._kl_divergence(mu, logvar)
        return self.beta * kl_div

class BetaTCVAERegulariser(LossRegulariser):
    """
    This is a simplified implementation of the beta-TCVAE regulariser term based
    on [1]. It fixes alpha = gamma = 1 in Eq. (4) of [1].
    
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

        return (log_qz - log_qz_product).mean()
    
    def __call__(
        self,
        z: torch.Tensor,
        mu: torch.Tensor,
        logvar: torch.Tensor,
    ) -> torch.Tensor:
        kl_div = self._kl_divergence(mu, logvar)
        tc = self._total_correlation(z, mu, logvar)
        
        # By fixing alpha = gamma = 1, Eq. (4) of [1] simplifies to:
        #   ELBO + (1 - beta) * TC
        return kl_div + (1 - self.beta) * tc

ID_TO_REGULARISER = {
    "beta_vae": LossRegulariser,
    "beta_tcvae": BetaTCVAERegulariser,
}
