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
    
    [1]: Chen RT, Li X, Grosse RB, Duvenaud DK. Isolating sources of
    disentanglement in variational autoencoders. Advances in neural information
    processing systems. 2018;31.
    """
    
    # TODO noqa
    def gaussian_log_density(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        normalization = torch.log(2 * torch.tensor(torch.pi))
        inv_sigma = torch.exp(-logvar)
        tmp = (z - mu)
        return -0.5 * (tmp * tmp * inv_sigma + logvar + normalization)
    
    def _total_correlation(self, z: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        log_qz_prob = self.gaussian_log_density(
            z.unsqueeze(1),
            mu.unsqueeze(0),
            logvar.unsqueeze(0),
        )
        
        log_qz_product = torch.logsumexp(log_qz_prob, dim=1).sum(dim=1)
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
        
        print(kl_div, tc)
        
        import sys
        sys.exit()
        
        # By fixing alpha = gamma = 1, Eq. (4) of [1] simplifies to:
        #   ELBO + (beta - 1) * TC
        return kl_div + (self.beta - 1) * tc

ID_TO_REGULARISER = {
    "beta_vae": LossRegulariser,
    "beta_tcvae": BetaTCVAERegulariser,
}
