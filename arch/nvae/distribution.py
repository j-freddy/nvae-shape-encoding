import numpy as np
import torch

from utils.utils import soft_clamp

class Normal:
    """
    See class Normal in official NVAE implementation (distributions.py)
    """
    
    def __init__(self, mu: torch.Tensor, logsig: torch.Tensor, temp: float=1.0):
        mu = soft_clamp(mu)
        logsig = soft_clamp(logsig)
        
        self.mu = mu
        self.sigma = (torch.exp(logsig) + 1e-2) * temp

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def log_p(self, samples: torch.Tensor) -> torch.Tensor:
        samples_n = (samples - self.mu) / self.sigma
        return -0.5 * samples_n * samples_n - 0.5 * np.log(2 * np.pi) - torch.log(self.sigma)

    def kl(self, other: "Normal") -> torch.Tensor:
        a = (self.mu - other.mu) / other.sigma
        b = self.sigma / other.sigma
        return 0.5 * (a * a + b * b) - 0.5 - torch.log(b)
