import torch

class Normal:
    def __init__(self, mu, logsig, temp: float=1.0):
        self.mu = mu
        self.sigma = (torch.exp(logsig) + 1e-2) * temp

    def sample(self):
        return self.mu + self.sigma * torch.randn_like(self.mu)

    def log_p(self, samples: torch.Tensor) -> torch.Tensor:
        NotImplemented
