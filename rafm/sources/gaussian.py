"""Standard isotropic Gaussian source."""
import torch
from torch import Tensor


class GaussianSource:
    """Isotropic Gaussian source N(0, I_d)."""
    name = "gaussian"

    def sample(self, n: int, d: int, device: str = "cpu") -> Tensor:
        return torch.randn(n, d, device=device)
