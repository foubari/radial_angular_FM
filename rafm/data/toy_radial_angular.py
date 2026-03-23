"""2D Radial-Angular Toy dataset.

Radial component: heavy-tailed (half-Student-t(df=3) by default)
Angular component: multimodal (mixture of von Mises directions)

Serves as the primary qualitative/visual dataset.
"""
import torch
from torch import Tensor

from rafm.data.base import BaseDataset


class ToyRadialAngular(BaseDataset):
    """2D toy dataset with heavy-tailed radius and multimodal angular distribution.

    Args:
        n_samples:    total number of samples to generate
        df:           degrees of freedom for Student-t radial component
        n_modes:      number of angular modes (von Mises mixture)
        kappa:        concentration parameter for each von Mises component
        scale:        scale factor for the radial component
        split_seed:   seed for train/val/test split (fixed, separate from model seed)
    """
    name = "toy_radial_angular"
    dim = 2

    def __init__(
        self,
        n_samples: int = 50_000,
        df: float = 3.0,
        n_modes: int = 4,
        kappa: float = 5.0,
        scale: float = 1.0,
        split_seed: int = 0,
    ):
        self.df = df
        self.n_modes = n_modes
        self.kappa = kappa
        self.scale = scale

        data = self._generate(n_samples)
        self._make_splits(data, split_seed=split_seed)

    def _generate(self, n: int) -> Tensor:
        # Radial component: |Student-t(df)|
        r = torch.distributions.StudentT(df=self.df).sample((n,)).abs() * self.scale  # (n,)

        # Angular component: uniform mixture of von Mises
        # von Mises in 2D: sample angle theta, then (cos(theta), sin(theta))
        mode_centers = torch.linspace(0, 2 * torch.pi, self.n_modes + 1)[:-1]  # (n_modes,)
        mode_idx = torch.randint(self.n_modes, (n,))
        mu = mode_centers[mode_idx]  # (n,)

        # Sample from von Mises: approximate via rejection or scipy
        # Using a simple approximation: theta ~ N(mu, 1/kappa) mod 2pi
        theta = mu + torch.randn(n) / (self.kappa ** 0.5)

        x = r.unsqueeze(-1) * torch.stack([torch.cos(theta), torch.sin(theta)], dim=-1)  # (n, 2)
        return x
