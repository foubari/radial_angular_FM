"""Correlated anisotropic Gaussian dataset (control benchmark).

X = z @ A^T, z ~ N(0, I_d).
On this dataset, Gaussian FM should already work well.
RAFM should remain comparable — no catastrophic failure.
"""
import torch
from torch import Tensor
from pathlib import Path

from rafm.data.base import BaseDataset


class GaussianAniso(BaseDataset):
    """Correlated anisotropic Gaussian in d dimensions.

    Args:
        dim:        dimension (e.g. 16 or 32)
        n_samples:  total number of samples to generate
        correlated: if True, use random mixing matrix A; else A = I
        matrix_seed: seed for generating A (reproducible across runs)
        split_seed:  seed for train/val/test split
    """
    dim: int

    def __init__(
        self,
        dim: int = 16,
        n_samples: int = 50_000,
        correlated: bool = True,
        matrix_seed: int = 42,
        split_seed: int = 0,
    ):
        self.dim = dim
        self.name = f"gaussian_aniso_d{dim}"
        if correlated:
            self.name += "_cor"

        rng = torch.Generator()
        rng.manual_seed(matrix_seed)
        if correlated:
            self.A = torch.randn(dim, dim, generator=rng)
        else:
            self.A = torch.eye(dim)

        data = self._generate(n_samples)
        self._make_splits(data, split_seed=split_seed)

    def _generate(self, n: int) -> Tensor:
        z = torch.randn(n, self.dim)
        return z @ self.A.T

    def save_matrix(self, path: str | Path) -> None:
        torch.save(self.A, path)
