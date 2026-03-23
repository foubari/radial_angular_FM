"""Correlated Student-t dataset.

Generalizes the Cauchy class from 18727.../data.py to arbitrary df.
X = z @ A^T, z ~ Student-t(df, d) i.i.d.

The mixing matrix A is fixed at construction and saved for reproducibility.
"""
import torch
from torch import Tensor
from pathlib import Path

from rafm.data.base import BaseDataset


class StudentT(BaseDataset):
    """Correlated Student-t in d dimensions.

    Args:
        dim:        dimension (e.g. 16 or 32)
        df:         degrees of freedom (e.g. 3 or 5)
        n_samples:  total number of samples to generate
        correlated: if True, use random mixing matrix A; else A = I
        matrix_seed: seed for generating A (reproducible across runs)
        split_seed:  seed for train/val/test split
    """
    dim: int

    def __init__(
        self,
        dim: int = 16,
        df: float = 3.0,
        n_samples: int = 50_000,
        correlated: bool = True,
        matrix_seed: int = 42,
        split_seed: int = 0,
    ):
        self.dim = dim
        self.df = df
        self.name = f"student_t_d{dim}_df{df}"
        if correlated:
            self.name += "_cor"

        # Fixed mixing matrix
        rng = torch.Generator()
        rng.manual_seed(matrix_seed)
        if correlated:
            self.A = torch.randn(dim, dim, generator=rng)
        else:
            self.A = torch.eye(dim)

        data = self._generate(n_samples)
        self._make_splits(data, split_seed=split_seed)

    def _generate(self, n: int) -> Tensor:
        z = torch.distributions.StudentT(df=self.df).sample((n, self.dim))
        return z @ self.A.T

    def save_matrix(self, path: str | Path) -> None:
        torch.save(self.A, path)

    @classmethod
    def load_matrix(cls, path: str | Path) -> Tensor:
        return torch.load(path)
