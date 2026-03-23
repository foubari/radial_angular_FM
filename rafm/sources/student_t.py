"""Isotropic Student-t prior source (optional baseline).

This is a non-data-aware heavy-tail prior: X ~ Student-t(df, d) i.i.d.
Unlike the radial source, this does NOT match the data radial law.
Use as an additional baseline to isolate the benefit of data-matched radial law.
"""
import torch
from torch import Tensor


class StudentTSource:
    """Isotropic Student-t source with configurable degrees of freedom."""

    def __init__(self, df: float = 3.0):
        self.df = df
        self.name = f"student_t_df{df}"
        self._dist = torch.distributions.StudentT(df=df)

    def sample(self, n: int, d: int, device: str = "cpu") -> Tensor:
        return self._dist.sample((n, d)).to(device)
