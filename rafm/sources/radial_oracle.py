"""Oracle radial source — SYNTHETIC DATASETS ONLY.

The oracle source requires knowing the exact analytic radial CDF F_R of the data.
This is only available for synthetic distributions (Student-t, Gaussian, toy 2D).
Do NOT use this for real datasets such as PIV.

Samples X = R * U, where R ~ p_R (sampled via inverse-CDF) and U ~ Unif(S^{d-1}).
"""
import torch
from torch import Tensor

from rafm.utils.sphere import uniform_on_sphere


class RadialOracleSource:
    """Radial source with oracle (analytic) radial CDF.

    Args:
        radial_sampler: callable(n, device) -> Tensor of shape (n,)
            that samples from the exact analytic radial distribution p_R.
        name_suffix: string appended to identify the dataset/distribution.
    """
    def __init__(self, radial_sampler, name_suffix: str = ""):
        self._radial_sampler = radial_sampler
        self.name = f"radial_oracle_{name_suffix}" if name_suffix else "radial_oracle"

    def sample(self, n: int, d: int, device: str = "cpu") -> Tensor:
        r = self._radial_sampler(n, device=device).view(n, 1)  # (n, 1)
        u = uniform_on_sphere(n, d, device=device)             # (n, d)
        return r * u


# ---------------------------------------------------------------------------
# Factory functions for analytic radial samplers of known distributions
# ---------------------------------------------------------------------------

def student_t_radial_sampler(df: float, d: int, A: Tensor | None = None):
    """Radial sampler for a correlated Student-t distribution.

    A multivariate Student-t(df) with covariance A @ A^T has radial law equal to
    the norm of A @ z where z ~ Student-t(df, d) i.i.d.

    We sample ||A @ z|| directly.
    """
    def _sample(n: int, device: str = "cpu") -> Tensor:
        z = torch.distributions.StudentT(df=df).sample((n, d)).to(device)  # (n, d)
        if A is not None:
            x = z @ A.T.to(device)  # (n, d)
        else:
            x = z
        return torch.norm(x, dim=-1)  # (n,)
    return _sample


def gaussian_aniso_radial_sampler(d: int, A: Tensor | None = None):
    """Radial sampler for a correlated anisotropic Gaussian.

    X = A @ z, z ~ N(0, I_d). ||X|| ~ chi distribution scaled by ||A||.
    """
    def _sample(n: int, device: str = "cpu") -> Tensor:
        z = torch.randn(n, d, device=device)
        if A is not None:
            x = z @ A.T.to(device)
        else:
            x = z
        return torch.norm(x, dim=-1)  # (n,)
    return _sample


def toy_radial_sampler(df: float = 3.0):
    """Radial sampler for the 2D toy: heavy-tailed radius ~ half-Student-t(df)."""
    def _sample(n: int, device: str = "cpu") -> Tensor:
        # |Student-t(df)| as a scalar radius
        r = torch.distributions.StudentT(df=df).sample((n,)).abs().to(device)
        return r
    return _sample
