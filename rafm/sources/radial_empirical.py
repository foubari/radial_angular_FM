"""Empirical radial source — works for any dataset including real data.

Samples X = R * U, where R ~ p_hat_R (estimated from training data) and U ~ Unif(S^{d-1}).

Two estimation modes:
  - "ecdf"   : raw empirical CDF resampling via quantile interpolation (ported from SDEs.py)
  - "kde"    : kernel density estimation on radii (with optional log-radius transform)

For raw eCDF mode, KL is NOT a natural metric (distribution is not strictly positive).
Use Wasserstein / KS / quantile errors as primary metrics.
"""
import torch
from torch import Tensor
from sklearn.neighbors import KernelDensity
import numpy as np

from rafm.utils.sphere import uniform_on_sphere


class RadialEmpiricalSource:
    """Empirical radial source fitted from training data.

    Args:
        mode:       "ecdf" (default) or "kde"
        log_radius: if True, fit the estimator on log(r + eps) then back-transform
        kde_bandwidth: bandwidth for KDE (default: 0.1 * std(r_train))
        kde_kernel:  kernel for KDE (default: "gaussian")
    """

    def __init__(
        self,
        mode: str = "ecdf",
        log_radius: bool = False,
        kde_bandwidth: float | None = None,
        kde_kernel: str = "gaussian",
    ):
        assert mode in ("ecdf", "kde"), f"Unknown mode: {mode}"
        self.mode = mode
        self.log_radius = log_radius
        self.kde_bandwidth = kde_bandwidth
        self.kde_kernel = kde_kernel
        self._r_train: Tensor | None = None
        self._kde = None

        suffix = f"_{mode}"
        if log_radius:
            suffix += "_log"
        self.name = f"radial_empirical{suffix}"

    def fit(self, train_data: Tensor) -> "RadialEmpiricalSource":
        """Fit the radial estimator from training data.

        Args:
            train_data: (n_train, d) — training samples (NEVER mix with val/test)
        """
        r = torch.norm(train_data, dim=-1).cpu()  # (n_train,)
        if self.log_radius:
            r_fit = torch.log(r + 1e-6)
        else:
            r_fit = r
        self._r_train = r_fit

        if self.mode == "kde":
            bw = self.kde_bandwidth
            if bw is None:
                bw = 0.1 * float(r_fit.std())
            r_np = r_fit.numpy().reshape(-1, 1)
            self._kde = KernelDensity(kernel=self.kde_kernel, bandwidth=bw).fit(r_np)

        return self

    def sample(self, n: int, d: int, device: str = "cpu") -> Tensor:
        """Sample n points from the empirical radial source."""
        assert self._r_train is not None, "Call fit() before sample()."

        r = self._sample_radii(n, device)  # (n,)
        u = uniform_on_sphere(n, d, device=device)  # (n, d)
        return r.view(n, 1) * u

    def sample_radii(self, n: int, device: str = "cpu") -> Tensor:
        """Sample only radii (useful for Exp 0 diagnostics)."""
        assert self._r_train is not None, "Call fit() before sample_radii()."
        return self._sample_radii(n, device)

    def _sample_radii(self, n: int, device: str) -> Tensor:
        if self.mode == "ecdf":
            u = torch.rand(n)
            r_fit = self._r_train
            r_gen = torch.quantile(r_fit, u)  # inverse-CDF / quantile interpolation
            if self.log_radius:
                r_gen = torch.exp(r_gen) - 1e-6
            # Clamp to non-negative (safety for KDE mode, eCDF always >= 0)
            r_gen = r_gen.clamp(min=0.0)
        else:  # kde
            r_gen_np = self._kde.sample(n)  # (n, 1)
            r_gen = torch.from_numpy(r_gen_np).float().squeeze(-1)
            if self.log_radius:
                r_gen = torch.exp(r_gen) - 1e-6
            r_gen = r_gen.clamp(min=0.0)

        return r_gen.to(device)
