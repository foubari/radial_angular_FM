"""ODE sampler for generation: integrates dx/dt = v_theta(x, t) from t=0 to t=1.

This is a deterministic ODE — NO diffusion term. Adapted from sde_scheme.py
(Euler, Heun, RK4) with the stochastic term removed.

For spherical geodesic paths, enable `project_tangent=True` (set via cfg or path name).
This projects the learned velocity onto the tangent space of the current x at each step,
preventing norm drift due to residual radial components in the learned field.
This is theoretically justified: the true conditional field is tangent by construction
(Prop. 3 in the paper), so the learned approximation should be too.
"""
import time

import torch
from torch import Tensor


def _project_tangent(v: Tensor, x: Tensor) -> Tensor:
    """Project v onto the tangent space of the sphere at x: v - (x.v / ||x||^2) * x."""
    R2 = (x * x).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    radial_coeff = (x * v).sum(dim=-1, keepdim=True) / R2
    return v - radial_coeff * x


class Sampler:
    """ODE-based sampler for generation.

    Args:
        model:          trained v_theta(x, t)
        source:         source distribution (must have .sample(n, d, device))
        cfg:            config dict
        project_tangent: if True, project velocity onto tangent space at each step.
                         Recommended for spherical geodesic paths to prevent norm drift.
    """

    def __init__(self, model, source, cfg: dict, project_tangent: bool = False):
        self.model = model
        self.source = source
        self.cfg = cfg
        self.project_tangent = project_tangent or (cfg.get("path") == "spherical_geodesic")
        self.device = cfg.get("device", "cpu")
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    @torch.no_grad()
    def sample(self, n: int, d: int) -> dict:
        """Generate n samples.

        Returns:
            dict with:
              "samples":     (n, d) generated samples on CPU
              "nfe":         number of function evaluations
              "sample_time_s": wall-clock sampling time
        """
        solver = self.cfg.get("solver", "rk4")
        nfe = self.cfg.get("nfe", 128)

        self.model.eval()
        x0 = self.source.sample(n, d, device=self.device)
        t0 = time.time()

        if solver == "euler":
            samples = self._euler(x0, nfe)
        elif solver == "heun":
            samples = self._heun(x0, nfe)
        elif solver == "rk4":
            samples = self._rk4(x0, nfe)
        else:
            raise ValueError(f"Unknown solver: {solver}")

        elapsed = time.time() - t0
        return {
            "samples": samples.cpu(),
            "nfe": nfe if solver == "euler" else nfe * 2 if solver == "heun" else nfe * 4,
            "sample_time_s": elapsed,
        }

    def _v(self, x: Tensor, t: Tensor) -> Tensor:
        v = self.model(x, t)
        if self.project_tangent:
            v = _project_tangent(v, x)
        return v

    def _euler(self, x: Tensor, n_steps: int) -> Tensor:
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=self.device)
            x = x + dt * self._v(x, t)
        return x

    def _heun(self, x: Tensor, n_steps: int) -> Tensor:
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t = torch.full((x.shape[0],), i * dt, device=self.device)
            t_next = torch.full((x.shape[0],), i * dt + dt, device=self.device)
            v1 = self._v(x, t)
            x_pred = x + dt * v1
            v2 = self._v(x_pred, t_next)
            x = x + dt * 0.5 * (v1 + v2)
        return x

    def _rk4(self, x: Tensor, n_steps: int) -> Tensor:
        dt = 1.0 / n_steps
        for i in range(n_steps):
            t0 = torch.full((x.shape[0],), i * dt, device=self.device)
            t_mid = torch.full((x.shape[0],), i * dt + dt / 2, device=self.device)
            t_end = torch.full((x.shape[0],), i * dt + dt, device=self.device)

            k1 = self._v(x, t0)
            k2 = self._v(x + dt / 2 * k1, t_mid)
            k3 = self._v(x + dt / 2 * k2, t_mid)
            k4 = self._v(x + dt * k3, t_end)
            x = x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
        return x
