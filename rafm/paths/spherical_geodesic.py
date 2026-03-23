"""Spherical geodesic conditional path (core RAFM contribution).

Path: x_t = R * slerp(u0, u1, t), where x0 = R * u0, x1 = R * u1, R = ||x1||.

Properties (verified in tests/test_paths.py):
  - Norm preservation: ||x_t|| = R for all t
  - Tangent velocity: x_t^T * dot_x_t = 0
  - Handles near-antipodal pairs (theta -> pi) via measurable completion
  - Handles near-identical pairs (theta -> 0) via linear fallback

The conditional vector field at x_t is:
  u_t(x_t | x1) = (1/(1-t)) * Log_{x_t}^R(x1)
computed in closed form from (x0, x1, t) for numerical stability.
"""
import torch
from torch import Tensor

from rafm.utils.sphere import slerp, slerp_velocity


class SphericalGeodesicPath:
    """Spherical geodesic conditional path on scaled sphere of radius R = ||x1||.

    The source x0 MUST already lie on the same sphere as x1 (||x0|| == ||x1||).
    This is guaranteed when using RadialOracleSource or RadialEmpiricalSource
    with the coupling x0 = R * u0, x1 = R * u1 as described in the paper.
    """
    name = "spherical_geodesic"

    def sample_path(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """
        Args:
            x0: (batch, d) — source on sphere of radius R = ||x1||
            x1: (batch, d) — target
            t:  (batch,)   — time in [0, 1]
        Returns:
            x_t: (batch, d) on sphere of radius R
        """
        return slerp(x0, x1, t.view(-1, 1))

    def conditional_vector_field(self, x0: Tensor, x1: Tensor, t: Tensor) -> Tensor:
        """Closed-form velocity from paper (Prop 3 / conditional vector field).

        Returns:
            dot_x_t: (batch, d) — tangent to sphere, target for CFM loss
        """
        return slerp_velocity(x0, x1, t.view(-1, 1))
