"""Conditional Flow Matching loss.

L_CFM(theta) = E_{t, (x0, x1)} [ || v_theta(x_t, t) - u_t(x0, x1, t) ||^2 ]

This is simulation-free: no ODE/SDE integration during training.
"""
import torch
from torch import Tensor


def cfm_loss(
    model: torch.nn.Module,
    path,
    source,
    x1: Tensor,
    device: str = "cpu",
) -> Tensor:
    """Compute one CFM loss batch.

    For RAFM (spherical path): x0 is sampled as R * u0 where R = ||x1||.
    For Gaussian FM: x0 ~ N(0, I).
    For source-only FM: x0 = R * u0 (radial source) but path is Euclidean.

    Args:
        model:  v_theta(x, t) — MLP
        path:   EuclideanPath or SphericalGeodesicPath
        source: GaussianSource or RadialOracleSource or RadialEmpiricalSource
        x1:     (batch, d) — target samples from data
        device: compute device

    Returns:
        loss: scalar tensor
    """
    batch, d = x1.shape
    x1 = x1.to(device)

    # Sample time t ~ Unif[0, 1]
    t = torch.rand(batch, device=device)

    # Sample source x0
    if hasattr(source, "sample") and _is_radial_source(source):
        # Radial sources: couple x0 to x1 — x0 = R * u0 where R = ||x1||
        R = torch.norm(x1, dim=-1, keepdim=True)          # (batch, 1)
        from rafm.utils.sphere import uniform_on_sphere
        u0 = uniform_on_sphere(batch, d, device=device)   # (batch, d)
        x0 = R * u0                                        # (batch, d)
    else:
        x0 = source.sample(batch, d, device=device)

    # Interpolate
    x_t = path.sample_path(x0, x1, t)

    # Target conditional vector field
    u_t = path.conditional_vector_field(x0, x1, t)

    # Predict
    v_pred = model(x_t, t)

    return ((v_pred - u_t) ** 2).sum(dim=-1).mean()


def _is_radial_source(source) -> bool:
    return "radial" in getattr(source, "name", "")
