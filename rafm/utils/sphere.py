"""Utilities for operations on the unit sphere S^{d-1} and scaled spheres."""
import torch
from torch import Tensor


def uniform_on_sphere(n: int, d: int, device: str = "cpu") -> Tensor:
    """Sample n points uniformly on S^{d-1}.

    Uses the Gaussian normalization trick: if X_i ~ N(0,1), then X/||X|| ~ Unif(S^{d-1}).
    """
    x = torch.randn(n, d, device=device)
    return x / torch.norm(x, dim=-1, keepdim=True).clamp(min=1e-12)


def slerp(x0: Tensor, x1: Tensor, t: float | Tensor) -> Tensor:
    """Spherical linear interpolation between x0 and x1 at time t.

    Assumes x0 and x1 lie on the same sphere (||x0|| == ||x1|| == R).
    Handles near-antipodal and near-identical pairs numerically.

    Args:
        x0: (batch, d) — source points on sphere of radius R
        x1: (batch, d) — target points on sphere of radius R
        t:  scalar or (batch,) — interpolation parameter in [0, 1]

    Returns:
        x_t: (batch, d) — interpolated points on the same sphere
    """
    R = torch.norm(x1, dim=-1, keepdim=True).clamp(min=1e-12)   # (batch, 1)
    u0 = x0 / R
    u1 = x1 / R

    # Cosine of angle between u0 and u1
    cos_theta = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)  # (batch, 1)
    theta = torch.acos(cos_theta)  # angle in [0, pi]

    # Near-identical pairs: fallback to linear interpolation (theta ~ 0)
    near_zero = theta.squeeze(-1) < 1e-6  # (batch,)

    # Near-antipodal pairs: perturb u0 with a random perpendicular direction
    near_antipodal = theta.squeeze(-1) > (torch.pi - 1e-3)  # (batch,)
    if near_antipodal.any():
        u0 = _handle_antipodal(u0, u1, near_antipodal)
        cos_theta = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

    sin_theta = torch.sin(theta).clamp(min=1e-12)  # (batch, 1)

    if not isinstance(t, Tensor):
        t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
    t = t.view(-1, 1) if t.dim() == 1 else t  # (batch, 1)

    # Slerp formula: sin((1-t)*theta)/sin(theta) * u0 + sin(t*theta)/sin(theta) * u1
    w0 = torch.sin((1.0 - t) * theta) / sin_theta
    w1 = torch.sin(t * theta) / sin_theta
    u_t = w0 * u0 + w1 * u1

    # Fallback to linear for near-identical pairs
    if near_zero.any():
        u_lin = (1.0 - t) * u0 + t * u1
        u_t[near_zero] = u_lin[near_zero]

    # Re-normalize to stay exactly on unit sphere, then scale back
    u_t = u_t / torch.norm(u_t, dim=-1, keepdim=True).clamp(min=1e-12)
    return R * u_t


def slerp_velocity(x0: Tensor, x1: Tensor, t: float | Tensor) -> Tensor:
    """Closed-form conditional vector field for spherical geodesic path.

    At time t, for x_t = slerp(x0, x1, t), the velocity is:
        dot_x_t = (theta / sin(theta)) * [-cos((1-t)*theta)*u0 + cos(t*theta)*u1] * R/R
    which simplifies to the Riemannian logarithm formula:
        u_t(x_t | x1) = (1/(1-t)) * Log_{x_t}^R(x1)

    We compute it directly from (x0, x1, t) for numerical stability.
    """
    R = torch.norm(x1, dim=-1, keepdim=True).clamp(min=1e-12)
    u0 = x0 / R
    u1 = x1 / R

    cos_theta = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)

    near_antipodal = theta.squeeze(-1) > (torch.pi - 1e-3)
    if near_antipodal.any():
        u0 = _handle_antipodal(u0, u1, near_antipodal)
        cos_theta = (u0 * u1).sum(dim=-1, keepdim=True).clamp(-1.0, 1.0)
        theta = torch.acos(cos_theta)

    sin_theta = torch.sin(theta).clamp(min=1e-12)

    if not isinstance(t, Tensor):
        t = torch.tensor(t, dtype=x0.dtype, device=x0.device)
    t = t.view(-1, 1) if t.dim() == 1 else t

    # dot_x_t = R * (theta/sin_theta) * [-cos((1-t)*theta)*u0 + cos(t*theta)*u1]
    c0 = -torch.cos((1.0 - t) * theta)
    c1 = torch.cos(t * theta)
    velocity = R * (theta / sin_theta) * (c0 * u0 + c1 * u1)

    # Fallback for near-zero theta: velocity = x1 - x0
    near_zero = theta.squeeze(-1) < 1e-6
    if near_zero.any():
        velocity[near_zero] = (x1 - x0)[near_zero]

    return velocity


def riemannian_log(x: Tensor, y: Tensor) -> Tensor:
    """Riemannian logarithm on the sphere of radius R = ||x||.

    Log_x^R(y) = (phi / sin(phi)) * (y - <x,y>/R^2 * x)
    where phi = arccos(<x,y>/R^2).
    """
    R2 = (x * x).sum(dim=-1, keepdim=True).clamp(min=1e-12)
    cos_phi = ((x * y).sum(dim=-1, keepdim=True) / R2).clamp(-1.0, 1.0)
    phi = torch.acos(cos_phi)
    sin_phi = torch.sin(phi).clamp(min=1e-12)

    proj = (x * y).sum(dim=-1, keepdim=True) / R2 * x
    log_xy = (phi / sin_phi) * (y - proj)

    # Fallback for near-zero phi
    near_zero = phi.squeeze(-1) < 1e-6
    if near_zero.any():
        log_xy[near_zero] = (y - x)[near_zero]

    return log_xy


def _handle_antipodal(u0: Tensor, u1: Tensor, mask: Tensor) -> Tensor:
    """For antipodal pairs in mask, perturb u0 with a random perpendicular direction."""
    u0 = u0.clone()
    n_antipodal = mask.sum().item()
    d = u0.shape[-1]
    # Random vector, project out u1 component to get perpendicular
    eps_vec = torch.randn(int(n_antipodal), d, device=u0.device)
    u1_anti = u1[mask]
    eps_vec = eps_vec - (eps_vec * u1_anti).sum(dim=-1, keepdim=True) * u1_anti
    eps_vec = eps_vec / torch.norm(eps_vec, dim=-1, keepdim=True).clamp(min=1e-12)
    # Slightly perturb u0 toward perpendicular direction (small eps = 1e-3)
    u0[mask] = u0[mask] + 1e-3 * eps_vec
    u0[mask] = u0[mask] / torch.norm(u0[mask], dim=-1, keepdim=True).clamp(min=1e-12)
    return u0
