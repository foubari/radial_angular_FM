"""Non-learnable preprocessing modules.

These are explicit ablations — never enabled by default in FM baselines.
Ported from 18727.../NN.py.
"""
import torch
import torch.nn as nn


class NormalizeLogRadius(nn.Module):
    """x -> (x/||x||, log(||x||)).

    Output dimension: input_dim + 1.
    Use only as explicit preprocessing ablation.
    """
    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.norm(x, dim=-1, keepdim=True) + self.eps
        return torch.cat([x / norm, torch.log(norm)], dim=-1)


class PolarCoordinatesWithLogRadius(nn.Module):
    """Convert x in R^d to polar/spherical coordinates.

    d=2: (log_r, theta), output dim = 2
    d=3: (log_r, theta, phi), output dim = 3
    Use only as explicit preprocessing ablation.
    """
    def __init__(self, eps: float = 1e-12):
        super().__init__()
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d = x.shape[-1]
        norm = torch.norm(x, dim=-1, keepdim=True).clamp_min(self.eps)
        log_norm = torch.log(norm)
        if d == 2:
            theta = torch.atan2(x[..., 1], x[..., 0]).unsqueeze(-1)
            return torch.cat([log_norm, theta], dim=-1)
        elif d == 3:
            theta = torch.atan2(x[..., 1], x[..., 0]).unsqueeze(-1)
            phi = torch.acos((x[..., 2] / norm.squeeze(-1)).clamp(-1 + 1e-7, 1 - 1e-7)).unsqueeze(-1)
            return torch.cat([log_norm, theta, phi], dim=-1)
        else:
            raise NotImplementedError("PolarCoordinates supports d=2 or d=3 only.")
