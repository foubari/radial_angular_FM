"""Full-space distributional metrics: Sliced Wasserstein, MMD."""
import torch
from torch import Tensor
import numpy as np


def distributional_metrics(samples: Tensor, test_data: Tensor, n_projections: int = 500) -> dict:
    """Compute global distributional metrics.

    Args:
        samples:       (n_gen, d)
        test_data:     (n_test, d)
        n_projections: number of random projections for Sliced Wasserstein

    Returns:
        dict with keys: sliced_w1, mmd
    """
    return {
        "sliced_w1": sliced_wasserstein(samples, test_data, n_projections),
        "mmd": compute_mmd(samples, test_data),
    }


def sliced_wasserstein(a: Tensor, b: Tensor, n_projections: int = 500) -> float:
    """Sliced Wasserstein-1 distance."""
    d = a.shape[-1]
    device = a.device
    # Random unit directions
    dirs = torch.randn(n_projections, d, device=device)
    dirs = dirs / torch.norm(dirs, dim=-1, keepdim=True)

    # Project
    pa = (a @ dirs.T)  # (n_a, n_proj)
    pb = (b.to(device) @ dirs.T)  # (n_b, n_proj)

    pa_sorted, _ = torch.sort(pa, dim=0)
    # Resample pb to same size as pa
    if pa.shape[0] != pb.shape[0]:
        idx = torch.randint(pb.shape[0], (pa.shape[0],), device=device)
        pb = pb[idx]
    pb_sorted, _ = torch.sort(pb, dim=0)
    return float(torch.abs(pa_sorted - pb_sorted).mean())


def compute_mmd(x: Tensor, y: Tensor, sigma: float | None = None) -> float:
    """Maximum Mean Discrepancy with Gaussian kernel.

    Ported from 18727.../quantitative_comparison.py.
    """
    x = x.float()
    y = y.float()
    if sigma is None:
        # Median heuristic
        all_data = torch.cat([x, y], dim=0)
        dists = torch.cdist(all_data, all_data)
        sigma = float(torch.median(dists[dists > 0]))

    Kxx = _rbf_kernel(x, x, sigma)
    Kyy = _rbf_kernel(y, y, sigma)
    Kxy = _rbf_kernel(x, y, sigma)
    return float(Kxx.mean() + Kyy.mean() - 2 * Kxy.mean())


def _rbf_kernel(x: Tensor, y: Tensor, sigma: float) -> Tensor:
    xx = (x * x).sum(dim=-1, keepdim=True)
    yy = (y * y).sum(dim=-1, keepdim=True)
    xy = x @ y.T
    sq_dists = xx + yy.T - 2 * xy
    return torch.exp(-sq_dists / (2 * sigma ** 2))
