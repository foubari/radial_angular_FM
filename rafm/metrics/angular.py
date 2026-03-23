"""Angular metrics: distributional quality within radius bins.

Validates the "remaining mismatch is angular" narrative from the paper.
Within each norm quantile bin, we compare angular distributions of
generated vs test samples using Sliced Wasserstein on unit vectors.
"""
import torch
from torch import Tensor

from rafm.metrics.distributional import sliced_wasserstein


def angular_metrics(
    samples: Tensor,
    test_data: Tensor,
    n_bins: int = 4,
    n_projections: int = 200,
) -> dict:
    """Sliced Wasserstein on unit vectors within norm quantile bins.

    Args:
        samples:       (n_gen, d)
        test_data:     (n_test, d)
        n_bins:        number of radius quantile bins
        n_projections: projections for Sliced Wasserstein

    Returns:
        dict with keys: angular_sw_bin{i} for i in 0..n_bins-1, angular_sw_mean
    """
    r_test = torch.norm(test_data, dim=-1)
    quantiles = torch.linspace(0, 1, n_bins + 1)
    thresholds = torch.quantile(r_test, quantiles)

    results = {}
    sw_values = []

    for i in range(n_bins):
        lo, hi = thresholds[i].item(), thresholds[i + 1].item()

        r_gen = torch.norm(samples, dim=-1)
        mask_gen = (r_gen >= lo) & (r_gen < hi)
        mask_test = (r_test >= lo) & (r_test < hi)

        if mask_gen.sum() < 10 or mask_test.sum() < 10:
            results[f"angular_sw_bin{i}"] = float("nan")
            continue

        # Unit vectors within this bin
        u_gen = samples[mask_gen]
        u_gen = u_gen / torch.norm(u_gen, dim=-1, keepdim=True).clamp(min=1e-12)
        u_test = test_data[mask_test]
        u_test = u_test / torch.norm(u_test, dim=-1, keepdim=True).clamp(min=1e-12)

        sw = sliced_wasserstein(u_gen, u_test, n_projections)
        results[f"angular_sw_bin{i}"] = sw
        sw_values.append(sw)

    results["angular_sw_mean"] = float(sum(sw_values) / len(sw_values)) if sw_values else float("nan")
    return results
