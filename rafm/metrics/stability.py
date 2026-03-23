"""Stability metrics: NaN rate and exploding norm rate."""
import torch
from torch import Tensor


def stability_metrics(samples: Tensor, norm_threshold: float | None = None) -> dict:
    """Check generated sample stability.

    Args:
        samples:        (n, d) — generated samples
        norm_threshold: if None, uses 100 * median(||x||) as threshold

    Returns:
        dict with: nan_rate, exploding_norm_rate, invalid_rate
    """
    n = samples.shape[0]
    nan_mask = torch.isnan(samples).any(dim=-1)
    nan_rate = float(nan_mask.float().mean())

    valid_samples = samples[~nan_mask]
    if len(valid_samples) == 0:
        return {"nan_rate": nan_rate, "exploding_norm_rate": 1.0, "invalid_rate": 1.0}

    norms = torch.norm(valid_samples, dim=-1)
    if norm_threshold is None:
        norm_threshold = 100.0 * float(torch.median(norms))

    exploding_rate = float((norms > norm_threshold).float().mean())
    invalid_rate = nan_rate + (1 - nan_rate) * exploding_rate

    return {
        "nan_rate": nan_rate,
        "exploding_norm_rate": exploding_rate,
        "invalid_rate": invalid_rate,
    }
