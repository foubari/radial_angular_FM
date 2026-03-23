"""Radial metrics — theory-aligned primary metrics.

All metrics computed on sample norms ||x|| vs test data norms.
These are the primary metrics for validating the radial source correction.

For empirical eCDF-based sources: use these metrics (NOT KL) as primary reporters.
"""
import torch
from torch import Tensor
import numpy as np


def radial_metrics(samples: Tensor, test_data: Tensor) -> dict:
    """Compute all radial metrics.

    Args:
        samples:   (n_gen, d) — generated samples
        test_data: (n_test, d) — held-out test data (NEVER train data)

    Returns:
        dict with keys: radial_w1, ks_stat, q95_err, q99_err, q995_err,
                        tail_exceedance_95, tail_exceedance_99
    """
    r_gen = torch.norm(samples, dim=-1).float()
    r_test = torch.norm(test_data, dim=-1).float()
    return {
        "radial_w1": _wasserstein1_1d(r_gen, r_test),
        "ks_stat": _ks_stat(r_gen, r_test),
        **_quantile_errors(r_gen, r_test),
        **_tail_exceedance(r_gen, r_test),
    }


def _wasserstein1_1d(a: Tensor, b: Tensor) -> float:
    """1D Wasserstein-1 distance: integral |F_a - F_b|."""
    a_sorted, _ = torch.sort(a)
    b_sorted, _ = torch.sort(b)
    n, m = len(a), len(b)
    # Merge and compute W1 via sorted CDFs
    all_vals = torch.cat([a_sorted, b_sorted])
    all_vals, _ = torch.sort(all_vals)
    fa = torch.searchsorted(a_sorted, all_vals).float() / n
    fb = torch.searchsorted(b_sorted, all_vals).float() / m
    diffs = torch.diff(all_vals, prepend=all_vals[:1])
    return float((torch.abs(fa - fb) * diffs).sum())


def _ks_stat(a: Tensor, b: Tensor) -> float:
    """KS statistic: sup_r |F_a(r) - F_b(r)|."""
    a_sorted, _ = torch.sort(a)
    b_sorted, _ = torch.sort(b)
    n, m = len(a), len(b)
    all_vals = torch.cat([a_sorted, b_sorted])
    all_vals, _ = torch.sort(all_vals)
    fa = torch.searchsorted(a_sorted, all_vals).float() / n
    fb = torch.searchsorted(b_sorted, all_vals).float() / m
    return float(torch.max(torch.abs(fa - fb)))


def _quantile_errors(r_gen: Tensor, r_test: Tensor) -> dict:
    """Extreme quantile errors at 95%, 99%, 99.5%."""
    levels = [0.95, 0.99, 0.995]
    result = {}
    for q in levels:
        q_gen = torch.quantile(r_gen, q).item()
        q_test = torch.quantile(r_test, q).item()
        key = f"q{int(q*1000)}_err"
        result[key] = abs(q_gen - q_test) / (abs(q_test) + 1e-12)
    return result


def _tail_exceedance(r_gen: Tensor, r_test: Tensor) -> dict:
    """Tail exceedance calibration: fraction above test quantile thresholds."""
    result = {}
    for q_level in [0.95, 0.99]:
        threshold = torch.quantile(r_test, q_level).item()
        expected = 1.0 - q_level
        actual = float((r_gen > threshold).float().mean())
        result[f"tail_exc_{int(q_level*100)}"] = abs(actual - expected)
    return result
