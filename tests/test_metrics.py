"""Tests for metric functions."""
import torch
import pytest

from rafm.metrics.radial import radial_metrics, _ks_stat, _wasserstein1_1d
from rafm.metrics.distributional import distributional_metrics, sliced_wasserstein, compute_mmd
from rafm.metrics.stability import stability_metrics


class TestRadialMetrics:
    def test_identical_distributions(self):
        """Metrics should be near zero for identical distributions."""
        data = torch.randn(500, 4) * 2
        metrics = radial_metrics(data, data)
        assert metrics["radial_w1"] < 0.05
        assert metrics["ks_stat"] < 0.05

    def test_different_distributions(self):
        """Heavy-tail vs Gaussian should have large radial metrics."""
        gauss = torch.randn(1_000, 8)
        heavy = torch.distributions.StudentT(df=1).sample((1_000, 8))
        metrics = radial_metrics(heavy, gauss)
        assert metrics["radial_w1"] > 0.1

    def test_quantile_keys_present(self):
        data = torch.randn(200, 4)
        metrics = radial_metrics(data, data)
        assert "q950_err" in metrics
        assert "q990_err" in metrics
        assert "q995_err" in metrics

    def test_ks_zero_identical(self):
        r = torch.rand(500) * 5
        assert _ks_stat(r, r) == 0.0

    def test_w1_symmetry(self):
        a = torch.randn(300)
        b = torch.randn(300) * 2
        assert abs(_wasserstein1_1d(a, b) - _wasserstein1_1d(b, a)) < 1e-5


class TestDistributionalMetrics:
    def test_sliced_w1_zero_identical(self):
        x = torch.randn(200, 4)
        sw = sliced_wasserstein(x, x, n_projections=50)
        assert sw < 0.05

    def test_mmd_zero_identical(self):
        x = torch.randn(200, 4)
        mmd = compute_mmd(x, x)
        assert mmd < 1e-5

    def test_mmd_positive_different(self):
        x = torch.randn(200, 4)
        y = torch.randn(200, 4) * 5
        mmd = compute_mmd(x, y)
        assert mmd > 0.1


class TestStabilityMetrics:
    def test_no_nans(self):
        samples = torch.randn(100, 4)
        m = stability_metrics(samples)
        assert m["nan_rate"] == 0.0

    def test_nan_detection(self):
        samples = torch.randn(100, 4)
        samples[5, 2] = float("nan")
        m = stability_metrics(samples)
        assert m["nan_rate"] > 0.0

    def test_exploding_norm_detection(self):
        samples = torch.randn(100, 4)
        samples[0] = samples[0] * 1000  # exploding sample
        m = stability_metrics(samples)
        assert m["exploding_norm_rate"] > 0.0
