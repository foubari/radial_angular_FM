"""Tests for source distributions."""
import torch
import pytest

from rafm.sources.gaussian import GaussianSource
from rafm.sources.radial_empirical import RadialEmpiricalSource
from rafm.sources.radial_oracle import RadialOracleSource, student_t_radial_sampler


class TestGaussianSource:
    def test_shape(self):
        src = GaussianSource()
        x = src.sample(100, 8)
        assert x.shape == (100, 8)

    def test_distribution(self):
        src = GaussianSource()
        x = src.sample(10_000, 4)
        # Mean should be near 0
        assert x.mean().abs() < 0.1


class TestRadialEmpiricalSource:
    def _make_train_data(self, n=5000, d=8):
        # Student-t data (heavy-tailed)
        return torch.distributions.StudentT(df=3).sample((n, d))

    def test_fit_and_sample_shape(self):
        train = self._make_train_data()
        src = RadialEmpiricalSource(mode="ecdf").fit(train)
        x = src.sample(500, 8)
        assert x.shape == (500, 8)

    def test_radial_cdf_matches_train(self):
        """Radial CDF of samples should match training data."""
        from rafm.metrics.radial import _ks_stat
        train = self._make_train_data(n=10_000)
        src = RadialEmpiricalSource(mode="ecdf").fit(train)
        x = src.sample(5_000, 8)
        r_gen = torch.norm(x, dim=-1)
        r_train = torch.norm(train, dim=-1)
        ks = _ks_stat(r_gen, r_train)
        assert ks < 0.05, f"KS stat too high: {ks:.4f}"

    def test_log_radius_mode(self):
        train = self._make_train_data()
        src = RadialEmpiricalSource(mode="ecdf", log_radius=True).fit(train)
        x = src.sample(200, 8)
        assert x.shape == (200, 8)
        assert not torch.isnan(x).any()

    def test_kde_mode(self):
        train = self._make_train_data()
        src = RadialEmpiricalSource(mode="kde").fit(train)
        x = src.sample(200, 8)
        assert x.shape == (200, 8)
        assert not torch.isnan(x).any()

    def test_no_train_before_sample_raises(self):
        src = RadialEmpiricalSource()
        with pytest.raises(AssertionError):
            src.sample(10, 4)


class TestRadialOracleSource:
    def test_shape(self):
        sampler = student_t_radial_sampler(df=3.0, d=8)
        src = RadialOracleSource(sampler)
        x = src.sample(200, 8)
        assert x.shape == (200, 8)

    def test_radial_law_preserved(self):
        """Oracle samples should have the correct radial distribution."""
        from rafm.metrics.radial import _ks_stat
        d = 8
        df = 3.0
        # Generate reference Student-t data
        n = 20_000
        z = torch.distributions.StudentT(df=df).sample((n, d))
        r_ref = torch.norm(z, dim=-1)

        sampler = student_t_radial_sampler(df=df, d=d, A=None)
        src = RadialOracleSource(sampler)
        x = src.sample(n, d)
        r_gen = torch.norm(x, dim=-1)

        ks = _ks_stat(r_gen, r_ref)
        assert ks < 0.05, f"Oracle radial KS too high: {ks:.4f}"
