"""Tests for the Flow Matching training and sampling pipeline."""
import torch
import pytest

from rafm.models.mlp import MLP
from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.sources.gaussian import GaussianSource
from rafm.flow_matching.loss import cfm_loss
from rafm.flow_matching.sampler import Sampler


class TestCFMLoss:
    def test_loss_finite(self):
        model = MLP(input_dim=4, hidden_dim=32, n_layers=2)
        path = EuclideanPath()
        source = GaussianSource()
        x1 = torch.randn(32, 4)
        loss = cfm_loss(model, path, source, x1, device="cpu")
        assert torch.isfinite(loss)
        assert loss > 0

    def test_loss_backprop(self):
        model = MLP(input_dim=4, hidden_dim=32, n_layers=2)
        path = EuclideanPath()
        source = GaussianSource()
        x1 = torch.randn(16, 4)
        loss = cfm_loss(model, path, source, x1)
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_spherical_loss_finite(self):
        model = MLP(input_dim=4, hidden_dim=32, n_layers=2)
        from rafm.sources.radial_empirical import RadialEmpiricalSource
        train_data = torch.distributions.StudentT(df=3).sample((1000, 4))
        source = RadialEmpiricalSource(mode="ecdf").fit(train_data)
        path = SphericalGeodesicPath()
        x1 = torch.distributions.StudentT(df=3).sample((16, 4))
        loss = cfm_loss(model, path, source, x1)
        assert torch.isfinite(loss)


class TestSampler:
    def _make_sampler(self, solver="euler", nfe=10):
        model = MLP(input_dim=4, hidden_dim=32, n_layers=2)
        source = GaussianSource()
        cfg = {"solver": solver, "nfe": nfe, "device": "cpu"}
        return Sampler(model, source, cfg), model

    def test_euler_shape(self):
        sampler, _ = self._make_sampler("euler", nfe=5)
        result = sampler.sample(50, 4)
        assert result["samples"].shape == (50, 4)
        assert not torch.isnan(result["samples"]).any()

    def test_heun_shape(self):
        sampler, _ = self._make_sampler("heun", nfe=5)
        result = sampler.sample(50, 4)
        assert result["samples"].shape == (50, 4)

    def test_rk4_shape(self):
        sampler, _ = self._make_sampler("rk4", nfe=5)
        result = sampler.sample(50, 4)
        assert result["samples"].shape == (50, 4)

    def test_nfe_recorded(self):
        sampler, _ = self._make_sampler("euler", nfe=10)
        result = sampler.sample(10, 4)
        assert result["nfe"] == 10
