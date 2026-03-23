"""Tests for conditional paths.

Critical tests (from scientific guardrails):
  - Norm preservation: ||x_t|| = R for all t
  - Tangent velocity: x_t^T * dot_x_t = 0
  - Continuity near antipodal: no numerical jumps when theta -> pi
  - Euclidean path: correct linear interpolation
"""
import torch
import pytest

from rafm.paths.euclidean import EuclideanPath
from rafm.paths.spherical_geodesic import SphericalGeodesicPath
from rafm.utils.sphere import slerp, slerp_velocity, _handle_antipodal


def make_sphere_pair(batch: int, d: int, seed: int = 0):
    """Create source/target pairs on the same sphere."""
    torch.manual_seed(seed)
    R = torch.rand(batch, 1) * 3 + 1  # radii in [1, 4]
    u0 = torch.randn(batch, d)
    u0 = u0 / torch.norm(u0, dim=-1, keepdim=True)
    u1 = torch.randn(batch, d)
    u1 = u1 / torch.norm(u1, dim=-1, keepdim=True)
    x0 = R * u0
    x1 = R * u1
    return x0, x1


class TestEuclideanPath:
    def test_interpolation_at_zero(self):
        x0 = torch.randn(16, 4)
        x1 = torch.randn(16, 4)
        path = EuclideanPath()
        t = torch.zeros(16)
        xt = path.sample_path(x0, x1, t)
        assert torch.allclose(xt, x0, atol=1e-6)

    def test_interpolation_at_one(self):
        x0 = torch.randn(16, 4)
        x1 = torch.randn(16, 4)
        path = EuclideanPath()
        t = torch.ones(16)
        xt = path.sample_path(x0, x1, t)
        assert torch.allclose(xt, x1, atol=1e-6)

    def test_vector_field_constant(self):
        x0 = torch.randn(16, 4)
        x1 = torch.randn(16, 4)
        path = EuclideanPath()
        for t_val in [0.0, 0.3, 0.7, 1.0]:
            t = torch.full((16,), t_val)
            xt = path.sample_path(x0, x1, t)
            u = path.conditional_vector_field(x0, x1, t)
            expected = x1 - x0
            assert torch.allclose(u, expected, atol=1e-6)


class TestSphericalGeodesicPath:
    def test_norm_preservation(self):
        """||x_t|| = R = ||x_1|| for all t."""
        x0, x1 = make_sphere_pair(64, 8)
        path = SphericalGeodesicPath()
        R = torch.norm(x1, dim=-1)
        for t_val in [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0]:
            t = torch.full((64,), t_val)
            xt = path.sample_path(x0, x1, t)
            norms = torch.norm(xt, dim=-1)
            assert torch.allclose(norms, R, atol=1e-5), \
                f"Norm not preserved at t={t_val}: max error {(norms - R).abs().max():.2e}"

    def test_tangent_velocity(self):
        """x_t^T * dot_x_t = 0 (velocity tangent to sphere)."""
        x0, x1 = make_sphere_pair(64, 8)
        path = SphericalGeodesicPath()
        for t_val in [0.05, 0.2, 0.5, 0.8, 0.95]:
            t = torch.full((64,), t_val)
            xt = path.sample_path(x0, x1, t)
            vt = path.conditional_vector_field(x0, x1, t)
            dot = (xt * vt).sum(dim=-1)
            assert torch.allclose(dot, torch.zeros_like(dot), atol=1e-4), \
                f"Velocity not tangent at t={t_val}: max |dot|={dot.abs().max():.2e}"

    def test_endpoints(self):
        """x_0 = source, x_1 = target."""
        x0, x1 = make_sphere_pair(32, 4)
        path = SphericalGeodesicPath()
        t0 = torch.zeros(32)
        t1 = torch.ones(32)
        assert torch.allclose(path.sample_path(x0, x1, t0), x0, atol=1e-5)
        assert torch.allclose(path.sample_path(x0, x1, t1), x1, atol=1e-5)

    def test_antipodal_no_nan(self):
        """No NaN or infinite values for near-antipodal pairs."""
        torch.manual_seed(42)
        d = 8
        x1 = torch.randn(16, d)
        R = torch.norm(x1, dim=-1, keepdim=True)
        x1 = x1 / torch.norm(x1, dim=-1, keepdim=True) * R
        # Make x0 exactly antipodal
        x0 = -x1 + 1e-4 * torch.randn(16, d)  # near-antipodal
        x0 = x0 / torch.norm(x0, dim=-1, keepdim=True) * R

        path = SphericalGeodesicPath()
        for t_val in [0.1, 0.5, 0.9]:
            t = torch.full((16,), t_val)
            xt = path.sample_path(x0, x1, t)
            vt = path.conditional_vector_field(x0, x1, t)
            assert not torch.isnan(xt).any(), f"NaN in x_t at t={t_val} (antipodal)"
            assert not torch.isnan(vt).any(), f"NaN in v_t at t={t_val} (antipodal)"
            assert not torch.isinf(xt).any(), f"Inf in x_t at t={t_val} (antipodal)"

    def test_antipodal_continuity(self):
        """Norm preservation holds even near antipodal."""
        torch.manual_seed(0)
        d = 4
        x1 = torch.randn(8, d)
        R = torch.norm(x1, dim=-1, keepdim=True).clamp(min=0.5)
        x1 = x1 / torch.norm(x1, dim=-1, keepdim=True) * R
        x0 = -x1 + 1e-3 * torch.randn(8, d)
        x0 = x0 / torch.norm(x0, dim=-1, keepdim=True) * R

        path = SphericalGeodesicPath()
        R_flat = R.squeeze(-1)
        for t_val in [0.3, 0.5, 0.7]:
            t = torch.full((8,), t_val)
            xt = path.sample_path(x0, x1, t)
            norms = torch.norm(xt, dim=-1)
            assert torch.allclose(norms, R_flat, atol=1e-4), \
                f"Norm not preserved near antipodal at t={t_val}"
