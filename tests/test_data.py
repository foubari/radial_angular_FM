"""Tests for datasets and train/val/test split integrity."""
import torch
import pytest

from rafm.data.toy_radial_angular import ToyRadialAngular
from rafm.data.student_t import StudentT
from rafm.data.gaussian_aniso import GaussianAniso


class TestBaseDataset:
    def _make_studentt(self):
        return StudentT(dim=4, df=3.0, n_samples=5_000)

    def test_split_sizes(self):
        ds = self._make_studentt()
        total = len(ds._data)
        assert ds.n_train + ds.n_val + ds.n_test == total

    def test_splits_disjoint(self):
        """Train, val, test indices must be disjoint."""
        ds = self._make_studentt()
        train = set(ds._train_idx.tolist())
        val = set(ds._val_idx.tolist())
        test = set(ds._test_idx.tolist())
        assert len(train & val) == 0
        assert len(train & test) == 0
        assert len(val & test) == 0

    def test_sample_shapes(self):
        ds = self._make_studentt()
        assert ds.sample_train(32).shape == (32, 4)
        assert ds.sample_val(16).shape == (16, 4)
        assert ds.sample_test(8).shape == (8, 4)

    def test_reproducible_splits(self):
        """Same split_seed should give same splits."""
        ds1 = StudentT(dim=4, df=3.0, n_samples=2_000, split_seed=0)
        ds2 = StudentT(dim=4, df=3.0, n_samples=2_000, split_seed=0)
        assert torch.equal(ds1._train_idx, ds2._train_idx)
        assert torch.equal(ds1._test_idx, ds2._test_idx)

    def test_different_seeds_give_different_splits(self):
        ds1 = StudentT(dim=4, df=3.0, n_samples=2_000, split_seed=0)
        ds2 = StudentT(dim=4, df=3.0, n_samples=2_000, split_seed=99)
        assert not torch.equal(ds1._train_idx, ds2._train_idx)


class TestToyDataset:
    def test_shape_and_dim(self):
        ds = ToyRadialAngular(n_samples=2_000)
        assert ds.dim == 2
        assert ds.get_train_data().shape[1] == 2

    def test_no_nan(self):
        ds = ToyRadialAngular(n_samples=2_000)
        assert not torch.isnan(ds._data).any()


class TestStudentT:
    def test_heavy_tail(self):
        """Student-t(df=3) should have heavier tail than Gaussian."""
        ds_t = StudentT(dim=8, df=3.0, n_samples=20_000)
        r_t = torch.norm(ds_t.get_test_data(), dim=-1)
        # 99th percentile should be larger than what N(0,I_8) would give
        from scipy.stats import chi
        chi_99 = chi.ppf(0.99, df=8)
        assert float(torch.quantile(r_t, 0.99)) > chi_99


class TestGaussianAniso:
    def test_shape(self):
        ds = GaussianAniso(dim=16, n_samples=3_000)
        assert ds.dim == 16
        assert ds.get_train_data().shape[1] == 16

    def test_reproducible_matrix(self):
        ds1 = GaussianAniso(dim=8, matrix_seed=42)
        ds2 = GaussianAniso(dim=8, matrix_seed=42)
        assert torch.allclose(ds1.A, ds2.A)
