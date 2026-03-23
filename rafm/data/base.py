"""Abstract base class for datasets with strict train/val/test split enforcement."""
from abc import ABC, abstractmethod

import torch
from torch import Tensor


class BaseDataset(ABC):
    """Base dataset with fixed, reproducible train/val/test splits.

    Subclasses must:
      - set self.name, self.dim
      - implement _load_all_data() -> Tensor of shape (N, d)
      - call self._make_splits() in __init__ after loading data

    Splits are always deterministic given the dataset and split ratios.
    The empirical radial law MUST be estimated only from self.sample_train().
    """

    name: str
    dim: int

    def _make_splits(
        self,
        data: Tensor,
        train_frac: float = 0.6,
        val_frac: float = 0.2,
        # test_frac = 1 - train_frac - val_frac
        split_seed: int = 0,
    ) -> None:
        """Create fixed index-based splits. Split seed is separate from model seed."""
        n = data.shape[0]
        rng = torch.Generator()
        rng.manual_seed(split_seed)
        perm = torch.randperm(n, generator=rng)

        n_train = int(n * train_frac)
        n_val = int(n * val_frac)

        self._train_idx = perm[:n_train]
        self._val_idx = perm[n_train:n_train + n_val]
        self._test_idx = perm[n_train + n_val:]
        self._data = data

    def sample_train(self, n: int) -> Tensor:
        idx = self._train_idx[torch.randint(len(self._train_idx), (n,))]
        return self._data[idx]

    def sample_val(self, n: int) -> Tensor:
        idx = self._val_idx[torch.randint(len(self._val_idx), (n,))]
        return self._data[idx]

    def sample_test(self, n: int) -> Tensor:
        idx = self._test_idx[torch.randint(len(self._test_idx), (n,))]
        return self._data[idx]

    def get_test_data(self) -> Tensor:
        """Return the full test split (for final metric computation)."""
        return self._data[self._test_idx]

    def get_train_data(self) -> Tensor:
        """Return the full train split (for fitting empirical radial source)."""
        return self._data[self._train_idx]

    @property
    def n_train(self) -> int:
        return len(self._train_idx)

    @property
    def n_val(self) -> int:
        return len(self._val_idx)

    @property
    def n_test(self) -> int:
        return len(self._test_idx)
