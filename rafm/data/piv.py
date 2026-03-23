"""PIV dataset loader with configurable data_root.

Requires running prepare_piv.py first to preprocess the public PIV dataset
(DOI 10.57745/DHJXM6) into a local .pt file.

The original MSGM code (data.py) used a hard-coded local path:
  /Users/vresseiguier/Coding/MultiplicativeDiffusion/newPIV
This loader replaces that with a configurable `data_root` argument.
"""
from pathlib import Path

import torch
from torch import Tensor

from rafm.data.base import BaseDataset


class PIVDataset(BaseDataset):
    """PIV vorticity dataset.

    Args:
        data_root:  path to directory containing 'piv_d{dim}.pt'
                    (created by prepare_piv.py)
        dim:        number of dimensions to use (e.g. 16 or 32)
        split_seed: seed for train/val/test split
    """
    dim: int

    def __init__(
        self,
        data_root: str | Path,
        dim: int = 16,
        split_seed: int = 0,
    ):
        self.dim = dim
        self.name = f"piv_d{dim}"
        data_root = Path(data_root)
        pt_file = data_root / f"piv_d{dim}.pt"

        if not pt_file.exists():
            raise FileNotFoundError(
                f"Preprocessed PIV file not found: {pt_file}\n"
                f"Run: python -m rafm.data.prepare_piv --data_root {data_root}"
            )

        data = torch.load(pt_file)  # (N, 32) or similar
        assert data.shape[1] >= dim, f"PIV data has {data.shape[1]} dims, requested {dim}"
        data = data[:, :dim].float()

        # Center (mean already removed by prepare_piv.py, but ensure)
        data = data - data.mean(dim=0, keepdim=True)

        self._make_splits(data, split_seed=split_seed)
