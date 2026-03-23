"""Prepare the public PIV dataset for use in RAFM experiments.

Dataset: "Non-time-resolved PIV dataset of flow over a circular cylinder
         at Reynolds number 3900", DOI 10.57745/DHJXM6

This script:
  1. Loads *_vortdiv.npy files from the raw PIV folder (similar to MSGM data.py)
  2. Applies the same preprocessing as MSGM: divide by 2.5, center
  3. Exports piv_d32.pt (32 dimensions) and piv_d16.pt (16 dimensions)

Usage:
    python -m rafm.data.prepare_piv --raw_dir /path/to/raw/piv --out_dir /path/to/output

The raw data must be downloaded manually from DOI 10.57745/DHJXM6.
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def prepare_piv(raw_dir: str | Path, out_dir: str | Path) -> None:
    raw_dir = Path(raw_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prefix = "Serie_"
    npdata = np.empty((32, 0))

    files = sorted(raw_dir.glob(prefix + "*_vortdiv.npy"))
    if not files:
        raise FileNotFoundError(f"No *_vortdiv.npy files found in {raw_dir}")

    print(f"Loading {len(files)} files from {raw_dir} ...")
    for f in files:
        dataPt = np.load(f)
        if np.any(np.isnan(dataPt)):
            print(f"  Warning: NaN in {f.name}, skipping")
            continue
        npdata = np.concatenate((npdata, dataPt.reshape(-1, 1)), axis=1)

    npdata = npdata.T / 2.5                        # (N, 32), normalize like MSGM
    npdata = npdata - npdata.mean(axis=0)           # center

    data = torch.from_numpy(npdata).float()         # (N, 32)
    print(f"Final data shape: {data.shape}")

    # Save d=32 and d=16 versions
    for d in [32, 16]:
        out_path = out_dir / f"piv_d{d}.pt"
        torch.save(data[:, :d], out_path)
        print(f"Saved {out_path} — shape {data[:, :d].shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", required=True, help="Directory with *_vortdiv.npy files")
    parser.add_argument("--out_dir", required=True, help="Output directory for .pt files")
    args = parser.parse_args()
    prepare_piv(args.raw_dir, args.out_dir)
