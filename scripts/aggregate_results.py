"""Aggregate results across seeds and generate summary CSV.

Usage:
    python scripts/aggregate_results.py --experiment exp1_main_benchmark
    python scripts/aggregate_results.py --experiment exp1_main_benchmark --output tables/
"""
import argparse
import json
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))


def aggregate(output_dir: str, experiment: str) -> pd.DataFrame:
    """Aggregate metrics.json files across seeds."""
    pattern = str(Path(output_dir) / experiment / "*" / "*" / "seed_*" / "metrics.json")
    files = glob.glob(pattern)

    if not files:
        print(f"No results found: {pattern}")
        return pd.DataFrame()

    records = []
    for f in files:
        parts = Path(f).parts
        seed_idx = next((i for i, p in enumerate(parts) if p.startswith("seed_")), None)
        if seed_idx is None:
            continue
        method = parts[seed_idx - 1]
        dataset = parts[seed_idx - 2]
        with open(f) as fp:
            metrics = json.load(fp)
        records.append({"dataset": dataset, "method": method, "file": f, **metrics})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c not in ("dataset", "method", "file")]

    rows = []
    for (dataset, method), grp in df.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method}
        for col in metric_cols:
            try:
                vals = grp[col].dropna().astype(float).values
                if len(vals) > 0:
                    row[f"{col}_mean"] = float(np.mean(vals))
                    row[f"{col}_std"] = float(np.std(vals))
                    row[f"{col}"] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
            except (TypeError, ValueError):
                pass
        rows.append(row)

    return pd.DataFrame(rows).set_index(["dataset", "method"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp1_main_benchmark")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--out", default=None, help="Output CSV path")
    args = parser.parse_args()

    df = aggregate(args.output_dir, args.experiment)
    if df.empty:
        print("No results to aggregate.")
        return

    print(df.to_string())

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path)
        print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
