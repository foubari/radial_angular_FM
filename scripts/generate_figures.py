"""Generate publication figures from aggregated results.

Usage:
    python scripts/generate_figures.py --experiment exp1_main_benchmark
    python scripts/generate_figures.py --all
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))


def plot_sample_efficiency(results_dir: str, out_dir: Path) -> None:
    """Plot metric vs n_train for Exp 2."""
    import json, glob
    import numpy as np
    import matplotlib.pyplot as plt
    from rafm.plotting.style import set_style, PALETTE, METHOD_LABELS

    set_style()
    pattern = str(Path(results_dir) / "exp2_sample_efficiency" / "*" / "n*" / "*" / "seed_*" / "metrics.json")
    files = glob.glob(pattern)

    data = {}  # {method: {n_train: [metric_val]}}
    for f in files:
        parts = Path(f).parts
        seed_idx = next((i for i, p in enumerate(parts) if p.startswith("seed_")), None)
        if seed_idx is None:
            continue
        method = parts[seed_idx - 1]
        n_str = parts[seed_idx - 2]
        n_train = int(n_str.replace("n", ""))
        with open(f) as fp:
            m = json.load(fp)
        data.setdefault(method, {}).setdefault(n_train, []).append(m.get("radial_w1", float("nan")))

    fig, ax = plt.subplots(figsize=(4, 3))
    for method, n_data in sorted(data.items()):
        ns = sorted(n_data.keys())
        means = [np.mean(n_data[n]) for n in ns]
        stds = [np.std(n_data[n]) for n in ns]
        color = PALETTE.get(method, "gray")
        ax.plot(ns, means, "-o", color=color, label=METHOD_LABELS.get(method, method))
        ax.fill_between(ns, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.15, color=color)

    ax.set_xscale("log")
    ax.set_xlabel("n_train")
    ax.set_ylabel("Radial W1")
    ax.legend()
    ax.set_title("Sample efficiency")
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "sample_efficiency_radial_w1.pdf")
    plt.close(fig)
    print(f"Saved {out_dir}/sample_efficiency_radial_w1.pdf")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp1_main_benchmark")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--figures_dir", default="figures")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.figures_dir)

    if args.all or args.experiment == "exp2_sample_efficiency":
        plot_sample_efficiency(args.output_dir, out_dir / "exp2")


if __name__ == "__main__":
    main()
