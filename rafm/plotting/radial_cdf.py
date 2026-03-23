"""Radial CDF overlay plots for Experiment 0."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from rafm.plotting.style import PALETTE, LINE_STYLES, METHOD_LABELS, set_style


def plot_radial_cdf(
    radii_dict: dict,
    title: str = "",
    out_path: Path | None = None,
) -> plt.Figure:
    """Overlay CDFs of ||x|| for multiple sources and the data.

    Args:
        radii_dict: {label: Tensor of shape (n,)} — norms for each source/data
                    e.g. {"data": r_test, "gaussian_fm": r_gauss, "rafm_oracle": r_oracle, ...}
        title:      figure title
        out_path:   if provided, save figure to this path

    Returns:
        matplotlib Figure
    """
    set_style()
    fig, ax = plt.subplots(figsize=(4, 3))

    for label, radii in radii_dict.items():
        r = radii.float().numpy() if hasattr(radii, "numpy") else np.array(radii)
        r_sorted = np.sort(r)
        cdf = np.arange(1, len(r_sorted) + 1) / len(r_sorted)

        color = PALETTE.get(label, "gray")
        ls = LINE_STYLES.get(label, "-")
        lw = 2.0 if label == "data" else 1.5
        ax.plot(r_sorted, cdf, color=color, linestyle=ls, linewidth=lw,
                label=METHOD_LABELS.get(label, label))

    ax.set_xlabel(r"$\|x\|$")
    ax.set_ylabel("CDF")
    ax.legend(loc="lower right", framealpha=0.9)
    if title:
        ax.set_title(title)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    return fig


def plot_radial_survival(
    radii_dict: dict,
    title: str = "",
    out_path: Path | None = None,
) -> plt.Figure:
    """Survival function (1 - CDF) on log-log scale to highlight tail behavior."""
    set_style()
    fig, ax = plt.subplots(figsize=(4, 3))

    for label, radii in radii_dict.items():
        r = radii.float().numpy() if hasattr(radii, "numpy") else np.array(radii)
        r_sorted = np.sort(r)
        sf = 1.0 - np.arange(1, len(r_sorted) + 1) / len(r_sorted)

        color = PALETTE.get(label, "gray")
        ls = LINE_STYLES.get(label, "-")
        lw = 2.0 if label == "data" else 1.5
        ax.plot(r_sorted, sf, color=color, linestyle=ls, linewidth=lw,
                label=METHOD_LABELS.get(label, label))

    ax.set_xlabel(r"$\|x\|$")
    ax.set_ylabel(r"$P(\|X\| > r)$")
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend(loc="upper right", framealpha=0.9)
    if title:
        ax.set_title(title)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    return fig
