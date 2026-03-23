"""Pairwise scatter plots comparing generated vs test distributions."""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from rafm.plotting.style import PALETTE, set_style


def plot_pairplot(
    samples: torch.Tensor,
    test_data: torch.Tensor,
    n_show: int = 2000,
    dims: list[int] | None = None,
    title: str = "",
    out_path: Path | None = None,
) -> plt.Figure:
    """2D scatter / density comparison of first few dimensions.

    Args:
        samples:   (n_gen, d)
        test_data: (n_test, d)
        n_show:    subsample for display
        dims:      which dimensions to show (default [0, 1])
        title:     figure title
        out_path:  save path
    """
    set_style()
    dims = dims or [0, 1]
    assert len(dims) == 2

    s = samples[:n_show, dims].float().numpy()
    t = test_data[:n_show, dims].float().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    for ax, data, label in zip(axes, [t, s], ["Test data", "Generated"]):
        ax.hexbin(data[:, 0], data[:, 1], gridsize=40, cmap="Blues", bins="log")
        ax.set_title(label)
        ax.set_xlabel(f"dim {dims[0]}")
        ax.set_ylabel(f"dim {dims[1]}")
        ax.set_aspect("equal")

    if title:
        fig.suptitle(title)
    plt.tight_layout()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    return fig


def plot_norm_histogram(
    radii_dict: dict,
    log_scale: bool = False,
    out_path: Path | None = None,
) -> plt.Figure:
    """Histogram of norms for multiple sources."""
    set_style()
    fig, ax = plt.subplots(figsize=(4, 3))

    for label, radii in radii_dict.items():
        r = radii.float().numpy() if hasattr(radii, "numpy") else np.array(radii)
        if log_scale:
            r = np.log(r + 1e-6)
        color = PALETTE.get(label, "gray")
        ax.hist(r, bins=80, alpha=0.5, density=True, color=color, label=label)

    ax.set_xlabel(r"$\log(\|x\| + \epsilon)$" if log_scale else r"$\|x\|$")
    ax.set_ylabel("density")
    ax.legend()

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_path)
        plt.close(fig)

    return fig
