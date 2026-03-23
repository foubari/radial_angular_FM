"""Shared matplotlib style for NeurIPS publication figures."""
import matplotlib.pyplot as plt
import seaborn as sns

# Colorblind-safe palette (Wong 2011)
PALETTE = {
    "gaussian_fm":           "#0072B2",   # blue
    "source_only_oracle":    "#009E73",   # green
    "source_only_empirical": "#56B4E9",   # sky blue
    "rafm_oracle":           "#E69F00",   # orange
    "rafm_empirical":        "#D55E00",   # vermillion
    "msgm":                  "#CC79A7",   # reddish purple
    "student_t_prior":       "#F0E442",   # yellow
    "data":                  "#000000",   # black
}

LINE_STYLES = {
    "gaussian_fm":           "-",
    "source_only_oracle":    "--",
    "source_only_empirical": "-.",
    "rafm_oracle":           "-",
    "rafm_empirical":        "--",
    "msgm":                  ":",
    "data":                  "-",
}

METHOD_LABELS = {
    "gaussian_fm":           "Gaussian FM",
    "source_only_oracle":    "Source-only (oracle)",
    "source_only_empirical": "Source-only (empirical)",
    "rafm_oracle":           "RAFM (oracle)",
    "rafm_empirical":        "RAFM (empirical)",
    "msgm":                  "Mult. Diffusion",
    "student_t_prior":       "Student-t prior FM",
    "data":                  "Data",
}


def set_style(context: str = "paper") -> None:
    """Set matplotlib/seaborn style for NeurIPS figures."""
    sns.set_theme(style="whitegrid", context=context)
    plt.rcParams.update({
        "font.family":       "serif",
        "font.size":         9,
        "axes.labelsize":    9,
        "axes.titlesize":    10,
        "legend.fontsize":   8,
        "xtick.labelsize":   8,
        "ytick.labelsize":   8,
        "figure.dpi":        150,
        "savefig.dpi":       300,
        "savefig.bbox":      "tight",
        "savefig.format":    "pdf",
    })


# NeurIPS column widths in inches
SINGLE_COL_W = 3.25
DOUBLE_COL_W = 6.75
