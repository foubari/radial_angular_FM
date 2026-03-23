"""Generate LaTeX tables from aggregated CSV results."""
import pandas as pd
from pathlib import Path


def df_to_latex(
    df: pd.DataFrame,
    highlight_best: bool = True,
    lower_is_better: list[str] | None = None,
    out_path: Path | None = None,
) -> str:
    """Convert a DataFrame of metric results to a LaTeX table.

    Args:
        df:              DataFrame with methods as rows, metrics as columns.
                         Values should be "mean +- std" strings or floats.
        highlight_best:  bold the best value per column
        lower_is_better: list of metric names where lower = better (default: all)
        out_path:        save .tex file

    Returns:
        LaTeX table string
    """
    if lower_is_better is None:
        lower_is_better = list(df.columns)

    if highlight_best:
        df_display = df.copy()
        for col in df.columns:
            if col not in lower_is_better:
                continue
            # Extract numeric values from "mean ± std" strings if needed
            try:
                vals = df[col].apply(lambda x: float(str(x).split("±")[0].strip())
                                     if "±" in str(x) else float(x))
                best_idx = vals.idxmin()
                df_display.loc[best_idx, col] = r"\textbf{" + str(df_display.loc[best_idx, col]) + r"}"
            except (ValueError, TypeError):
                pass
    else:
        df_display = df

    latex = df_display.to_latex(escape=False, index=True)

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(latex)

    return latex


def aggregate_results(results_dir: Path, experiment: str) -> pd.DataFrame:
    """Load all metrics.json files for an experiment and aggregate across seeds.

    Returns a DataFrame with methods as rows and metrics (mean ± std) as columns.
    """
    import json, glob
    import numpy as np

    pattern = str(results_dir / experiment / "*" / "*" / "seed_*" / "metrics.json")
    files = glob.glob(pattern)

    records = []
    for f in files:
        parts = Path(f).parts
        # Extract dataset and method from path
        seed_idx = next(i for i, p in enumerate(parts) if p.startswith("seed_"))
        method = parts[seed_idx - 1]
        dataset = parts[seed_idx - 2]
        with open(f) as fp:
            metrics = json.load(fp)
        records.append({"dataset": dataset, "method": method, **metrics})

    if not records:
        return pd.DataFrame()

    df = pd.DataFrame(records)
    metric_cols = [c for c in df.columns if c not in ("dataset", "method")]

    rows = []
    for (dataset, method), grp in df.groupby(["dataset", "method"]):
        row = {"dataset": dataset, "method": method}
        for col in metric_cols:
            vals = grp[col].dropna().values
            if len(vals) > 0:
                row[col] = f"{np.mean(vals):.4f} ± {np.std(vals):.4f}"
        rows.append(row)

    return pd.DataFrame(rows).set_index(["dataset", "method"])
