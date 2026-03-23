"""Generate LaTeX tables from aggregated results.

Usage:
    python scripts/generate_tables.py --experiment exp1_main_benchmark
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.aggregate_results import aggregate
from rafm.plotting.tables import df_to_latex

# Metrics to include in the main table
PRIMARY_METRICS = [
    "radial_w1", "ks_stat", "q950_err", "q990_err", "q995_err",
    "tail_exc_95", "sliced_w1", "mmd",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", default="exp1_main_benchmark")
    parser.add_argument("--output_dir", default="outputs")
    parser.add_argument("--out_dir", default="figures/tables")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = aggregate(args.output_dir, args.experiment)
    if df.empty:
        print("No results.")
        return

    # Keep only mean values (not std) for the display columns
    display_cols = [c for c in df.columns if not c.endswith("_mean") and
                    not c.endswith("_std") and c.split("_mean")[0] in PRIMARY_METRICS]

    if not display_cols:
        # Fallback: use all string columns
        display_cols = [c for c in df.columns if "±" in str(df[c].iloc[0] if len(df) > 0 else "")]

    if display_cols:
        df_display = df[display_cols]
    else:
        df_display = df

    latex = df_to_latex(
        df_display,
        highlight_best=True,
        lower_is_better=list(df_display.columns),
        out_path=out_dir / f"{args.experiment}_main.tex",
    )
    print(latex)
    print(f"\nTable saved to {out_dir}/{args.experiment}_main.tex")


if __name__ == "__main__":
    main()
