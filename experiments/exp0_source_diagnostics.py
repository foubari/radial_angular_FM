"""Experiment 0: Source diagnostics — no training required.

For each dataset:
  1. Compute empirical radial CDF of test data
  2. Compare against: Gaussian source, oracle radial source (synthetic only),
     empirical radial sources (eCDF, log-eCDF) fitted on train split
  3. Report KS statistic and W1 between each source's radial law and test data
  4. Plot CDF, survival function, and norm histogram

Usage:
    python -m experiments.exp0_source_diagnostics
    python -m experiments.exp0_source_diagnostics --config configs/exp0/source_diagnostics.yaml
"""
import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config, get_run_dir, save_config
from rafm.utils.seeds import set_all_seeds
from rafm.metrics.radial import _ks_stat, _wasserstein1_1d
from rafm.plotting.radial_cdf import plot_radial_cdf, plot_radial_survival
from rafm.plotting.pairplots import plot_norm_histogram


DATASETS_FACTORY = {
    "toy_radial_angular": lambda cfg: _make_toy(cfg),
    "student_t":          lambda cfg: _make_studentt(cfg),
    "gaussian_aniso":     lambda cfg: _make_gaussian(cfg),
    "piv":                lambda cfg: _make_piv(cfg),
}

IS_SYNTHETIC = {"toy_radial_angular", "student_t", "gaussian_aniso"}


def _make_toy(cfg):
    from rafm.data.toy_radial_angular import ToyRadialAngular
    return ToyRadialAngular(
        n_samples=cfg.get("n_samples", 50_000),
        df=cfg.get("df", 3.0),
        split_seed=cfg.get("split_seed", 0),
    )

def _make_studentt(cfg):
    from rafm.data.student_t import StudentT
    return StudentT(
        dim=cfg["dim"], df=cfg.get("df", 3.0),
        n_samples=cfg.get("n_samples", 50_000),
        correlated=cfg.get("correlated", True),
        split_seed=cfg.get("split_seed", 0),
    )

def _make_gaussian(cfg):
    from rafm.data.gaussian_aniso import GaussianAniso
    return GaussianAniso(
        dim=cfg["dim"],
        n_samples=cfg.get("n_samples", 50_000),
        split_seed=cfg.get("split_seed", 0),
    )

def _make_piv(cfg):
    from rafm.data.piv import PIVDataset
    return PIVDataset(
        data_root=cfg["data_root"], dim=cfg["dim"],
        split_seed=cfg.get("split_seed", 0),
    )


def run_exp0(cfg: dict) -> None:
    set_all_seeds(cfg.get("base_seed", 42))
    out_base = Path(cfg.get("output_dir", "outputs")) / "exp0"
    fig_base = Path(cfg.get("figures_dir", "figures")) / "exp0"
    n_source = cfg.get("n_source_samples", 10_000)

    for ds_cfg in cfg["datasets"]:
        ds_name = ds_cfg["name"]
        dim = ds_cfg.get("dim", 2)
        tag = f"{ds_name}_d{dim}"
        print(f"\n{'='*50}")
        print(f"Dataset: {tag}")

        dataset = DATASETS_FACTORY[ds_name](ds_cfg)
        train_data = dataset.get_train_data()
        test_data = dataset.get_test_data()
        r_test = torch.norm(test_data, dim=-1)

        radii_dict = {"data": r_test}
        results = {}

        # --- Gaussian source ---
        from rafm.sources.gaussian import GaussianSource
        gauss = GaussianSource()
        x_gauss = gauss.sample(n_source, dim)
        r_gauss = torch.norm(x_gauss, dim=-1)
        radii_dict["gaussian_fm"] = r_gauss
        results["gaussian_fm"] = {
            "ks": _ks_stat(r_gauss, r_test),
            "w1": _wasserstein1_1d(r_gauss, r_test),
        }

        # --- Oracle source (synthetic only) ---
        if ds_name in IS_SYNTHETIC:
            from rafm.sources.radial_oracle import (
                RadialOracleSource, student_t_radial_sampler,
                gaussian_aniso_radial_sampler, toy_radial_sampler
            )
            if ds_name == "student_t":
                sampler = student_t_radial_sampler(ds_cfg.get("df", 3.0), dim, dataset.A)
            elif ds_name == "gaussian_aniso":
                sampler = gaussian_aniso_radial_sampler(dim, dataset.A)
            else:
                sampler = toy_radial_sampler(ds_cfg.get("df", 3.0))

            oracle_src = RadialOracleSource(sampler)
            x_oracle = oracle_src.sample(n_source, dim)
            r_oracle = torch.norm(x_oracle, dim=-1)
            radii_dict["rafm_oracle"] = r_oracle
            results["radial_oracle"] = {
                "ks": _ks_stat(r_oracle, r_test),
                "w1": _wasserstein1_1d(r_oracle, r_test),
            }

        # --- Empirical source (eCDF) ---
        from rafm.sources.radial_empirical import RadialEmpiricalSource
        emp_ecdf = RadialEmpiricalSource(mode="ecdf").fit(train_data)
        x_emp_ecdf = emp_ecdf.sample(n_source, dim)
        r_emp_ecdf = torch.norm(x_emp_ecdf, dim=-1)
        radii_dict["source_only_empirical"] = r_emp_ecdf
        results["radial_empirical_ecdf"] = {
            "ks": _ks_stat(r_emp_ecdf, r_test),
            "w1": _wasserstein1_1d(r_emp_ecdf, r_test),
        }

        # --- Empirical source (log-eCDF) ---
        emp_log = RadialEmpiricalSource(mode="ecdf", log_radius=True).fit(train_data)
        x_emp_log = emp_log.sample(n_source, dim)
        r_emp_log = torch.norm(x_emp_log, dim=-1)
        radii_dict["rafm_empirical"] = r_emp_log
        results["radial_empirical_log"] = {
            "ks": _ks_stat(r_emp_log, r_test),
            "w1": _wasserstein1_1d(r_emp_log, r_test),
        }

        # --- Print results ---
        print(f"{'Source':<30} {'KS':>10} {'W1':>10}")
        print("-" * 52)
        for src_name, vals in results.items():
            print(f"{src_name:<30} {vals['ks']:>10.4f} {vals['w1']:>10.4f}")

        # --- Save metrics ---
        out_dir = out_base / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        import json
        (out_dir / "source_metrics.json").write_text(json.dumps(results, indent=2))

        # --- Figures ---
        plot_radial_cdf(radii_dict, title=tag,
                        out_path=fig_base / tag / "radial_cdf.pdf")
        plot_radial_survival(radii_dict, title=tag,
                             out_path=fig_base / tag / "radial_survival.pdf")
        plot_norm_histogram(radii_dict, log_scale=False,
                            out_path=fig_base / tag / "norm_hist.pdf")
        plot_norm_histogram(radii_dict, log_scale=True,
                            out_path=fig_base / tag / "log_norm_hist.pdf")

        print(f"Figures saved to {fig_base / tag}/")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp0/source_diagnostics.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_exp0(cfg)


if __name__ == "__main__":
    main()
