"""Experiment 1: Main generation benchmark.

For each dataset config, for each method, for each seed:
  - Train the model
  - Generate samples
  - Evaluate all metrics on test data
  - Save everything

Usage:
    python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d16.yaml
    python -m experiments.exp1_main_benchmark --config configs/exp1/toy2d.yaml --method gaussian_fm
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config, get_run_dir, save_samples
from rafm.utils.seeds import set_all_seeds, get_seed_list
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics


def build_source(source_name: str, train_data: torch.Tensor | None, dataset):
    if source_name == "gaussian":
        from rafm.sources.gaussian import GaussianSource
        return GaussianSource()
    elif source_name == "radial_oracle":
        from rafm.sources.radial_oracle import (
            RadialOracleSource, student_t_radial_sampler,
            gaussian_aniso_radial_sampler, toy_radial_sampler
        )
        ds_name = dataset.name
        dim = dataset.dim
        if "student_t" in ds_name:
            sampler = student_t_radial_sampler(
                getattr(dataset, "df", 3.0), dim, dataset.A
            )
        elif "gaussian" in ds_name:
            sampler = gaussian_aniso_radial_sampler(dim, dataset.A)
        else:
            sampler = toy_radial_sampler(getattr(dataset, "df", 3.0))
        return RadialOracleSource(sampler)
    elif source_name in ("radial_empirical_ecdf", "radial_empirical"):
        from rafm.sources.radial_empirical import RadialEmpiricalSource
        return RadialEmpiricalSource(mode="ecdf").fit(train_data)
    elif source_name == "radial_empirical_log":
        from rafm.sources.radial_empirical import RadialEmpiricalSource
        return RadialEmpiricalSource(mode="ecdf", log_radius=True).fit(train_data)
    else:
        raise ValueError(f"Unknown source: {source_name}")


def build_path(path_name: str):
    if path_name == "euclidean":
        from rafm.paths.euclidean import EuclideanPath
        return EuclideanPath()
    elif path_name == "spherical_geodesic":
        from rafm.paths.spherical_geodesic import SphericalGeodesicPath
        return SphericalGeodesicPath()
    else:
        raise ValueError(f"Unknown path: {path_name}")


def run_one_method(method_cfg: dict, dataset, cfg: dict, seed: int, out_base: Path) -> None:
    method_name = method_cfg["name"]

    # MSGM uses a separate adapter
    if method_cfg.get("adapter") == "msgm":
        from baselines.msgm_runner import run_msgm
        run_msgm(dataset, cfg, seed, out_base / method_name / f"seed_{seed}")
        return

    set_all_seeds(seed)
    dim = dataset.dim
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_data = dataset.get_train_data()
    test_data = dataset.get_test_data()

    source = build_source(method_cfg["source"], train_data, dataset)
    path = build_path(method_cfg["path"])

    from rafm.models.mlp import MLP
    model = MLP(
        input_dim=dim,
        hidden_dim=cfg.get("hidden_dim", 128),
        n_layers=cfg.get("n_layers", 3),
        premodule=cfg.get("premodule", None),
    )

    run_dir = get_run_dir(out_base, "", "", method_name, seed)
    run_dir.mkdir(parents=True, exist_ok=True)

    from rafm.flow_matching.trainer import Trainer
    trainer = Trainer(model, path, source, dataset, cfg, seed, run_dir)
    train_stats = trainer.train()

    from rafm.flow_matching.sampler import Sampler
    sample_cfg = dict(cfg)
    sample_cfg["path"] = method_cfg.get("path", "euclidean")
    sampler = Sampler(model, source, sample_cfg)
    n_gen = cfg.get("n_gen_samples", 10_000)
    gen_result = sampler.sample(n_gen, dim)
    samples = gen_result["samples"]

    save_samples(samples, run_dir / "samples.pt")

    # Evaluate — all metrics on TEST data only
    metrics = {}
    metrics.update(radial_metrics(samples, test_data))
    metrics.update(distributional_metrics(
        samples, test_data, n_projections=cfg.get("n_projections_sw", 500)
    ))
    metrics.update(angular_metrics(
        samples, test_data,
        n_bins=cfg.get("n_angular_bins", 4),
        n_projections=cfg.get("n_projections_angular", 200),
    ))
    metrics.update(stability_metrics(samples))
    metrics["nfe"] = gen_result["nfe"]
    metrics["sample_time_s"] = gen_result["sample_time_s"]
    metrics["total_train_time_s"] = train_stats["total_train_time_s"]

    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"  [{method_name} seed={seed}] radial_w1={metrics['radial_w1']:.4f} "
          f"ks={metrics['ks_stat']:.4f} sliced_w1={metrics['sliced_w1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--method", default=None, help="Run only this method name")
    parser.add_argument("--exclude", default=None, help="Comma-separated method names to skip")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_base = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]

    # Build dataset
    ds_cfg = cfg["dataset"]
    from experiments.exp0_source_diagnostics import DATASETS_FACTORY
    dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
    out_base = out_base / dataset.name

    seeds = get_seed_list(cfg.get("n_seeds", 5), cfg.get("base_seed", 42))

    methods = cfg["methods"]
    if args.method:
        methods = [m for m in methods if m["name"] == args.method]
    if args.exclude:
        excluded = {s.strip() for s in args.exclude.split(",")}
        methods = [m for m in methods if m["name"] not in excluded]

    for method_cfg in methods:
        print(f"\nMethod: {method_cfg['name']}")
        for seed in seeds:
            run_one_method(method_cfg, dataset, cfg, seed, out_base)


if __name__ == "__main__":
    main()
