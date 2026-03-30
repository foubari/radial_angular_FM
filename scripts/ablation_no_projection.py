"""Ablation: RAFM sampling WITHOUT tangent projection.

Reuses already-trained RAFM checkpoints and re-samples with project_tangent=False.
Saves results as 'rafm_empirical_no_proj' alongside existing results.

Usage:
    python scripts/ablation_no_projection.py
    python scripts/ablation_no_projection.py --configs configs/exp1/studentt_d16.yaml,configs/exp1/studentt_d32.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config, save_samples
from rafm.utils.seeds import get_seed_list
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.angular import angular_metrics
from rafm.metrics.stability import stability_metrics
from rafm.flow_matching.sampler import Sampler
from rafm.models.mlp import MLP


DEFAULT_CONFIGS = [
    "configs/exp1/toy2d.yaml",
    "configs/exp1/studentt_d16.yaml",
    "configs/exp1/studentt_d32.yaml",
    "configs/exp1/gaussian_d16.yaml",
    "configs/exp1/piv_d16.yaml",
    "configs/exp1/piv_d64.yaml",
    "configs/exp1/piv_d256.yaml",
]


def run_ablation(config_path: str) -> None:
    cfg = load_config(config_path)
    out_base = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]

    ds_cfg = cfg["dataset"]
    from experiments.exp0_source_diagnostics import DATASETS_FACTORY
    dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
    out_base = out_base / dataset.name

    seeds = get_seed_list(cfg.get("n_seeds", 3), cfg.get("base_seed", 42))
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\nDataset: {dataset.name}")

    for seed in seeds:
        out_dir = out_base / "rafm_empirical_no_proj" / f"seed_{seed}"
        if (out_dir / "metrics.json").exists():
            print(f"  [no_proj seed={seed}] already done, skipping")
            continue

        # Load original RAFM checkpoint
        ckpt_path = out_base / "rafm_empirical" / f"seed_{seed}" / "checkpoint.pt"
        if not ckpt_path.exists():
            print(f"  [no_proj seed={seed}] checkpoint not found at {ckpt_path}, skipping")
            continue

        dim = dataset.dim
        model = MLP(
            input_dim=dim,
            hidden_dim=cfg.get("hidden_dim", 128),
            n_layers=cfg.get("n_layers", 3),
            premodule=cfg.get("premodule", None),
        ).to(device)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        model.load_state_dict(state_dict)

        # Build empirical source (same as training)
        from rafm.sources.radial_empirical import RadialEmpiricalSource
        train_data = dataset.get_train_data()
        source = RadialEmpiricalSource(mode="ecdf").fit(train_data)

        # Sample WITHOUT projection
        sample_cfg = dict(cfg)
        sample_cfg["path"] = "spherical_geodesic"
        sampler = Sampler(model, source, sample_cfg, project_tangent=False)

        n_gen = cfg.get("n_gen_samples", 10_000)
        test_data = dataset.get_test_data()
        gen_result = sampler.sample(n_gen, dim)
        samples = gen_result["samples"]

        out_dir.mkdir(parents=True, exist_ok=True)
        save_samples(samples, out_dir / "samples.pt")

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
        metrics["project_tangent"] = False

        (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
        print(f"  [no_proj seed={seed}] radial_w1={metrics['radial_w1']:.4f} "
              f"ks={metrics['ks_stat']:.4f} sliced_w1={metrics['sliced_w1']:.4f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--configs", default=None,
        help="Comma-separated list of config paths. Default: all exp1 configs."
    )
    args = parser.parse_args()

    configs = args.configs.split(",") if args.configs else DEFAULT_CONFIGS
    for config_path in configs:
        run_ablation(config_path.strip())


if __name__ == "__main__":
    main()
