"""Experiment 4 (optional): Solver sensitivity / quality vs NFE.

Fixes a trained RAFM model and evaluates generation quality at various NFE.
Shows the quality-vs-compute tradeoff for different solvers.

Usage:
    python -m experiments.exp4_solver_sensitivity --config configs/exp4/solver_sensitivity.yaml
"""
import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config, load_checkpoint
from rafm.utils.seeds import set_all_seeds, get_seed_list
from rafm.metrics.radial import radial_metrics
from rafm.metrics.distributional import distributional_metrics
from rafm.metrics.stability import stability_metrics
from experiments.exp1_main_benchmark import build_source, build_path


def run_exp4(cfg: dict) -> None:
    out_base = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]
    out_base.mkdir(parents=True, exist_ok=True)

    ds_cfg = cfg["dataset"]
    from experiments.exp0_source_diagnostics import DATASETS_FACTORY
    dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
    dim = dataset.dim
    test_data = dataset.get_test_data()
    train_data = dataset.get_train_data()

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    seeds = get_seed_list(cfg.get("n_seeds", 3), cfg.get("base_seed", 42))
    solvers = cfg.get("solvers", ["euler", "heun", "rk4"])
    nfe_list = cfg.get("nfe_list", [4, 8, 16, 32, 64, 128, 256])
    ckpt_path = cfg.get("checkpoint")

    all_results = {}

    for method_cfg in cfg["methods"]:
        method_name = method_cfg["name"]
        source = build_source(method_cfg["source"], train_data, dataset)

        from rafm.models.mlp import MLP
        model = MLP(input_dim=dim, hidden_dim=cfg.get("hidden_dim", 128))

        if ckpt_path:
            load_checkpoint(model, None, ckpt_path)
            print(f"Loaded checkpoint from {ckpt_path}")
        else:
            print("No checkpoint provided — training a fresh model for exp4...")
            set_all_seeds(seeds[0])
            path = build_path(method_cfg["path"])
            from rafm.flow_matching.trainer import Trainer
            trainer = Trainer(model, path, source, dataset, cfg, seeds[0],
                              out_base / method_name / "train")
            trainer.train()

        model = model.to(device)
        model.eval()

        for solver in solvers:
            for nfe in nfe_list:
                sample_cfg = dict(cfg)
                sample_cfg["solver"] = solver
                sample_cfg["nfe"] = nfe
                sample_cfg["device"] = device

                from rafm.flow_matching.sampler import Sampler
                sampler = Sampler(model, source, sample_cfg)
                n_gen = cfg.get("n_gen_samples", 10_000)

                key = f"{method_name}_{solver}_nfe{nfe}"
                gen = sampler.sample(n_gen, dim)
                samples = gen["samples"]

                metrics = {}
                metrics.update(radial_metrics(samples, test_data))
                metrics.update(distributional_metrics(samples, test_data))
                metrics.update(stability_metrics(samples))
                metrics["nfe"] = gen["nfe"]
                metrics["sample_time_s"] = gen["sample_time_s"]
                metrics["solver"] = solver

                all_results[key] = metrics
                print(f"  {key}: radial_w1={metrics['radial_w1']:.4f} "
                      f"sliced_w1={metrics['sliced_w1']:.4f} t={metrics['sample_time_s']:.2f}s")

    (out_base / "solver_sensitivity_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nResults saved to {out_base}/solver_sensitivity_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp4/solver_sensitivity.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_exp4(cfg)


if __name__ == "__main__":
    main()
