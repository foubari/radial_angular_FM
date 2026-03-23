"""Experiment 2: Sample efficiency of empirical radial source estimation.

Varies n_train in [500, 1000, 5000, 20000] and measures how empirical
RAFM converges toward oracle RAFM as training data increases.

Usage:
    python -m experiments.exp2_sample_efficiency
    python -m experiments.exp2_sample_efficiency --config configs/exp2/sample_efficiency.yaml
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
from rafm.metrics.stability import stability_metrics
from experiments.exp1_main_benchmark import build_source, build_path


def run_exp2(cfg: dict) -> None:
    out_base = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]
    fig_base = Path(cfg.get("figures_dir", "figures")) / cfg["experiment"]

    ds_cfg = cfg["dataset"]
    from experiments.exp0_source_diagnostics import DATASETS_FACTORY
    dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
    out_base = out_base / dataset.name

    seeds = get_seed_list(cfg.get("n_seeds", 3), cfg.get("base_seed", 42))
    n_train_sizes = cfg["n_train_sizes"]
    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    test_data = dataset.get_test_data()
    all_train_data = dataset.get_train_data()  # full train split

    all_results = {}

    for n_train in n_train_sizes:
        print(f"\n--- n_train = {n_train} ---")
        # Subsample train data (from train split only)
        if n_train > len(all_train_data):
            print(f"  Warning: n_train={n_train} > available train={len(all_train_data)}, using all")
            train_subset = all_train_data
        else:
            idx = torch.randperm(len(all_train_data))[:n_train]
            train_subset = all_train_data[idx]

        for method_cfg in cfg["methods"]:
            method_name = method_cfg["name"]
            for seed in seeds:
                set_all_seeds(seed)

                source_name = method_cfg.get("source", "gaussian")
                # Use only the train subset for radial estimation
                source = build_source(source_name, train_subset, dataset)
                path = build_path(method_cfg.get("path", "euclidean"))

                from rafm.models.mlp import MLP
                model = MLP(
                    input_dim=dataset.dim,
                    hidden_dim=cfg.get("hidden_dim", 128),
                    n_layers=cfg.get("n_layers", 3),
                )

                run_dir = out_base / f"n{n_train}" / method_name / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)

                # Temporarily modify cfg for this n_train run
                run_cfg = dict(cfg)
                run_cfg["_n_train_override"] = n_train

                from rafm.flow_matching.trainer import Trainer

                class SubsetDataset:
                    """Wrapper to use only the train subset."""
                    dim = dataset.dim
                    name = dataset.name
                    def sample_train(self, n):
                        idx = torch.randint(len(train_subset), (n,))
                        return train_subset[idx]
                    def sample_val(self, n):
                        return dataset.sample_val(n)
                    def get_train_data(self):
                        return train_subset

                trainer = Trainer(model, path, source, SubsetDataset(), run_cfg, seed, run_dir)
                train_stats = trainer.train()

                from rafm.flow_matching.sampler import Sampler
                sampler = Sampler(model, source, run_cfg)
                n_gen = cfg.get("n_gen_samples", 10_000)
                gen_result = sampler.sample(n_gen, dataset.dim)
                samples = gen_result["samples"]

                save_samples(samples, run_dir / "samples.pt")

                metrics = {}
                metrics.update(radial_metrics(samples, test_data))
                metrics.update(distributional_metrics(samples, test_data))
                metrics.update(stability_metrics(samples))
                metrics["n_train"] = n_train
                metrics["nfe"] = gen_result["nfe"]

                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
                key = f"{method_name}_n{n_train}_seed{seed}"
                all_results[key] = metrics
                print(f"  [{method_name} n={n_train} seed={seed}] "
                      f"radial_w1={metrics['radial_w1']:.4f}")

    (out_base / "all_results.json").write_text(json.dumps(all_results, indent=2))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp2/sample_efficiency.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_exp2(cfg)


if __name__ == "__main__":
    main()
