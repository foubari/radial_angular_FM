"""Experiment 3: Compute / runtime / scalability comparison.

Measures:
  - Wall-clock time per training step
  - Total training time (extrapolated to 100k steps)
  - Sampling time for n_gen samples at various NFE
  - Peak GPU memory

Usage:
    python -m experiments.exp3_runtime --config configs/exp3/runtime.yaml
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from rafm.utils.io import load_config
from rafm.utils.seeds import set_all_seeds
from rafm.metrics.efficiency import get_peak_memory_mb, reset_memory_counter, Timer
from experiments.exp1_main_benchmark import build_source, build_path


def time_training(model, path, source, dataset, cfg, n_steps=1000, device="cpu"):
    """Measure per-step training time."""
    from rafm.flow_matching.loss import cfm_loss
    from torch.optim import Adam

    model = model.to(device)
    optimizer = Adam(model.parameters(), lr=cfg.get("lr", 1e-3))
    batch_size = cfg.get("batch_size", 256)

    # Warmup
    for _ in range(10):
        x1 = dataset.sample_train(batch_size).to(device)
        loss = cfm_loss(model, path, source, x1, device=device)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    reset_memory_counter(device)
    t0 = time.time()
    for _ in range(n_steps):
        x1 = dataset.sample_train(batch_size).to(device)
        optimizer.zero_grad()
        loss = cfm_loss(model, path, source, x1, device=device)
        loss.backward()
        optimizer.step()

    elapsed = time.time() - t0
    peak_mem = get_peak_memory_mb(device)
    return {"time_per_step_ms": elapsed / n_steps * 1000, "peak_memory_mb": peak_mem}


def time_sampling(model, source, dim, cfg, nfe_list, device="cpu"):
    """Measure sampling time at various NFE."""
    from rafm.flow_matching.sampler import Sampler
    n_gen = cfg.get("n_gen_samples", 10_000)
    results = {}
    for nfe in nfe_list:
        sample_cfg = dict(cfg)
        sample_cfg["nfe"] = nfe
        sample_cfg["device"] = device
        sampler = Sampler(model, source, sample_cfg)
        # Warmup
        sampler.sample(100, dim)
        # Actual timing
        gen = sampler.sample(n_gen, dim)
        results[f"sample_time_nfe{nfe}_s"] = gen["sample_time_s"]
    return results


def run_exp3(cfg: dict) -> None:
    out_base = Path(cfg.get("output_dir", "outputs")) / cfg["experiment"]
    out_base.mkdir(parents=True, exist_ok=True)

    device = cfg.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    nfe_list = cfg.get("nfe_list", [32, 64, 128])
    timing_steps = cfg.get("time_train_steps", 1000)
    n_timing_runs = cfg.get("n_timing_runs", 3)

    all_results = {}

    for ds_cfg in cfg["datasets"]:
        from experiments.exp0_source_diagnostics import DATASETS_FACTORY
        dataset = DATASETS_FACTORY[ds_cfg["name"]](ds_cfg)
        dim = dataset.dim
        print(f"\nDataset: {dataset.name}")
        train_data = dataset.get_train_data()

        for method_cfg in cfg["methods"]:
            method_name = method_cfg["name"]
            print(f"  Method: {method_name}")

            if method_cfg.get("adapter") == "msgm":
                print("  (MSGM timing via adapter)")
                try:
                    from baselines.msgm_adapter import MSGMAdapter
                    adapter = MSGMAdapter(train_data, cfg, device)
                    t_stats = adapter.time_training(timing_steps)
                    s_stats = adapter.time_sampling(cfg.get("n_gen_samples", 10_000), nfe_list)
                    all_results[f"{dataset.name}_{method_name}"] = {**t_stats, **s_stats}
                except Exception as e:
                    print(f"  MSGM timing failed: {e}")
                continue

            set_all_seeds(42)
            source = build_source(method_cfg["source"], train_data, dataset)
            path = build_path(method_cfg["path"])

            from rafm.models.mlp import MLP
            model = MLP(input_dim=dim, hidden_dim=cfg.get("hidden_dim", 128))

            # Average over n_timing_runs
            train_times = []
            for _ in range(n_timing_runs):
                t_stats = time_training(model, path, source, dataset, cfg, timing_steps, device)
                train_times.append(t_stats["time_per_step_ms"])

            s_stats = time_sampling(model, source, dim, cfg, nfe_list, device)
            key = f"{dataset.name}_{method_name}"
            all_results[key] = {
                "time_per_step_ms_mean": sum(train_times) / len(train_times),
                "peak_memory_mb": t_stats["peak_memory_mb"],
                **s_stats,
            }
            print(f"  {key}: {all_results[key]['time_per_step_ms_mean']:.2f} ms/step")

    (out_base / "runtime_results.json").write_text(json.dumps(all_results, indent=2))
    print(f"\nRuntime results saved to {out_base}/runtime_results.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/exp3/runtime.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    run_exp3(cfg)


if __name__ == "__main__":
    main()
