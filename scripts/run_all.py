"""Orchestrate all experiments sequentially or in parallel.

Usage:
    python scripts/run_all.py --experiments 0,1,2,3
    python scripts/run_all.py --experiments 0           # diagnostics only
    python scripts/run_all.py --experiments 1 --dataset studentt_d16
"""
import argparse
import subprocess
import sys
from pathlib import Path

EXPERIMENT_CONFIGS = {
    "0": ["configs/exp0/source_diagnostics.yaml"],
    "1": [
        "configs/exp1/toy2d.yaml",
        "configs/exp1/studentt_d16.yaml",
        "configs/exp1/studentt_d32.yaml",
        "configs/exp1/gaussian_d16.yaml",
        "configs/exp1/gaussian_d32.yaml",
        # PIV configs require prepare_piv.py to be run first:
        # "configs/exp1/piv_d16.yaml",
        # "configs/exp1/piv_d32.yaml",
    ],
    "2": ["configs/exp2/sample_efficiency.yaml"],
    "3": ["configs/exp3/runtime.yaml"],
    "4": ["configs/exp4/solver_sensitivity.yaml"],
}

EXPERIMENT_SCRIPTS = {
    "0": "experiments.exp0_source_diagnostics",
    "1": "experiments.exp1_main_benchmark",
    "2": "experiments.exp2_sample_efficiency",
    "3": "experiments.exp3_runtime",
    "4": "experiments.exp4_solver_sensitivity",
}


def run_experiment(exp_id: str, config: str) -> int:
    script = EXPERIMENT_SCRIPTS[exp_id]
    cmd = [sys.executable, "-m", script, "--config", config]
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd)
    return result.returncode


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiments", default="0,1,2,3",
                        help="Comma-separated list of experiment IDs")
    parser.add_argument("--dataset", default=None,
                        help="Only run configs matching this dataset name")
    args = parser.parse_args()

    exp_ids = [e.strip() for e in args.experiments.split(",")]
    total_failed = 0

    for exp_id in exp_ids:
        if exp_id not in EXPERIMENT_CONFIGS:
            print(f"Unknown experiment: {exp_id}")
            continue

        configs = EXPERIMENT_CONFIGS[exp_id]
        if args.dataset:
            configs = [c for c in configs if args.dataset in c]

        for config in configs:
            if not Path(config).exists():
                print(f"Config not found: {config} — skipping")
                continue
            rc = run_experiment(exp_id, config)
            if rc != 0:
                print(f"FAILED: {config} (exit code {rc})")
                total_failed += 1

    if total_failed > 0:
        print(f"\n{total_failed} experiment(s) failed.")
        sys.exit(1)
    else:
        print("\nAll experiments completed successfully.")


if __name__ == "__main__":
    main()
