# Radial-Angular Flow Matching (RAFM)

Official implementation for:

> **Correcting Source Mismatch in Flow Matching with Radial-Angular Transport**

---

## Setup

```bash
conda create -n rafm python=3.11
conda activate rafm
pip install -e .
```

Requires PyTorch ≥ 2.0 with CUDA. MSGM baseline additionally requires the reference code placed at `18727_Multiplicative_Diffusion_code/` (see [Baselines](#baselines)).

---

## Overview

Standard Flow Matching uses a Gaussian source N(0, I), which mismatches the radial law of heavy-tailed or real-world data. RAFM corrects this by:

1. **Radial coupling** — source points are sampled with the same norm as their target: x₀ = ‖x₁‖ · u₀, u₀ ~ Uniform(S^{d−1})
2. **Spherical geodesic path** — transport follows a slerp interpolation, preserving ‖x_t‖ = ‖x₁‖ throughout
3. **Tangent projection** — at sampling time, the learned velocity is projected onto the tangent space of the sphere, decoupling radial and angular generation

The radial distribution is controlled entirely by the source (fitted via eCDF); the network only learns angular rotation.

---

## Repository structure

```
rafm/                    Core library
  models/                MLP architecture (shared with MSGM for fair comparison)
  sources/               Gaussian, RadialOracle, RadialEmpirical sources
  paths/                 Euclidean and SphericalGeodesic conditional paths
  flow_matching/         CFM loss, trainer, ODE sampler
  data/                  Datasets: ToyRadialAngular, StudentT, GaussianAniso, PIV
  metrics/               Radial, distributional, angular, stability metrics
  utils/                 Seeds, sphere geometry, I/O

configs/                 YAML configs (inherit from configs/defaults.yaml)
experiments/             Experiment runners (exp0–exp4)
baselines/               MSGM adapter (wraps reference code without modification)
scripts/                 Aggregation, ablations, table/figure generation
tests/                   Unit tests (pytest)
```

---

## Experiments

### Run all

```bash
python scripts/run_all.py --experiments 0,1,2,3
```

### Exp 0 — Source diagnostics (no training)

Compares radial CDFs of Gaussian, oracle, and empirical sources against test data.

```bash
python -m experiments.exp0_source_diagnostics
```

### Exp 1 — Main generation benchmark

```bash
# Synthetic
python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d16.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d32.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/gaussian_d16.yaml

# Real (PIV) — requires data preparation below
python -m experiments.exp1_main_benchmark --config configs/exp1/piv_d64.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/piv_d256.yaml
```

### Exp 2 — Sample efficiency

```bash
python -m experiments.exp2_sample_efficiency
```

### Exp 3 — Runtime comparison

```bash
python -m experiments.exp3_runtime
```

### Exp 4 — Solver sensitivity

```bash
python -m experiments.exp4_solver_sensitivity
```

### Ablation — tangent projection

Re-samples trained RAFM models without tangent projection to isolate its contribution:

```bash
python scripts/ablation_no_projection.py
```

---

## PIV dataset

Download the public dataset manually:

> "Non-time-resolved PIV dataset of flow over a circular cylinder at Re=3900"
> DOI: [10.57745/DHJXM6](https://doi.org/10.57745/DHJXM6)

Preprocess at multiple resolutions:

```bash
python -m rafm.data.prepare_piv --zip dataverse_files.zip --out_dir data/piv --grids 8x8,16x16
```

This generates `data/piv/piv_d64.pt` (8×8 grid) and `data/piv/piv_d256.pt` (16×16 grid).

---

## Methods

A method is a **(source, path)** pair within a shared CFM trainer.

| Method | Source | Path |
|---|---|---|
| Gaussian FM | N(0, I) | Euclidean |
| Source-only (empirical) | Radial eCDF | Euclidean |
| Source-only (oracle) | Analytic radial CDF† | Euclidean |
| **RAFM (empirical)** | Radial eCDF | Spherical geodesic |
| **RAFM (oracle)** | Analytic radial CDF† | Spherical geodesic |
| MSGM | — | SDE (score matching baseline) |

† Oracle source requires known analytic CDF — synthetic datasets only.

All FM methods share the same architecture: **3-layer MLP, 128 hidden units, Swish activations**.

---

## Baselines

MSGM uses the reference implementation from the original authors. Place it at:

```
18727_Multiplicative_Diffusion_code/
  SDEs.py
  NN.py
  sde_scheme.py
```

The adapter at `baselines/msgm_adapter.py` wraps this code without modifying it.

---

## Aggregate results

```bash
python scripts/aggregate_results.py --out outputs/results_summary.csv
```

---

## Tests

```bash
pytest tests/ -v
```

Key tests: norm preservation under slerp, tangent velocity field, antipodal handling (`tests/test_paths.py`).
