# Radial-Angular Flow Matching (RAFM)

Code for the experiments of the paper:

> **Correcting Source Mismatch in Flow Matching with Radial-Angular Transport**
> NeurIPS 2025

---

## Setup

```bash
conda create -n rafm python=3.11
conda activate rafm
pip install -e .
```

---

## Repository structure

```
rafm/                   Core library
  models/               MLP (same architecture as MSGM baseline for fairness)
  sources/              Gaussian, RadialOracle, RadialEmpirical source distributions
  paths/                Euclidean and SphericalGeodesic conditional paths
  flow_matching/        CFM loss, training loop, ODE sampler
  data/                 Datasets (ToyRadialAngular, StudentT, GaussianAniso, PIV)
  metrics/              Radial, distributional, angular, stability, efficiency metrics
  plotting/             Publication-ready figures and LaTeX tables
  utils/                Seeds, sphere utilities, logging, I/O

configs/                YAML experiment configs (inherit from configs/defaults.yaml)
experiments/            Thin runner scripts (exp0–exp4)
baselines/              MSGM adapter (wraps reference code without modifying it)
scripts/                Aggregation, table and figure generation
tests/                  Unit tests (pytest)
```

---

## Experiments

### Exp 0 — Source diagnostics (no training)

Compares radial CDFs of Gaussian, oracle, and empirical sources against test data.

```bash
python -m experiments.exp0_source_diagnostics
```

### Exp 1 — Main generation benchmark

```bash
python -m experiments.exp1_main_benchmark --config configs/exp1/studentt_d16.yaml
python -m experiments.exp1_main_benchmark --config configs/exp1/toy2d.yaml
```

### Exp 2 — Sample efficiency

```bash
python -m experiments.exp2_sample_efficiency
```

### Exp 3 — Runtime comparison

```bash
python -m experiments.exp3_runtime
```

### Exp 4 — Solver sensitivity (optional)

```bash
python -m experiments.exp4_solver_sensitivity
```

### Run all

```bash
python scripts/run_all.py --experiments 0,1,2,3
```

---

## PIV dataset

The PIV benchmark requires downloading the public dataset manually:

> "Non-time-resolved PIV dataset of flow over a circular cylinder at Reynolds number 3900"
> DOI: [10.57745/DHJXM6](https://doi.org/10.57745/DHJXM6)

Once downloaded, preprocess:

```bash
python -m rafm.data.prepare_piv --raw_dir /path/to/raw/piv --out_dir data/piv
```

Then set `data_root: data/piv` in `configs/exp1/piv_d16.yaml`.

---

## Methods

A method is a **(source, path)** pair plugged into a generic CFM trainer.

| Method | Source | Path |
|---|---|---|
| Gaussian FM | Gaussian N(0, I) | Euclidean |
| Source-only (oracle) | Radial oracle* | Euclidean |
| Source-only (empirical) | Radial eCDF | Euclidean |
| RAFM (oracle) | Radial oracle* | Spherical geodesic |
| RAFM (empirical) | Radial eCDF | Spherical geodesic |
| MSGM | — | — (score matching baseline) |

*Oracle radial source is available for **synthetic datasets only** (analytic CDF known).
For real data (PIV), only the empirical source is used.

All FM methods use the same architecture: **3-layer MLP, hidden=128, Swish, no preprocessing**.
`NormalizeLogRadius` is available as an explicit ablation only (`premodule: NormalizeLogRadius` in config).

---

## Metrics

**Primary (theory-aligned, no KL for raw eCDF):**
- Radial W1, KS statistic
- Extreme quantile errors at 95%, 99%, 99.5%
- Tail exceedance calibration

**Secondary:**
- Sliced Wasserstein, MMD
- Angular Sliced Wasserstein (within norm quantile bins)
- Stability: NaN rate, exploding norm rate

**Efficiency:**
- Wall-clock (train + sample), peak GPU memory, NFE

---

## Aggregate results and generate tables

```bash
python scripts/aggregate_results.py --experiment exp1_main_benchmark
python scripts/generate_tables.py --experiment exp1_main_benchmark
python scripts/generate_figures.py --all
```

---

## Tests

```bash
conda install pytest
pytest tests/ -v
```

Critical tests: norm preservation, tangent velocity, antipodal handling (`tests/test_paths.py`).

---

## Success criteria

The experiments validate these 6 claims:

1. On Student-t: source-only FM improves radial metrics vs Gaussian FM
2. On Student-t: full RAFM improves further on global/angular metrics
3. On Gaussian control: RAFM stays within seed variability of Gaussian FM
4. Empirical RAFM converges to oracle RAFM as `n_train` increases
5. vs MSGM: better quality/compute trade-off on comparable settings
6. Stability: NaN/divergence rate < 1% for all FM methods

If any criterion fails, it is documented explicitly — negative results are not hidden.
