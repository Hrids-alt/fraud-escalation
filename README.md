# FraudFin

FraudFin is a cost-aware, two-stage system for detecting illicit Bitcoin
transactions using the Elliptic dataset.

It compares local-only and graph-feature models and investigates whether
selectively escalating uncertain transactions can reduce simulated
computational cost while retaining illicit-transaction recall.

## Main result (test steps 35-49, evaluated once)

| System | Precision | Recall | F1 |
|---|---|---|---|
| Local-only | 0.701 | 0.732 | 0.716 |
| Graph-always | 0.793 | 0.734 | 0.763 |
| **Cascade** | **0.896** | 0.721 | **0.799** |
| Random escalation (same rate) | 0.704 | 0.733 | 0.718 |

Random Forest; Stage 2 values are the mean over 3 seeds. The cascade sends 6.8% of transactions to Stage 2 and keeps about 98% of graph-always recall. The simulated cost saving against graph-always is 78% (r = 5), 89% (r = 20) and 92% (r = 100). Cost is simulated, because the dataset's graph features are precomputed. After time step 42 every system collapses to near-zero F1 (temporal drift). See `docs/notes.md` for limitations.

## Dataset and Attribution

This project uses the Elliptic Bitcoin transaction dataset for academic experimentation in financial fraud detection.

**Original research:** Weber et al., *Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics* (2019).

- **Paper:** https://arxiv.org/abs/1908.02591
- **Dataset source:** https://www.kaggle.com/datasets/ellipticco/elliptic-data-set
- **Dataset licence:** Creative Commons Attribution-NonCommercial-NoDerivatives 4.0 International (CC BY-NC-ND 4.0)  
  https://creativecommons.org/licenses/by-nc-nd/4.0/

The raw dataset files are not included in this repository. Users must obtain the dataset from its original source and comply with the applicable licence terms.

This project is an independent academic implementation and is not endorsed by the dataset creators.

**Modifications:** The dataset is loaded, labelled according to this project's documented convention, and analysed for model development and evaluation. No modified copy of the dataset is redistributed.

## Reproduce

Run everything from the repository root. The raw Elliptic files are not included; download them from the dataset source listed under "Dataset and Attribution" above and place the three CSV files (`elliptic_txs_features.csv`, `elliptic_txs_classes.csv`, `elliptic_txs_edgelist.csv`) in `data/raw/`.

```bash
python -m venv .venv

# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate

pip install -r requirements.txt

python -m pytest -q -m "not slow"      # fast checks (the full suite is: python -m pytest -q)
```

**Earlier stages** (baselines and drift analysis, Days 1-5 of the project):

```bash
python run_baselines.py --features local
python run_baselines.py --features all
python scripts/e2_drift.py
```

**Main pipeline.** Run these in this order. Each step depends on the one before it.

```bash
# 1. Stage 1: train on steps 1-28, calibrate on steps 29-34, choose d
python scripts/e3_calibration.py

# 2. Freeze d2 and t_clear on steps 29-34 (writes thresholds_rf.json)
python scripts/e4_thresholds.py --family rf

# 3. Evaluate the cascade once on test steps 35-49
python scripts/e5_cascade_test.py --family rf

# 4. Seed spread of the final Stage 2 (thresholds unchanged)
python scripts/e6_seed_spread.py --family rf

# 5. Export the results for the web app
python scripts/export_for_ui.py
```

| Step | Reads | Writes |
|---|---|---|
| `e3_calibration.py` | `data/raw/` | `artifacts/stage1_*.joblib` (not committed), `results/stage1_*.json`, `results/e3_*.csv`, reliability figures |
| `e4_thresholds.py` | Stage 1 artifact, steps 1-34 | `thresholds_rf.json`, `results/e4_validation_sweep.csv` |
| `e5_cascade_test.py` | frozen Stage 1, `thresholds_rf.json`, steps 1-34 and 35-49 | `results/cascade_predictions.csv`, `results/e5_*.csv`, `results/e5_run_meta.json`, figures |
| `e6_seed_spread.py` | same frozen inputs | `results/e6_seed_spread.csv` |
| `export_for_ui.py` | `results/` and `thresholds_rf.json` only | `web/data/results.json` |

**Frozen on purpose.** `e4`, `e5` and `e6` refuse to overwrite their outputs unless you pass `--force`. This protects the protocol: thresholds are chosen on steps 29-34 only, the test steps 35-49 are evaluated once, and no threshold is changed after seeing test results. Use `--force` only to redo a step before any test evaluation, or for a documented re-run. `e3_calibration.py` rebuilds the Stage 1 artifacts, so re-running it means redoing everything after it.

**Reproducing on a fresh clone.** The repository already contains the frozen thresholds and results, so `e4`, `e5` and `e6` will stop with "already exists". To verify the pipeline from scratch, work in a separate copy of the repository and delete the generated files first. On Windows use `del`; on Linux/macOS use `rm`.

```bash
python scripts/e3_calibration.py
del thresholds_rf.json results\e4_validation_sweep.csv
del results\cascade_predictions.csv results\e6_seed_spread.csv
python scripts/e4_thresholds.py --family rf
python scripts/e5_cascade_test.py --family rf
python scripts/e6_seed_spread.py --family rf
python scripts/export_for_ui.py
```

The values should match the committed ones. Only `artifacts/` is missing from a fresh clone, so `e3_calibration.py` must be run once before anything else can use the frozen Stage 1.

**Cost is simulated.** The graph features are precomputed in this dataset, so the cost saving is modelled (local = 1, graph = r times local), not measured.