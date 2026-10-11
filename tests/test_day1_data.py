"""Day 1 checks for the Elliptic data loader.

Two groups of tests:
  1. TOY tests: tiny synthetic files, always runnable, prove the loader's
     logic (label mapping, error handling, cross-step edge detection).
  2. REAL-DATA tests: verify SPEC.md Section 14 against data/raw/.
     They are skipped automatically if the dataset is not found.

Run from the repo root:
    pytest tests/test_day1_data.py -v -s

Lines starting with "[DAY1]" are the facts to paste back.
Set ELLIPTIC_DIR to point at a different data folder if needed.
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import load_data as ld  # noqa: E402

DATA_DIR = Path(os.environ.get("ELLIPTIC_DIR", ROOT / "data" / "raw"))
RESULTS_DIR = ROOT / "results"


def say(message):
    print(f"[DAY1] {message}")


# --------------------------------------------------------------------------
# Toy-data helpers
# --------------------------------------------------------------------------

TOY_IDS = [10, 11, 12, 13]
TOY_STEPS = [1, 1, 2, 2]


def _write_toy(
    tmp_path,
    n_cols=167,
    classes=("unknown", "1", "2", "unknown"),
    edges=((10, 11), (12, 13)),
):
    rng = np.random.default_rng(0)
    feats = pd.DataFrame(rng.normal(size=(4, n_cols)))
    feats[0] = np.array(TOY_IDS)
    feats[1] = np.array(TOY_STEPS)
    feats.to_csv(tmp_path / ld.FEATURES_FILE, header=False, index=False)

    pd.DataFrame({"txId": TOY_IDS, "class": list(classes)}).to_csv(
        tmp_path / ld.CLASSES_FILE, index=False
    )
    pd.DataFrame(list(edges), columns=["txId1", "txId2"]).to_csv(
        tmp_path / ld.EDGES_FILE, index=False
    )
    return tmp_path


# --------------------------------------------------------------------------
# 1. Toy tests (always run)
# --------------------------------------------------------------------------

def test_toy_label_mapping(tmp_path):
    _write_toy(tmp_path)
    ds = ld.build_dataset(tmp_path)
    labels = ds.set_index("txId")["label"].to_dict()
    assert labels == {10: -1, 11: 1, 12: 0, 13: -1}


def test_toy_column_names_and_sizes(tmp_path):
    _write_toy(tmp_path)
    ds = ld.build_dataset(tmp_path)
    assert list(ds.columns[:2]) == ["txId", "time_step"]
    assert ds.columns[-1] == "label"
    assert len(ld.local_columns(ds)) == 93
    assert len(ld.agg_columns(ds)) == 72


def test_toy_feature_sets_never_contain_forbidden_columns(tmp_path):
    _write_toy(tmp_path)
    ds = ld.build_dataset(tmp_path)
    local = set(ld.local_columns(ds))
    agg = set(ld.agg_columns(ds))
    everything = set(ld.all_feature_columns(ds))
    assert local.isdisjoint(agg)
    assert everything == local | agg
    for forbidden in ld.NON_FEATURE_COLUMNS:
        assert forbidden not in everything


def test_toy_wrong_column_count_raises(tmp_path):
    _write_toy(tmp_path, n_cols=100)
    with pytest.raises(ValueError):
        ld.load_features(tmp_path)


def test_toy_unexpected_class_value_raises(tmp_path):
    _write_toy(tmp_path, classes=("unknown", "1", "3", "unknown"))
    with pytest.raises(ValueError):
        ld.load_classes(tmp_path)


def test_toy_cross_step_edge_is_detected(tmp_path):
    _write_toy(tmp_path, edges=((10, 11), (11, 12)))  # second edge: step 1 -> 2
    ds = ld.build_dataset(tmp_path)
    ed = ld.load_edges(tmp_path)
    report = ld.edge_step_report(ds, ed)
    assert report["n_edges"] == 2
    assert report["n_cross_step"] == 1
    assert report["n_missing_endpoint"] == 0

def test_toy_within_step_edges_report_zero_cross(tmp_path):
    _write_toy(tmp_path)
    ds = ld.build_dataset(tmp_path)
    ed = ld.load_edges(tmp_path)
    assert ld.edge_step_report(ds, ed)["n_cross_step"] == 0


# --------------------------------------------------------------------------
# 2. Real-data tests (skipped if data is absent)
# --------------------------------------------------------------------------

@pytest.fixture(scope="module")
def real():
    if not (DATA_DIR / ld.FEATURES_FILE).exists():
        pytest.skip(f"Dataset not found in {DATA_DIR}")
    ds = ld.build_dataset(DATA_DIR)
    ed = ld.load_edges(DATA_DIR)
    return ds, ed

@pytest.mark.slow
def test_real_shapes_and_counts(real):
    ds, ed = real
    summary = ld.dataset_summary(ds, ed)
    say(f"features file loaded: {ds.shape[0]} rows, "
        f"{ds.shape[1]} columns (incl. txId, time_step, label)")
    for key, value in summary.items():
        say(f"{key} = {value}")

    assert summary["n_transactions"] == ld.EXPECTED_N_TX
    assert summary["n_edges"] == ld.EXPECTED_N_EDGES
    assert summary["n_local_features"] == ld.N_LOCAL
    assert summary["n_agg_features"] == ld.N_AGG
    assert (summary["time_step_min"], summary["time_step_max"]) == (1, 49)

@pytest.mark.slow
def test_real_class_counts_and_direction(real):
    ds, _ = real
    n_ill = int((ds["label"] == 1).sum())
    n_lic = int((ds["label"] == 0).sum())
    n_unk = int((ds["label"] == -1).sum())
    say(f"label counts: illicit={n_ill}, licit={n_lic}, unlabelled={n_unk}")
    say(f"illicit share of labelled = {n_ill / (n_ill + n_lic):.4f}")

    assert n_ill == 4_545
    assert n_lic == 42_019
    # Illicit must be the minority class; if not, the 1/2 mapping is swapped.
    assert n_ill < n_lic

@pytest.mark.slow
def test_real_ids_unique_and_no_nans(real):
    ds, _ = real
    assert ds["txId"].is_unique
    feats = ld.all_feature_columns(ds)
    assert not ds[feats].isna().any().any()
    say("txId unique and no missing feature values: OK")

@pytest.mark.slow
def test_real_edges_stay_within_one_time_step(real):
    ds, ed = real
    report = ld.edge_step_report(ds, ed)
    say(f"edge report: {report}")
    assert report["n_missing_endpoint"] == 0
    assert report["n_cross_step"] == 0, (
        "Some edges cross time steps; SPEC Section 3 assumption is false."
    )

@pytest.mark.slow
def test_real_illicit_share_per_time_step(real):
    ds, _ = real
    g = ds.groupby("time_step")["label"].agg(
        n_total="size",
        n_illicit=lambda s: int((s == 1).sum()),
        n_licit=lambda s: int((s == 0).sum()),
        n_unlabelled=lambda s: int((s == -1).sum()),
    )
    labelled = (g["n_illicit"] + g["n_licit"]).replace(0, np.nan)
    g["illicit_share_of_labelled"] = (g["n_illicit"] / labelled).round(4)

    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    g.to_csv(RESULTS_DIR / "day1_illicit_by_step.csv")

    say("per-time-step table (time_step, n_total, n_illicit, n_licit, "
        "n_unlabelled, illicit_share_of_labelled):")
    for step, row in g.iterrows():
        say(f"  {step:>2} {int(row.n_total):>6} {int(row.n_illicit):>5} "
            f"{int(row.n_licit):>6} {int(row.n_unlabelled):>6} "
            f"{row.illicit_share_of_labelled}")

    zero_illicit = [int(s) for s in g.index[g["n_illicit"] == 0]]
    say(f"time steps with zero labelled illicit: {zero_illicit}")

    windows = {"train 1-28": (1, 28), "calib 29-34": (29, 34),
               "test 35-49": (35, 49)}
    for name, (lo, hi) in windows.items():
        sub = g.loc[lo:hi]
        say(f"{name}: labelled={int(sub.n_illicit.sum() + sub.n_licit.sum())}, "
            f"illicit={int(sub.n_illicit.sum())}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(9, 3.5))
        ax.bar(g.index, g["illicit_share_of_labelled"])
        ax.axvline(28.5, linestyle="--", color="grey")
        ax.axvline(34.5, linestyle="--", color="grey")
        ax.set_xlabel("time step")
        ax.set_ylabel("illicit share of labelled")
        ax.set_title("Illicit share per time step (dashed: split boundaries)")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "figures" / "day1_illicit_share.png", dpi=150)
        plt.close(fig)
        say("saved results/figures/day1_illicit_share.png")
    except ImportError:
        say("matplotlib not installed; plot skipped")

    assert len(g) == 49

@pytest.mark.slow
def test_real_split_point_diagnostic(real):
    """Diagnostic only (no assert): does the local/agg boundary show up?

    Assumption (unverified): transactions with no edges have empty
    neighbourhoods, so their aggregated features should look degenerate
    (e.g. mostly exactly 0) while local features do not. If the jump
    happens between local_93 and agg_1, the 93/72 split is supported.
    """
    ds, ed = real
    touched = pd.concat([ed["txId1"], ed["txId2"]]).unique()
    isolated = ~ds["txId"].isin(touched)
    say(f"transactions with no edges: {int(isolated.sum())}")

    if isolated.sum() == 0:
        say("no isolated transactions; diagnostic not possible")
        return

    local = ld.local_columns(ds)
    agg = ld.agg_columns(ds)
    zero_share = (ds.loc[isolated, local + agg] == 0).mean()
    say(f"isolated nodes: mean share of exact zeros, local block = "
        f"{zero_share[local].mean():.3f}, agg block = {zero_share[agg].mean():.3f}")
    for col in ["local_91", "local_92", "local_93", "agg_1", "agg_2", "agg_3"]:
        say(f"  zero share at {col} = {zero_share[col]:.3f}")