"""Tests for scripts/e4_thresholds.py (Day 7 runner). Synthetic data only."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e3 = _load("e3_calibration", "e3_calibration.py")
e4 = _load("e4_thresholds", "e4_thresholds.py")

from src.cascade import load_thresholds, validate_thresholds  # noqa: E402

SEED = 42
N_LOCAL, N_AGG = 8, 4


def _frame():
    """49 steps x 40 rows. Aggregated features carry extra signal."""
    rng = np.random.default_rng(1)
    n_steps, per_step = 49, 40
    n = n_steps * per_step
    local = rng.normal(size=(n, N_LOCAL))
    agg = rng.normal(size=(n, N_AGG))
    y = (local[:, 0] + agg[:, 0] + 0.5 * rng.normal(size=n) > 1.6).astype(int)
    df = pd.DataFrame(local, columns=[f"local_{i + 1}" for i in range(N_LOCAL)])
    for i in range(N_AGG):
        df[f"agg_{i + 1}"] = agg[:, i]
    df.insert(0, "time_step", np.repeat(np.arange(1, n_steps + 1), per_step))
    df.insert(0, "txId", np.arange(n))
    df["label"] = y
    return df


def _splits(df):
    return {
        "train_stage1": df[df.time_step.between(1, 28)].reset_index(drop=True),
        "calib": df[df.time_step.between(29, 34)].reset_index(drop=True),
    }


def _build_stage1(family, base_dir):
    df = _frame()
    splits = _splits(df)
    cols_local = [c for c in df.columns if c.startswith("local_")]
    cols_all = cols_local + [c for c in df.columns if c.startswith("agg_")]
    out_dir, art_dir = Path(base_dir) / "results", Path(base_dir) / "artifacts"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    e3.run_family(family, SEED, splits, cols_local, e3.threshold_grid({}), 10, out_dir, art_dir)
    from src.cascade import load_stage1
    return load_stage1(art_dir / f"stage1_{family}.joblib"), splits, cols_all, art_dir


@pytest.fixture(scope="module")
def lr_run(tmp_path_factory):
    artifact, splits, cols_all, art_dir = _build_stage1("lr", tmp_path_factory.mktemp("lr"))
    res = e4.compute_thresholds("lr", artifact, splits, cols_all,
                                e3.threshold_grid({}), 0.01, 0.95, SEED)
    return res, artifact, splits, cols_all, art_dir


def test_thresholds_respect_the_protocol(lr_run):
    res, artifact, *_ = lr_run
    assert res["d"] == pytest.approx(artifact["d"])      # d is loaded, never recomputed
    assert 0.0 <= res["d2"] <= 1.0
    assert 0.0 <= res["t_clear"] <= res["d"] + 1e-9
    validate_thresholds(res["t_clear"], res["d"])


def test_sweep_grid_covers_zero_to_d(lr_run):
    res, *_ = lr_run
    sweep = res["sweep"]
    assert sweep["t_clear"].iloc[0] == 0.0
    assert sweep["t_clear"].max() <= res["d"] + 1e-9


def test_escalate_everything_matches_graph_always(lr_run):
    res, *_ = lr_run
    first = res["sweep"].iloc[0]                       # t_clear = 0
    assert first["escalation_rate"] == 1.0
    assert first["recall_retention"] == pytest.approx(1.0)


def test_chosen_row_meets_target_or_is_flagged(lr_run):
    res, *_ = lr_run
    if res["target_met"]:
        assert res["chosen_row"]["recall_retention"] >= res["target"] - 1e-9
        # largest t_clear that still meets the target
        ok = res["sweep"][res["sweep"]["recall_retention"] >= res["target"] - 1e-9]
        assert res["t_clear"] == ok["t_clear"].max()
    else:
        best = res["sweep"]["recall_retention"].max()
        assert res["chosen_row"]["recall_retention"] == pytest.approx(best)


def test_rerun_is_deterministic(lr_run):
    res, artifact, splits, cols_all, _ = lr_run
    again = e4.compute_thresholds("lr", artifact, splits, cols_all,
                                  e3.threshold_grid({}), 0.01, 0.95, SEED)
    assert again["d2"] == pytest.approx(res["d2"])
    assert again["t_clear"] == pytest.approx(res["t_clear"])


def test_refuses_data_that_reaches_the_test_steps(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    df = _frame()
    bad = dict(splits)
    bad["calib"] = df[df.time_step.between(29, 35)].reset_index(drop=True)
    with pytest.raises(ValueError):
        e4.compute_thresholds("lr", artifact, bad, cols_all,
                              e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_refuses_an_artifact_built_from_other_windows(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    stale = dict(artifact)
    stale["calib_steps"] = [29, 33]
    with pytest.raises(ValueError):
        e4.compute_thresholds("lr", stale, splits, cols_all,
                              e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_refuses_the_wrong_family(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    with pytest.raises(ValueError):
        e4.compute_thresholds("rf", artifact, splits, cols_all,
                              e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_run_family_writes_frozen_files_and_refuses_to_overwrite(lr_run, tmp_path):
    _, _, splits, cols_all, art_dir = lr_run
    args = SimpleNamespace(artifacts=str(art_dir), thresholds_dir=str(tmp_path),
                           t_step=0.01, target=0.95, force=False)
    cfg = {"seed": SEED, "splits": {"train_stage1": [1, 28], "calib": [29, 34]}}
    sweep = e4.run_family("lr", cfg, splits, cols_all, e3.threshold_grid({}), args)
    assert (sweep["family"] == "lr").all()

    saved = load_thresholds(tmp_path / "thresholds_lr.json")
    assert saved["family"] == "lr"
    assert saved["chosen_on_steps"] == [29, 34]
    assert saved["t_clear"] <= saved["d"] + 1e-9
    assert json.loads((tmp_path / "thresholds_lr.json").read_text())["stage2_validation_trained_on_steps"] == [1, 28]

    with pytest.raises(SystemExit):                  # already frozen
        e4.run_family("lr", cfg, splits, cols_all, e3.threshold_grid({}), args)
    args.force = True
    e4.run_family("lr", cfg, splits, cols_all, e3.threshold_grid({}), args)  # explicit override works


@pytest.mark.slow
def test_random_forest_end_to_end(tmp_path):
    artifact, splits, cols_all, _ = _build_stage1("rf", tmp_path)
    res = e4.compute_thresholds("rf", artifact, splits, cols_all,
                                e3.threshold_grid({}), 0.01, 0.95, SEED)
    assert 0.0 <= res["t_clear"] <= res["d"] + 1e-9
    assert res["sweep"].iloc[0]["recall_retention"] == pytest.approx(1.0)


# ---- configuration and window validation ---------------------------------------------

def test_threshold_grid_defaults_and_valid_config():
    g = e4.threshold_grid({})
    assert g[0] == pytest.approx(0.01) and g[-1] == pytest.approx(0.99) and len(g) == 99
    g = e4.threshold_grid({"threshold_grid": {"start": 0.1, "stop": 0.5, "step": 0.1}})
    assert list(g) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])


@pytest.mark.parametrize("bad", [
    {"step": 0}, {"step": -0.01}, {"step": float("nan")}, {"step": float("inf")},
    {"start": float("nan")}, {"stop": float("inf")},
    {"start": -0.1}, {"stop": 1.5}, {"start": 0.8, "stop": 0.2},
    {"step": "abc"}, {"start": None},
])
def test_invalid_threshold_grid_config_is_rejected(bad):
    with pytest.raises(ValueError):
        e4.threshold_grid({"threshold_grid": bad})


@pytest.mark.parametrize("target, t_step", [
    (0.0, 0.01), (-0.5, 0.01), (1.5, 0.01), (float("nan"), 0.01), (float("inf"), 0.01),
    (0.95, 0.0), (0.95, -0.01), (0.95, float("nan")), (0.95, float("inf")), (0.95, 2.0),
])
def test_invalid_target_or_t_step_is_rejected(target, t_step):
    with pytest.raises(ValueError):
        e4.validate_cli_values(target, t_step)


def test_valid_target_and_t_step_are_accepted():
    e4.validate_cli_values(0.95, 0.01)
    e4.validate_cli_values(1.0, 0.5)


def test_compute_thresholds_rejects_a_bad_target_before_training(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    with pytest.raises(ValueError):
        e4.compute_thresholds("lr", artifact, splits, cols_all,
                              e3.threshold_grid({}), 0.01, 1.5, SEED)
    with pytest.raises(ValueError):
        e4.compute_thresholds("lr", artifact, splits, cols_all,
                              e3.threshold_grid({}), float("nan"), 0.95, SEED)


def test_windows_from_cfg():
    cfg = {"splits": {"train_stage1": [1, 28], "calib": [29, 34]}}
    assert e4.windows_from_cfg(cfg) == ((1, 28), (29, 34))
    for bad in ({}, {"splits": {}}, {"splits": {"train_stage1": [1], "calib": [29, 34]}},
                {"splits": {"train_stage1": [28, 1], "calib": [29, 34]}},
                {"splits": {"train_stage1": "x", "calib": [29, 34]}}):
        with pytest.raises(ValueError):
            e4.windows_from_cfg(bad)


def test_missing_time_step_in_training_data_is_rejected(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    bad = dict(splits)
    train = splits["train_stage1"]
    bad["train_stage1"] = train[train.time_step != 10].reset_index(drop=True)   # gap at step 10
    with pytest.raises(ValueError, match="train_stage1"):
        e4.compute_thresholds("lr", artifact, bad, cols_all, e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_missing_time_step_in_calibration_data_is_rejected(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    bad = dict(splits)
    calib = splits["calib"]
    bad["calib"] = calib[calib.time_step != 31].reset_index(drop=True)
    with pytest.raises(ValueError, match="calib"):
        e4.compute_thresholds("lr", artifact, bad, cols_all, e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_unexpected_extra_step_in_training_data_is_rejected(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    df = _frame()
    bad = dict(splits)
    bad["train_stage1"] = df[df.time_step.between(1, 29)].reset_index(drop=True)   # step 29 leaks in
    with pytest.raises(ValueError, match="unexpected"):
        e4.compute_thresholds("lr", artifact, bad, cols_all, e3.threshold_grid({}), 0.01, 0.95, SEED)


def test_windows_that_overlap_or_reach_the_test_steps_are_rejected(lr_run):
    _, artifact, splits, cols_all, _ = lr_run
    for windows in (((1, 28), (28, 34)), ((1, 28), (29, 35))):
        with pytest.raises(ValueError):
            e4.compute_thresholds("lr", artifact, splits, cols_all,
                                  e3.threshold_grid({}), 0.01, 0.95, SEED, windows=windows)