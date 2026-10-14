"""Tests for scripts/e5_cascade_test.py (Day 8 runner). Synthetic data only."""
import importlib.util
import json
import sys
from pathlib import Path

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
e5 = _load("e5_cascade_test", "e5_cascade_test.py")

from src.cascade import load_stage1, load_thresholds, save_thresholds  # noqa: E402

SEED = 42
N_LOCAL, N_AGG = 8, 4
CFG = {"seed": SEED, "splits": {"train_stage1": [1, 28], "calib": [29, 34],
                                "train_final": [1, 34], "test": [35, 49]}}

# --- toy example worked out by hand -----------------------------------------
# d = 0.5, d2 = 0.5, t_clear = 0.3
# row: label p1   p2   -> escalated?  cascade  local_only  graph_always
#  0     1    0.9  0.8     yes          1         1           1
#  1     1    0.2  0.9     no           0         0           1   (missed by the cascade)
#  2     0    0.4  0.6     yes          1         0           1   (false positive)
#  3     0    0.1  0.9     no           0         0           1
#  4     1    0.6  0.4     yes          0         1           0
#  5     0    0.3  0.2     yes          0         0           0   (p1 == t_clear escalates)
TOY = pd.DataFrame({
    "txId": [10, 11, 12, 13, 14, 15],
    "time_step": [35, 35, 36, 36, 37, 37],
    "label": [1, 1, 0, 0, 1, 0],
})
TOY_P1 = np.array([0.9, 0.2, 0.4, 0.1, 0.6, 0.3])
TOY_P2 = np.array([0.8, 0.9, 0.6, 0.9, 0.4, 0.2])
TOY_TH = {"d": 0.5, "d2": 0.5, "t_clear": 0.3}


def test_toy_cascade_decisions_by_hand():
    preds = e5.predictions_frame(TOY, TOY_P1, TOY_P2, TOY_TH)
    assert preds["escalated"].tolist() == [1, 0, 1, 0, 1, 1]
    assert preds["final_decision"].tolist() == [1, 0, 1, 0, 0, 0]
    assert preds["local_only_decision"].tolist() == [1, 0, 0, 0, 1, 0]
    assert preds["graph_always_decision"].tolist() == [1, 1, 1, 1, 0, 0]


def test_p2_is_hidden_for_cleared_rows():
    preds = e5.predictions_frame(TOY, TOY_P1, TOY_P2, TOY_TH)
    assert preds.loc[preds["escalated"] == 0, "p2"].isna().all()
    assert preds.loc[preds["escalated"] == 1, "p2"].notna().all()


def test_predictions_have_the_required_columns():
    preds = e5.predictions_frame(TOY, TOY_P1, TOY_P2, TOY_TH)
    for col in ("txId", "time_step", "p1", "escalated", "p2", "final_decision", "label"):
        assert col in preds.columns


def test_toy_summary_numbers_by_hand():
    summary, per_step, _ = e5.evaluate_systems(TOY, TOY_P1, TOY_P2, TOY_TH, seeds=(0, 1))
    row = summary.set_index("system")
    # cascade: tp 1 (row 0), fp 1 (row 2), fn 2 (rows 1, 4)
    assert row.loc["cascade", "tp"] == 1 and row.loc["cascade", "fp"] == 1
    assert row.loc["cascade", "fn"] == 2
    assert row.loc["cascade", "escalation_rate"] == pytest.approx(4 / 6)
    # graph-always: tp 2 (rows 0, 1), fp 2 (rows 2, 3), fn 1 (row 4) -> recall 2/3
    assert row.loc["graph_always", "recall"] == pytest.approx(2 / 3)
    # cascade recall 1/3 -> retention (1/3) / (2/3) = 0.5
    assert row.loc["cascade", "recall_retention"] == pytest.approx(0.5)
    assert set(per_step["time_step"]) == {35, 36, 37}


def test_random_control_matches_the_cascade_escalation_rate():
    summary, _, _ = e5.evaluate_systems(TOY, TOY_P1, TOY_P2, TOY_TH, seeds=(0, 1, 2))
    rand = summary[summary["system"] == "random_control"]
    assert len(rand) == 3
    assert rand["escalation_rate"].tolist() == pytest.approx([4 / 6] * 3)
    assert {"random_control_mean", "random_control_std"} <= set(summary["system"])


def test_pr_auc_is_nan_for_mixed_decisions():
    summary, _, _ = e5.evaluate_systems(TOY, TOY_P1, TOY_P2, TOY_TH, seeds=(0,))
    row = summary.set_index("system")
    assert np.isnan(row.loc["cascade", "pr_auc"])
    assert not np.isnan(row.loc["local_only", "pr_auc"])


def test_curve_marks_frozen_t_clear_and_stays_below_d():
    curve = e5.cost_recall_curve(TOY, TOY_P1, TOY_P2, TOY_TH, seeds=(0, 1), step=0.1)
    assert curve["t_clear"].max() <= TOY_TH["d"] + 1e-9
    assert curve["is_frozen_t_clear"].sum() == 1
    assert curve["escalation_rate"].is_monotonic_decreasing


# --- guards before the test steps are touched ---------------------------------

def _artifact(**kw):
    base = {"family": "rf", "d": 0.39, "train_steps": [1, 28], "calib_steps": [29, 34]}
    base.update(kw)
    return base


def _th(**kw):
    base = {"family": "rf", "d": 0.39, "d2": 0.26, "t_clear": 0.39}
    base.update(kw)
    return base


def test_guard_accepts_matching_inputs():
    e5.check_frozen_inputs("rf", _artifact(), _th())
    e5.check_artifact_windows(_artifact(), CFG)


def test_guard_rejects_d_mismatch():
    with pytest.raises(ValueError, match="d differs"):
        e5.check_frozen_inputs("rf", _artifact(d=0.40), _th())


def test_guard_rejects_wrong_family_and_bad_t_clear():
    with pytest.raises(ValueError):
        e5.check_frozen_inputs("lr", _artifact(), _th())
    with pytest.raises(ValueError):
        e5.check_frozen_inputs("rf", _artifact(), _th(t_clear=0.5))


def test_guard_rejects_artifact_from_other_windows():
    with pytest.raises(ValueError, match="Rebuild"):
        e5.check_artifact_windows(_artifact(calib_steps=[29, 33]), CFG)


def test_guard_rejects_wrong_test_or_train_window():
    steps = {"train_final": pd.DataFrame({"time_step": range(1, 35)}),
             "test": pd.DataFrame({"time_step": range(35, 49)})}   # step 49 missing
    with pytest.raises(ValueError, match="test"):
        e5.check_windows(CFG, steps)
    steps["test"] = pd.DataFrame({"time_step": range(35, 50)})
    e5.check_windows(CFG, steps)
    steps["train_final"] = pd.DataFrame({"time_step": range(1, 36)})  # leaks into test
    with pytest.raises(ValueError, match="train_final"):
        e5.check_windows(CFG, steps)


# --- end to end on synthetic data -------------------------------------------

def _frame():
    rng = np.random.default_rng(1)
    n_steps, per_step = 49, 60
    n = n_steps * per_step
    local = rng.normal(size=(n, N_LOCAL))
    agg = rng.normal(size=(n, N_AGG))
    y = (local[:, 0] + agg[:, 0] + 0.5 * rng.normal(size=n) > 1.4).astype(int)
    df = pd.DataFrame(local, columns=[f"local_{i + 1}" for i in range(N_LOCAL)])
    for i in range(N_AGG):
        df[f"agg_{i + 1}"] = agg[:, i]
    df.insert(0, "time_step", np.repeat(np.arange(1, n_steps + 1), per_step))
    df.insert(0, "txId", np.arange(n))
    df["label"] = y
    return df


@pytest.fixture(scope="module")
def frozen(tmp_path_factory):
    base = tmp_path_factory.mktemp("e5")
    df = _frame()
    cols_local = [c for c in df.columns if c.startswith("local_")]
    cols_all = cols_local + [c for c in df.columns if c.startswith("agg_")]
    splits = {
        "train_stage1": df[df.time_step.between(1, 28)].reset_index(drop=True),
        "calib": df[df.time_step.between(29, 34)].reset_index(drop=True),
        "train_final": df[df.time_step.between(1, 34)].reset_index(drop=True),
        "test": df[df.time_step.between(35, 49)].reset_index(drop=True),
    }
    out_dir, art_dir = base / "results", base / "artifacts"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    e3.run_family("lr", SEED, splits, cols_local, e3.threshold_grid({}), 10, out_dir, art_dir)
    artifact = load_stage1(art_dir / "stage1_lr.joblib")
    res = e4.compute_thresholds("lr", artifact, splits, cols_all, e3.threshold_grid({}),
                                0.01, 0.95, SEED)
    th_path = base / "thresholds_lr.json"
    save_thresholds(th_path, "lr", res["d"], res["d2"], res["t_clear"], res["target"],
                    res["target_met"], extra={"validation_escalation_rate":
                                              float(res["chosen_row"]["escalation_rate"])})
    return {"base": base, "splits": splits, "cols_all": cols_all, "artifact": artifact,
            "th": load_thresholds(th_path), "th_path": th_path}


def test_run_writes_all_outputs_and_scores_test_once(frozen):
    out = frozen["base"] / "run1"
    res = e5.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
                 frozen["th"], out, frozen["th_path"])
    for name in ("cascade_predictions.csv", "e5_summary.csv", "e5_per_step.csv",
                 "e5_cost_table.csv", "e5_cost_recall_curve.csv", "e5_run_meta.json",
                 "figures/e5_cost_recall.png", "figures/e5_per_step_f1.png"):
        assert (out / name).exists(), name
    preds = pd.read_csv(out / "cascade_predictions.csv")
    assert len(preds) == len(frozen["splits"]["test"])
    assert preds["time_step"].between(35, 49).all()
    assert set(res["costs"]["r"]) == {5, 20, 100}
    meta = json.loads((out / "e5_run_meta.json").read_text())
    assert meta["stage2_final_trained_on_steps"] == [1, 34]
    assert meta["test_steps"] == [35, 49]
    assert meta["thresholds"]["t_clear"] == pytest.approx(frozen["th"]["t_clear"])


def test_thresholds_are_not_changed_by_the_run(frozen):
    before = frozen["th_path"].read_bytes()
    e5.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
           frozen["th"], frozen["base"] / "run2", frozen["th_path"])
    assert frozen["th_path"].read_bytes() == before


def test_run_refuses_to_overwrite_without_force(frozen):
    out = frozen["base"] / "run3"
    args = ("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
            frozen["th"], out, frozen["th_path"])
    e5.run(*args)
    with pytest.raises(SystemExit):
        e5.run(*args)
    e5.run(*args, force=True)


def test_run_is_deterministic(frozen):
    a = e5.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
               frozen["th"], frozen["base"] / "run4", frozen["th_path"])
    b = e5.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
               frozen["th"], frozen["base"] / "run5", frozen["th_path"])
    pd.testing.assert_frame_equal(a["summary"], b["summary"])
    pd.testing.assert_frame_equal(a["preds"], b["preds"])