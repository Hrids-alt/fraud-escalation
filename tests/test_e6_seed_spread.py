"""Tests for scripts/e6_seed_spread.py (Day 9 seed spread). Synthetic data only."""
import importlib.util
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
e6 = _load("e6_seed_spread", "e6_seed_spread.py")

from src.cascade import load_stage1, load_thresholds, save_thresholds  # noqa: E402

SEED = 42
N_LOCAL, N_AGG = 8, 4
CFG = {"seed": SEED, "splits": {"train_stage1": [1, 28], "calib": [29, 34],
                                "train_final": [1, 34], "test": [35, 49]}}


def test_parse_seeds():
    assert e6.parse_seeds("42,43,44") == (42, 43, 44)
    assert e6.parse_seeds(" 1 , 2 ") == (1, 2)
    for bad in ("42", "a,b", "1,1", ""):
        with pytest.raises(ValueError):
            e6.parse_seeds(bad)


def test_spread_table_mean_and_std_by_hand():
    per_seed = pd.DataFrame({
        "seed": [1, 2, 3], "system": ["cascade"] * 3,
        "escalation_rate": [0.1, 0.1, 0.1], "precision": [0.8, 0.9, 1.0],
        "recall": [0.5, 0.5, 0.5], "f1": [0.6, 0.7, 0.8], "recall_retention": [1.0, 1.0, 1.0],
    })
    table = e6.spread_table(per_seed)
    mean = table[(table["system"] == "cascade") & (table["seed"] == "mean")].iloc[0]
    std = table[(table["system"] == "cascade") & (table["seed"] == "std")].iloc[0]
    assert mean["precision"] == pytest.approx(0.9)
    assert std["precision"] == pytest.approx(0.1)       # sample std of 0.8, 0.9, 1.0
    assert std["recall"] == pytest.approx(0.0)
    assert len(table) == 5                              # 3 seeds + mean + std


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
    base = tmp_path_factory.mktemp("e6")
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
                    res["target_met"])
    return {"base": base, "splits": splits, "cols_all": cols_all, "artifact": artifact,
            "th": load_thresholds(th_path), "th_path": th_path}


def test_run_writes_one_row_per_seed_and_system(frozen):
    out = frozen["base"] / "run1"
    table = e6.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
                   frozen["th"], out, seeds=(42, 43, 44))
    assert (out / "e6_seed_spread.csv").exists()
    per_seed = table[table["seed"].astype(str).isin(["42", "43", "44"])]
    assert len(per_seed) == 3 * len(e6.SYSTEMS)
    assert {"mean", "std"} <= set(table["seed"].astype(str))
    assert set(per_seed["system"]) == set(e6.SYSTEMS)


def test_thresholds_and_day8_files_are_untouched(frozen):
    before = frozen["th_path"].read_bytes()
    out = frozen["base"] / "run2"
    out.mkdir()
    day8 = out / "cascade_predictions.csv"
    day8.write_text("keep me", encoding="utf-8")
    e6.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
           frozen["th"], out, seeds=(42, 43))
    assert frozen["th_path"].read_bytes() == before
    assert day8.read_text(encoding="utf-8") == "keep me"
    assert sorted(p.name for p in out.iterdir()) == ["cascade_predictions.csv",
                                                      "e6_seed_spread.csv"]


def test_refuses_to_overwrite_without_force(frozen):
    out = frozen["base"] / "run3"
    args = ("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"], frozen["th"], out)
    e6.run(*args, seeds=(42, 43))
    with pytest.raises(SystemExit):
        e6.run(*args, seeds=(42, 43))
    e6.run(*args, seeds=(42, 43), force=True)


def test_refuses_inconsistent_frozen_inputs(frozen):
    bad = dict(frozen["th"])
    bad["d"] = frozen["th"]["d"] + 0.05
    with pytest.raises(ValueError, match="d differs"):
        e6.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
               bad, frozen["base"] / "run4", seeds=(42, 43))
    assert not (frozen["base"] / "run4" / "e6_seed_spread.csv").exists()


def test_stage1_is_the_same_for_every_seed(frozen):
    """Only Stage 2 varies, so the local-only row must be identical across seeds."""
    out = frozen["base"] / "run5"
    table = e6.run("lr", CFG, frozen["splits"], frozen["cols_all"], frozen["artifact"],
                   frozen["th"], out, seeds=(42, 43, 44))
    local = table[(table["system"] == "local_only")
                  & table["seed"].astype(str).isin(["42", "43", "44"])]
    assert local["f1"].nunique() == 1
    assert local["recall"].nunique() == 1