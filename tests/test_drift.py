"""Tests for src/drift.py. Counts below are worked out by hand.

    pytest tests/test_drift.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drift import (  # noqa: E402
    aggregate_seeds,
    plot_drift,
    pooled_metrics,
    step_metrics_table,
)

# Step 1: y=[1,1,0,0,0], pred=[1,0,1,0,0] -> tp=1 fp=1 fn=1 (P=.5 R=.5 F1=.5)
# Step 2: y=[1,1,1,0],   pred=[1,1,0,0]   -> tp=2 fp=0 fn=1 (P=1 R=2/3 F1=.8)
STEPS = [1] * 5 + [2] * 4
Y = [1, 1, 0, 0, 0, 1, 1, 1, 0]
P = [1, 0, 1, 0, 0, 1, 1, 0, 0]


def test_hand_computed_per_step_values():
    t = step_metrics_table(STEPS, Y, P)
    assert t.loc[1, ["tp", "fp", "fn"]].tolist() == [1, 1, 1]
    assert t.loc[2, ["tp", "fp", "fn"]].tolist() == [2, 0, 1]
    assert t.loc[1, "precision"] == pytest.approx(0.5)
    assert t.loc[1, "recall"] == pytest.approx(0.5)
    assert t.loc[1, "f1"] == pytest.approx(0.5)
    assert t.loc[2, "precision"] == pytest.approx(1.0)
    assert t.loc[2, "recall"] == pytest.approx(2 / 3)
    assert t.loc[2, "f1"] == pytest.approx(0.8)


def test_pooled_uses_summed_counts_not_mean_of_steps():
    t = step_metrics_table(STEPS, Y, P)
    pooled = pooled_metrics(t)
    assert (pooled["tp"], pooled["fp"], pooled["fn"]) == (3, 1, 2)
    assert pooled["precision"] == pytest.approx(0.75)
    assert pooled["recall"] == pytest.approx(0.6)
    assert pooled["f1"] == pytest.approx(6 / 9)


def test_counts_add_up_to_input():
    t = step_metrics_table(STEPS, Y, P)
    assert t["n_labelled"].sum() == len(Y)
    assert t["n_illicit"].sum() == sum(Y)
    assert t["n_flagged"].sum() == sum(P)


def test_undefined_cases_are_nan_not_zero():
    # step 1: nothing flagged -> precision undefined; recall 0, F1 0
    # step 2: no illicit rows -> recall and F1 undefined; precision 0
    t = step_metrics_table([1, 1, 2, 2], [1, 0, 0, 0], [0, 0, 0, 1])
    assert np.isnan(t.loc[1, "precision"])
    assert t.loc[1, "recall"] == 0
    assert t.loc[1, "f1"] == 0
    assert t.loc[2, "precision"] == 0
    assert np.isnan(t.loc[2, "recall"])
    assert np.isnan(t.loc[2, "f1"])


def test_unlabelled_or_non_binary_input_is_rejected():
    with pytest.raises(ValueError):
        step_metrics_table([1, 1], [1, -1], [1, 0])
    with pytest.raises(ValueError):
        step_metrics_table([1, 1], [1, 0], [1, 2])


def test_aggregate_seeds_mean_and_std():
    a = step_metrics_table(STEPS, Y, P)
    b = a.copy()
    b["f1"] = [0.7, 0.6]           # step 1, step 2
    agg = aggregate_seeds([a, b])
    assert agg.loc[1, "f1_mean"] == pytest.approx(0.6)   # (0.5 + 0.7) / 2
    assert agg.loc[1, "f1_std"] == pytest.approx(0.1)    # population std
    assert agg.loc[2, "f1_mean"] == pytest.approx(0.7)   # (0.8 + 0.6) / 2
    assert (agg["n_seeds"] == 2).all()


def test_single_seed_has_zero_std():
    agg = aggregate_seeds([step_metrics_table(STEPS, Y, P)])
    assert (agg["f1_std"] == 0).all()


def test_mismatched_steps_are_rejected():
    a = step_metrics_table(STEPS, Y, P)
    b = step_metrics_table([5, 5], [1, 0], [1, 0])
    with pytest.raises(ValueError):
        aggregate_seeds([a, b])


def test_plot_writes_a_file(tmp_path):
    rng = np.random.default_rng(0)
    steps = np.repeat(np.arange(35, 50), 40)
    y = rng.integers(0, 2, size=steps.size)
    summary = {}
    for name in ("LOCAL", "ALL"):
        p = np.where(rng.random(steps.size) < 0.8, y, 1 - y)
        summary[name] = aggregate_seeds([step_metrics_table(steps, y, p)])
    out = tmp_path / "drift.png"
    plot_drift(summary, "toy", out)
    assert out.exists() and out.stat().st_size > 5_000