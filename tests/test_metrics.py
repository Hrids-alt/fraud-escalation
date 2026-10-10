"""Hand-computed checks for src/evaluate.py (SPEC.md section 11).

Toy example (10 rows), worked out by hand:
    y_true : 1 1 1 0 0 0 0 0 0 0
    y_pred : 1 1 0 1 0 0 0 0 0 0
    -> tp=2, fp=1, fn=1, tn=6
    precision = 2/3, recall = 2/3, F1 = 2*2 / (2*2 + 1 + 1) = 2/3
    scores : 0.9 0.8 0.4 0.7 0.3 0.2 0.1 0.1 0.05 0.0
    ranking by score: pos(0.9), pos(0.8), neg(0.7), pos(0.4), neg ...
    precision at each positive: 1/1, 2/2, 3/4  ->  AP = (1 + 1 + 0.75) / 3
"""
import math

import numpy as np
import pandas as pd
import pytest

from src.evaluate import (
    cost_table,
    escalation_rate,
    illicit_metrics,
    per_step_metrics,
    recall_retention,
    relative_cost,
)

Y_TRUE = [1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
Y_PRED = [1, 1, 0, 1, 0, 0, 0, 0, 0, 0]
SCORE = [0.9, 0.8, 0.4, 0.7, 0.3, 0.2, 0.1, 0.1, 0.05, 0.0]


def test_toy_example_counts_and_metrics():
    m = illicit_metrics(Y_TRUE, Y_PRED, SCORE)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["n"] == 10 and m["n_illicit"] == 3 and m["n_flagged"] == 3
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)
    assert m["f1"] == pytest.approx(2 / 3)
    assert m["pr_auc"] == pytest.approx((1 + 1 + 0.75) / 3)


def test_everything_licit_has_zero_recall_and_undefined_precision():
    # The "predict everything licit" sanity check: looks fine on accuracy, useless on recall.
    m = illicit_metrics(Y_TRUE, [0] * 10)
    assert m["recall"] == 0.0
    assert math.isnan(m["precision"])
    assert m["f1"] == 0.0
    accuracy = (m["tp"] + m["tn"]) / m["n"]
    assert accuracy == pytest.approx(0.7)  # high-looking accuracy, zero recall


def test_no_illicit_examples_gives_nan_not_zero():
    m = illicit_metrics([0, 0, 0], [0, 0, 0], [0.1, 0.2, 0.3])
    assert math.isnan(m["recall"])
    assert math.isnan(m["precision"])
    assert math.isnan(m["f1"])
    assert math.isnan(m["pr_auc"])


def test_unlabelled_value_is_rejected():
    with pytest.raises(ValueError):
        illicit_metrics([1, -1, 0], [1, 0, 0])


def test_length_mismatch_and_empty_are_rejected():
    with pytest.raises(ValueError):
        illicit_metrics([1, 0], [1])
    with pytest.raises(ValueError):
        illicit_metrics([], [])


def test_per_step_metrics():
    df = pd.DataFrame(
        {
            "time_step": [1] * 5 + [2] * 5,
            "label": Y_TRUE,
            "pred": Y_PRED,
            "score": SCORE,
        }
    )
    out = per_step_metrics(df, score_col="score").set_index("time_step")
    # Step 1 rows: y_true 1 1 1 0 0, y_pred 1 1 0 1 0 -> tp=2, fp=1, fn=1
    assert out.loc[1, "tp"] == 2 and out.loc[1, "fp"] == 1 and out.loc[1, "fn"] == 1
    assert out.loc[1, "f1"] == pytest.approx(2 / 3)
    # Step 2 has no illicit and nothing flagged -> undefined, shown as NaN
    assert out.loc[2, "n_illicit"] == 0
    assert math.isnan(out.loc[2, "recall"]) and math.isnan(out.loc[2, "f1"])
    assert list(out.index) == [1, 2]


def test_escalation_rate():
    p1 = [0.05, 0.10, 0.30, 0.60, 0.95]
    assert escalation_rate(p1, 0.30) == pytest.approx(3 / 5)  # >= threshold is escalated
    assert escalation_rate(p1, 0.0) == 1.0
    assert escalation_rate(p1, 1.01) == 0.0


def test_recall_retention():
    assert recall_retention(0.45, 0.60) == pytest.approx(0.75)
    assert math.isnan(recall_retention(0.45, 0.0))
    assert math.isnan(recall_retention(0.45, float("nan")))


def test_relative_cost_matches_hand_calculation():
    c = relative_cost(0.3, 20)
    assert c["graph_always_cost"] == 21
    assert c["cascade_cost"] == pytest.approx(7.0)
    assert c["relative_savings"] == pytest.approx(1 - 7 / 21)
    c5 = relative_cost(0.3, 5)
    assert c5["relative_savings"] == pytest.approx(1 - 2.5 / 6)


def test_relative_cost_extremes():
    # Escalating everything saves nothing; escalating nothing saves r/(1+r).
    assert relative_cost(1.0, 20)["relative_savings"] == pytest.approx(0.0)
    assert relative_cost(0.0, 20)["relative_savings"] == pytest.approx(20 / 21)


def test_relative_cost_validation():
    with pytest.raises(ValueError):
        relative_cost(1.2, 5)
    with pytest.raises(ValueError):
        relative_cost(0.5, 0)


def test_cost_table_has_one_row_per_r():
    t = cost_table(0.3)
    assert list(t["r"]) == [5, 20, 100]
    assert np.all(np.diff(t["relative_savings"]) > 0)  # savings grow as graph features get costlier