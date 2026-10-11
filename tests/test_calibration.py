"""Tests for src/calibration.py. Hand-computed values are worked out in comments.

    pytest tests/test_calibration.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (  # noqa: E402
    SigmoidCalibrator,
    brier,
    calibrate_sigmoid,
    choose_d,
    fit_sigmoid,
    predict_p1,
    reliability_data,
)


class _Stub:
    """Minimal fitted-model stand-in: X is already the raw P(illicit)."""
    classes_ = np.array([0, 1])

    def predict_proba(self, X):
        x = np.asarray(X, dtype=float)
        return np.column_stack([1 - x, x])


def _simulate(n, a_true, b_true, seed=0):
    """Raw scores whose TRUE calibration map is sigmoid(a*logit(raw)+b)."""
    rng = np.random.default_rng(seed)
    raw = rng.beta(0.6, 2.0, size=n).clip(1e-3, 1 - 1e-3)
    z = np.log(raw / (1 - raw))
    p_true = 1 / (1 + np.exp(-(a_true * z + b_true)))
    y = (rng.random(n) < p_true).astype(int)
    return raw, y


# ---- brier -----------------------------------------------------------------

def test_brier_hand_computed():
    # errors 0.1, 0.2, 0.4, 0.4 -> (0.01 + 0.04 + 0.16 + 0.16) / 4 = 0.0925
    assert brier([1, 0, 1, 0], [0.9, 0.2, 0.6, 0.4]) == pytest.approx(0.0925)


def test_brier_rejects_bad_input():
    with pytest.raises(ValueError):
        brier([1, -1], [0.5, 0.5])
    with pytest.raises(ValueError):
        brier([1, 0], [1.2, 0.5])
    with pytest.raises(ValueError):
        brier([1, 0], [0.5])


# ---- reliability -----------------------------------------------------------

def test_reliability_uniform_hand_computed():
    # edges [0, .5, 1]: bin0 = {.1,.2,.3} observed 1/3; bin1 = {.8,.9} observed 1
    r = reliability_data([0, 0, 1, 1, 1], [0.1, 0.2, 0.3, 0.8, 0.9],
                         n_bins=2, strategy="uniform")
    assert r["n"].tolist() == [3, 2]
    assert r["mean_predicted"].tolist() == pytest.approx([0.2, 0.85])
    assert r["observed_rate"].tolist() == pytest.approx([1 / 3, 1.0])


def test_reliability_counts_and_upper_edge():
    p = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.0])
    y = np.array([0, 0, 1, 1, 1, 0])
    for strategy in ("uniform", "quantile"):
        r = reliability_data(y, p, n_bins=4, strategy=strategy)
        assert r["n"].sum() == len(p)           # p == 1.0 is not dropped


def test_reliability_all_identical_scores_gives_one_bin():
    r = reliability_data([0, 1, 0, 1], [0.3] * 4, n_bins=5)
    assert len(r) == 1 and r.loc[0, "observed_rate"] == 0.5


# ---- calibrator ------------------------------------------------------------

def test_recovers_known_miscalibration_and_improves_brier():
    raw, y = _simulate(60_000, a_true=0.5, b_true=0.5)
    cal = fit_sigmoid(raw, y)
    assert cal.a == pytest.approx(0.5, abs=0.05)
    assert cal.b == pytest.approx(0.5, abs=0.05)
    assert brier(y, cal.transform(raw)) < brier(y, raw)


def test_already_calibrated_scores_stay_close_to_identity():
    raw, y = _simulate(60_000, a_true=1.0, b_true=0.0, seed=1)
    cal = fit_sigmoid(raw, y)
    assert cal.a == pytest.approx(1.0, abs=0.05)
    assert cal.b == pytest.approx(0.0, abs=0.05)


def test_calibration_preserves_ranking_and_pr_auc():
    raw, y = _simulate(20_000, a_true=0.5, b_true=0.5, seed=2)
    cal = fit_sigmoid(raw, y)
    p = cal.transform(raw)
    assert cal.a > 0
    assert np.all(np.diff(p[np.argsort(raw)]) >= 0)       # monotone
    assert average_precision_score(y, p) == pytest.approx(
        average_precision_score(y, raw), abs=1e-9)


def test_calibrate_sigmoid_uses_the_models_probabilities():
    raw, y = _simulate(20_000, a_true=0.5, b_true=0.5, seed=3)
    via_model = calibrate_sigmoid(_Stub(), raw, y)
    direct = fit_sigmoid(raw, y)
    assert (via_model.a, via_model.b) == pytest.approx((direct.a, direct.b))
    assert predict_p1(_Stub(), via_model, raw) == pytest.approx(
        direct.transform(raw))
    assert predict_p1(_Stub(), None, raw) == pytest.approx(raw)


def test_round_trip_through_dict():
    cal = SigmoidCalibrator(0.7, -0.3)
    again = SigmoidCalibrator.from_dict(cal.as_dict())
    x = np.array([0.01, 0.2, 0.9])
    assert again.transform(x) == pytest.approx(cal.transform(x))


def test_single_class_validation_labels_are_rejected():
    with pytest.raises(ValueError):
        fit_sigmoid([0.1, 0.2, 0.3], [0, 0, 0])


def test_non_binary_labels_are_rejected():
    with pytest.raises(ValueError):
        fit_sigmoid([0.1, 0.2, 0.3], [1, 0, -1])


def test_noise_scores_with_flipped_signal_are_refused():
    # scores perfectly anti-correlated with the label -> slope would be < 0
    p = np.array([0.1] * 50 + [0.9] * 50)
    y = np.array([1] * 50 + [0] * 50)
    with pytest.raises(ValueError):
        fit_sigmoid(p, y)


# ---- choose_d --------------------------------------------------------------
# p = [.9 .8 .4 .3 .2], y = [1 1 1 0 0]
#   t=.10: flags all   -> tp3 fp2        F1 = 6/8 = 0.750
#   t=.25: flags 4     -> tp3 fp1        F1 = 6/7 = 0.857
#   t=.35: flags .9 .8 .4 -> tp3 fp0     F1 = 1.000   <- best
#   t=.50: flags .9 .8 -> tp2 fn1        F1 = 4/5 = 0.800
#   t=.85: flags .9    -> tp1 fn2        F1 = 2/4 = 0.500

P = [0.9, 0.8, 0.4, 0.3, 0.2]
Y = [1, 1, 1, 0, 0]


def test_choose_d_hand_computed():
    d, table = choose_d(Y, P, [0.1, 0.25, 0.35, 0.5, 0.85])
    assert d == 0.35
    assert table["f1"].tolist() == pytest.approx(
        [0.75, 6 / 7, 1.0, 0.8, 0.5])


def test_choose_d_ties_go_to_larger_threshold():
    d, _ = choose_d(Y, P, [0.35, 0.38])      # both flag exactly .9 .8 .4
    assert d == 0.38


def test_choose_d_rejects_empty_or_out_of_range_grid():
    with pytest.raises(ValueError):
        choose_d(Y, P, [])
    with pytest.raises(ValueError):
        choose_d(Y, P, [1.5, 2.0])