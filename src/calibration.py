"""Probability calibration for Stage 1 (SPEC.md Section 6, steps 2-3).

Signatures
----------
    fit_sigmoid(p_raw, y, eps=1e-4)               -> SigmoidCalibrator
    calibrate_sigmoid(model, X_val, y_val, eps)   -> SigmoidCalibrator
    predict_p1(model, calibrator, X)              -> np.ndarray
    brier(y_true, p)                              -> float
    reliability_data(y_true, p, n_bins, strategy) -> DataFrame
    choose_d(y_true, p, grid)                     -> (d, DataFrame)

Method: Platt (sigmoid) scaling on the logit of the model's probability,
    p_cal = sigmoid(a * logit(p_raw) + b)
fitted by (effectively unregularised) logistic regression on the validation
steps only. a = 1, b = 0 is the identity,
so a model that is already calibrated is left unchanged.

Facts worth knowing for the viva
  * a > 0 is enforced, so the map is strictly increasing: calibration never
    changes the ranking, hence never changes PR-AUC. It changes what the
    numbers MEAN, which is what the router threshold t_clear needs.
  * The calibrator learns the illicit share of the validation steps. If
    that share differs from the test steps, calibrated probabilities will
    be off on test. That is a finding to report, never a reason to
    recalibrate on test.
Flagged means p >= threshold everywhere (same rule as p1 >= t_clear).
"""

import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.linear_model import LogisticRegression

from src.evaluate import illicit_metrics

DEFAULT_EPS = 1e-4


def _binary(y, name="y"):
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(
            f"{name} must contain only 0 and 1 (labelled nodes only; "
            "did label -1 slip in?)"
        )
    return arr.astype(int)


def _probs(p, name="p"):
    arr = np.asarray(p, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if np.isnan(arr).any() or (arr < 0).any() or (arr > 1).any():
        raise ValueError(f"{name} must be probabilities in [0, 1] with no NaN")
    return arr


def _logit(p, eps):
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


class SigmoidCalibrator:
    """p_cal = sigmoid(a * logit(clip(p_raw)) + b)."""

    def __init__(self, a, b, eps=DEFAULT_EPS):
        if not a > 0:
            raise ValueError(f"slope a must be positive, got {a}")
        self.a = float(a)
        self.b = float(b)
        self.eps = float(eps)

    def transform(self, p_raw):
        p_raw = _probs(p_raw, "p_raw")
        return expit(self.a * _logit(p_raw, self.eps) + self.b)

    def as_dict(self):
        return {"a": self.a, "b": self.b, "eps": self.eps}

    @classmethod
    def from_dict(cls, d):
        return cls(d["a"], d["b"], d.get("eps", DEFAULT_EPS))

    def __repr__(self):
        return f"SigmoidCalibrator(a={self.a:.4f}, b={self.b:.4f})"


def fit_sigmoid(p_raw, y, eps=DEFAULT_EPS):
    """Fit Platt scaling on raw probabilities p_raw against 0/1 labels y."""
    p_raw = _probs(p_raw, "p_raw")
    y = _binary(y)
    if len(p_raw) != len(y):
        raise ValueError("p_raw and y must have the same length")
    if y.min() == y.max():
        raise ValueError("need both classes in the validation labels")

    z = _logit(p_raw, eps).reshape(-1, 1)
    # C is huge so the L2 penalty is negligible (works on every sklearn version)
    lr = LogisticRegression(C=1e10, solver="lbfgs", max_iter=5000)
    lr.fit(z, y)
    a, b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
    if not a > 0:
        raise ValueError(
            f"fitted slope {a:.4f} is not positive: the model has no usable "
            "signal on the validation steps, so calibrating would flip it."
        )
    return SigmoidCalibrator(a, b, eps)


def _p_illicit(model, X):
    proba = model.predict_proba(X)
    return proba[:, list(model.classes_).index(1)]


def calibrate_sigmoid(model, X_val, y_val, eps=DEFAULT_EPS):
    """Fit a SigmoidCalibrator on a fitted model's output for the validation
    rows. The model itself is not modified."""
    return fit_sigmoid(_p_illicit(model, X_val), y_val, eps)


def predict_p1(model, calibrator, X):
    """P(illicit) from the model, passed through the calibrator if given."""
    p_raw = _p_illicit(model, X)
    return p_raw if calibrator is None else calibrator.transform(p_raw)


def brier(y_true, p):
    """Mean squared error of predicted probabilities. Lower is better."""
    y = _binary(y_true, "y_true")
    p = _probs(p)
    if len(y) != len(p):
        raise ValueError("y_true and p must have the same length")
    if len(y) == 0:
        raise ValueError("empty input")
    return float(np.mean((p - y) ** 2))


def reliability_data(y_true, p, n_bins=10, strategy="quantile"):
    """Reliability-diagram points: per bin, mean predicted vs observed rate.

    strategy "quantile": equal-count bins (best for imbalanced data);
    "uniform": equal-width bins on [0, 1]. The last bin includes its upper
    edge. Empty bins are dropped.
    """
    y = _binary(y_true, "y_true")
    p = _probs(p)
    if len(y) != len(p):
        raise ValueError("y_true and p must have the same length")
    if len(y) == 0:
        raise ValueError("empty input")
    if n_bins < 1:
        raise ValueError("n_bins must be at least 1")

    if strategy == "uniform":
        edges = np.linspace(0.0, 1.0, n_bins + 1)
    elif strategy == "quantile":
        edges = np.unique(np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1)))
    else:
        raise ValueError("strategy must be 'quantile' or 'uniform'")

    if edges.size < 2:                      # every p identical
        idx = np.zeros(len(p), dtype=int)
        n_edges_bins = 1
    else:
        idx = np.clip(np.searchsorted(edges, p, side="right") - 1,
                      0, edges.size - 2)
        n_edges_bins = edges.size - 1

    rows = []
    for b in range(n_edges_bins):
        mask = idx == b
        if not mask.any():
            continue
        rows.append({
            "bin": b,
            "n": int(mask.sum()),
            "mean_predicted": float(p[mask].mean()),
            "observed_rate": float(y[mask].mean()),
        })
    return pd.DataFrame(rows)


def choose_d(y_true, p, grid):
    """Standalone decision threshold d: the grid value that maximises illicit
    F1 (flag when p >= threshold). Ties go to the larger threshold.

    Returns (d, table) where table has threshold, precision, recall, f1,
    n_flagged for every grid value.
    """
    y = _binary(y_true, "y_true")
    p = _probs(p)
    if len(y) != len(p):
        raise ValueError("y_true and p must have the same length")
    values = sorted({float(g) for g in grid})
    if not values:
        raise ValueError("grid is empty")
    if not any(0.0 <= g <= 1.0 for g in values):
        raise ValueError("grid values must lie in [0, 1]")

    rows = []
    for t in values:
        m = illicit_metrics(y, (p >= t).astype(int))
        rows.append({
            "threshold": t,
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "n_flagged": m["n_flagged"],
        })
    table = pd.DataFrame(rows)
    if table["f1"].isna().all():
        raise ValueError("F1 is undefined everywhere (no illicit rows?)")
    best = table["f1"].max()
    tied = table[np.isclose(table["f1"], best, rtol=0.0, atol=1e-12)]
    return float(tied["threshold"].max()), table