"""Evaluation helpers (SPEC.md sections 8 and 9).

Conventions
-----------
* Illicit is the positive class: y = 1 illicit, y = 0 licit.
* Only LABELLED transactions may be evaluated. The unlabelled value -1 is
  rejected on purpose, so unlabelled nodes can never leak into the metrics.
* Undefined metrics are returned as NaN, never silently as 0:
    - precision is NaN when nothing was flagged (tp + fp == 0)
    - recall    is NaN when there are no illicit transactions (tp + fn == 0)
    - F1 = 2*tp / (2*tp + fp + fn), NaN only when that denominator is 0
    - PR-AUC    is NaN when there are no illicit transactions or no scores
  Per-time-step tables therefore show a gap for steps with no illicit
  examples instead of a misleading 0.
* Plain accuracy is deliberately not provided.
"""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

DEFAULT_R_VALUES = (5, 20, 100)


def _as_binary(x, name):
    arr = np.asarray(x)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be 1-dimensional")
    if not np.isin(arr, [0, 1]).all():
        raise ValueError(
            f"{name} must contain only 0 and 1 "
            "(did unlabelled rows with label -1 slip in?)"
        )
    return arr.astype(int)


def _safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else float("nan")


def illicit_metrics(y_true, y_pred, y_score=None):
    """Illicit-class metrics for one set of transactions.

    Returns a dict with: n, n_illicit, n_flagged, tp, fp, fn, tn,
    precision, recall, f1, pr_auc.
    """
    y_true = _as_binary(y_true, "y_true")
    y_pred = _as_binary(y_pred, "y_pred")
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if len(y_true) == 0:
        raise ValueError("cannot evaluate an empty set")

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())

    pr_auc = float("nan")
    if y_score is not None and (y_true == 1).any():
        y_score = np.asarray(y_score, dtype=float)
        if len(y_score) != len(y_true):
            raise ValueError("y_score must have the same length as y_true")
        pr_auc = float(average_precision_score(y_true, y_score))

    return {
        "n": len(y_true),
        "n_illicit": tp + fn,
        "n_flagged": tp + fp,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "f1": _safe_div(2 * tp, 2 * tp + fp + fn),
        "pr_auc": pr_auc,
    }


def per_step_metrics(df, y_true_col="label", y_pred_col="pred",
                     score_col=None, step_col="time_step"):
    """illicit_metrics for every time step. One row per step, sorted by step."""
    rows = []
    for step, g in df.groupby(step_col, sort=True):
        scores = g[score_col] if score_col is not None else None
        m = illicit_metrics(g[y_true_col], g[y_pred_col], scores)
        rows.append({step_col: step, **m})
    return pd.DataFrame(rows)


def escalation_rate(p1, t_clear):
    """Share of transactions escalated to Stage 2: mean(p1 >= t_clear)."""
    p1 = np.asarray(p1, dtype=float)
    if p1.size == 0:
        raise ValueError("p1 is empty")
    return float((p1 >= t_clear).mean())


def recall_retention(recall_cascade, recall_graph_always):
    """recall(cascade) / recall(graph-always). NaN if the denominator is 0 or NaN."""
    if recall_graph_always is None or np.isnan(recall_graph_always) or recall_graph_always == 0:
        return float("nan")
    return float(recall_cascade) / float(recall_graph_always)


def relative_cost(escalation_rate, r):
    """Simulated relative cost (SPEC.md section 9).

    Local-feature cost = 1; graph features cost r times local.
        graph_always_cost = 1 + r
        cascade_cost      = 1 + escalation_rate * r
        relative_savings  = 1 - cascade_cost / graph_always_cost
    """
    if not 0.0 <= escalation_rate <= 1.0:
        raise ValueError("escalation_rate must be between 0 and 1")
    if r <= 0:
        raise ValueError("r must be positive")
    graph_always = 1.0 + r
    cascade = 1.0 + escalation_rate * r
    return {
        "r": r,
        "escalation_rate": escalation_rate,
        "graph_always_cost": graph_always,
        "cascade_cost": cascade,
        "relative_savings": 1.0 - cascade / graph_always,
    }


def cost_table(escalation_rate, r_values=DEFAULT_R_VALUES):
    """relative_cost for several r values as a DataFrame."""
    return pd.DataFrame([relative_cost(escalation_rate, r) for r in r_values])