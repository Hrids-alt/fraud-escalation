"""Cost-aware two-stage cascade (SPEC.md sections 6-9).

Pure functions on arrays: no training, no file I/O except the small helpers
at the bottom. Nothing here refits Stage 1; it only loads the frozen artifact
written by scripts/e3_calibration.py.

Vocabulary
    p1       calibrated Stage 1 probability of illicit (LOCAL features)
    p2       Stage 2 probability of illicit (ALL features, same model family)
    d        standalone Stage 1 threshold (frozen on Day 6)
    d2       Stage 2 decision threshold (chosen once on steps 29-34 from the
             validation copy of Stage 2, then frozen; see SPEC.md)
    t_clear  routing threshold, 0 <= t_clear <= d (chosen on steps 29-34)

Routing rule
    p1 <  t_clear  -> cleared as licit, graph features never used
    p1 >= t_clear  -> escalated; Stage 2 decides: illicit if p2 >= d2

Because the graph features are precomputed in this dataset, p2 may exist for
every row. The cascade only ever READS p2 for escalated rows, and the cost of
using it is simulated with src.evaluate.relative_cost.

Steps 35-49 are never used to choose any threshold. These functions take
labels only in sweep_t_clear (which is called on steps 29-34).
"""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.calibration import SigmoidCalibrator, predict_p1
from src.evaluate import illicit_metrics, recall_retention
from src.models import FAMILIES

REBUILD_HINT = "Run `python scripts/e3_calibration.py` to rebuild the Stage 1 artifacts."
TOL = 1e-9


# --------------------------------------------------------------------------
# Validation helpers
# --------------------------------------------------------------------------

def _as_prob_array(x, name, allow_nan=False):
    """1-D float array of probabilities: finite and within [0, 1].

    NaN is rejected unless allow_nan (p2 may be NaN for cleared rows).
    Infinite or out-of-range values are always rejected.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if np.isinf(arr).any():
        raise ValueError(f"{name} contains infinite values")
    nan = np.isnan(arr)
    if nan.any() and not allow_nan:
        raise ValueError(f"{name} contains NaN")
    known = arr[~nan]
    # A tiny overshoot (e.g. 1.0000000000000002 from averaging tree outputs) is
    # floating-point noise, not an error; anything larger is rejected.
    if known.size and (known.min() < -TOL or known.max() > 1.0 + TOL):
        raise ValueError(f"{name} must lie in [0, 1]")
    return np.clip(arr, 0.0, 1.0)


def _check_threshold(value, name):
    """A decision threshold must be a finite number in [0, 1]."""
    if not np.isfinite(value) or not (0.0 <= value <= 1.0):
        raise ValueError(f"{name} must be a number in [0, 1], got {value}")
    return float(value)


def validate_thresholds(t_clear, d, d2=None):
    """Raise ValueError unless 0 <= t_clear <= d <= 1 (and d2 in [0, 1] if given)."""
    _check_threshold(d, "d")
    if d2 is not None:
        _check_threshold(d2, "d2")
    if not np.isfinite(t_clear) or not (0.0 <= t_clear <= d + TOL):
        raise ValueError(
            f"t_clear must satisfy 0 <= t_clear <= d (d={d}), got t_clear={t_clear}"
        )


# --------------------------------------------------------------------------
# Routing and the systems compared in RQ2
# --------------------------------------------------------------------------

def route(p1, t_clear):
    """Boolean mask: True where the transaction is escalated (p1 >= t_clear)."""
    p1 = _as_prob_array(p1, "p1")
    return p1 >= _check_threshold(t_clear, "t_clear")


def _decide(escalated, p1, p2, d1, d2):
    """Labels: escalated rows use p2 >= d2, others use p1 >= d1."""
    p1 = _as_prob_array(p1, "p1")
    p2 = _as_prob_array(p2, "p2", allow_nan=True)
    if len(p1) != len(p2) or len(p1) != len(escalated):
        raise ValueError("p1, p2 and the escalation mask must have the same length")
    if np.isnan(p2[escalated]).any():
        raise ValueError("p2 is NaN for an escalated transaction")
    p2_safe = np.where(escalated, p2, 0.0)
    return np.where(escalated, p2_safe >= d2, p1 >= d1).astype(int)


def run_cascade(p1, p2, t_clear, d2):
    """The cascade. Cleared rows are licit (0); escalated rows follow Stage 2.

    Returns {"escalated": bool array, "pred": int array, "escalation_rate": float}.
    p2 is read only where escalated, so it may be NaN elsewhere.
    """
    p1 = _as_prob_array(p1, "p1")
    d2 = _check_threshold(d2, "d2")
    escalated = route(p1, t_clear)
    # d1 = 2.0 is never reached, so non-escalated rows are always licit.
    pred = _decide(escalated, p1, p2, d1=2.0, d2=d2)
    return {
        "escalated": escalated,
        "pred": pred,
        "escalation_rate": float(escalated.mean()) if len(escalated) else float("nan"),
    }


def local_only(p1, d):
    """Baseline: frozen Stage 1 with threshold d on every transaction."""
    p1 = _as_prob_array(p1, "p1")
    return (p1 >= _check_threshold(d, "d")).astype(int)


def graph_always(p2, d2):
    """Baseline: Stage 2 on every transaction."""
    p2 = _as_prob_array(p2, "p2")
    return (p2 >= _check_threshold(d2, "d2")).astype(int)


def random_control(p1, p2, d, d2, escalation_rate, seed):
    """Random-escalation control (SPEC.md section 7).

    Escalate exactly round(escalation_rate * n) transactions chosen at random;
    escalated rows use Stage 2 (p2 >= d2), the rest keep the Stage 1 decision
    (p1 >= d). Same return shape as run_cascade.
    """
    p1 = _as_prob_array(p1, "p1")
    p2 = _as_prob_array(p2, "p2", allow_nan=True)
    d = _check_threshold(d, "d")
    d2 = _check_threshold(d2, "d2")
    if len(p1) != len(p2):
        raise ValueError("p1 and p2 must have the same length")
    if not np.isfinite(escalation_rate) or not (0.0 <= escalation_rate <= 1.0):
        raise ValueError(f"escalation_rate must be a finite number in [0, 1], got {escalation_rate}")
    n = len(p1)
    k = int(round(escalation_rate * n))
    rng = np.random.default_rng(seed)
    chosen = rng.choice(n, size=k, replace=False) if k else np.array([], dtype=int)
    escalated = np.zeros(n, dtype=bool)
    escalated[chosen] = True
    pred = _decide(escalated, p1, p2, d1=d, d2=d2)
    return {"escalated": escalated, "pred": pred,
            "escalation_rate": float(escalated.mean()) if n else float("nan")}


# --------------------------------------------------------------------------
# Choosing t_clear (steps 29-34 only)
# --------------------------------------------------------------------------

def t_clear_grid(d, step=0.01):
    """Candidate t_clear values 0, step, 2*step, ... up to and including d (if on the grid).

    0 means 'escalate everything', so a retention target of 1.0 is always reachable.
    """
    if not np.isfinite(d) or not (0.0 <= d <= 1.0):
        raise ValueError(f"d must be a finite number in [0, 1], got {d}")
    if not np.isfinite(step) or not (0.0 < step <= 1.0):
        raise ValueError(f"step must be a finite number in (0, 1], got {step}")
    grid = np.round(np.arange(0.0, d + step / 2, step), 6)
    return grid[grid <= d + TOL]


def sweep_t_clear(y_true, p1, p2, d, d2, grid):
    """Validation sweep over t_clear on the calibration steps.

    p2 must come from the VALIDATION copy of Stage 2 (trained on steps 1-28).
    Retention is recall(cascade) / recall(graph-always at d2). Every grid value
    must satisfy 0 <= t_clear <= d.

    Returns a DataFrame with one row per t_clear: escalation_rate, precision,
    recall, f1, recall_retention.
    """
    y = np.asarray(y_true)
    if y.ndim != 1 or not np.isin(y, (0, 1)).all():
        raise ValueError("y_true must be a 1-D array containing only 0 and 1")
    p1 = _as_prob_array(p1, "p1")
    p2 = _as_prob_array(p2, "p2")
    if not (len(y) == len(p1) == len(p2)):
        raise ValueError("y_true, p1 and p2 must have the same length")
    _check_threshold(d, "d")
    _check_threshold(d2, "d2")
    try:
        values = sorted({float(g) for g in grid})
    except (TypeError, ValueError) as exc:
        raise ValueError(f"grid must contain only numbers: {exc}") from exc
    if not values:
        raise ValueError("grid is empty")
    for t in values:
        validate_thresholds(t, d, d2)

    recall_graph = illicit_metrics(y, graph_always(p2, d2))["recall"]
    rows = []
    for t in values:
        out = run_cascade(p1, p2, t, d2)
        m = illicit_metrics(y, out["pred"])
        rows.append({
            "t_clear": t,
            "escalation_rate": out["escalation_rate"],
            "precision": m["precision"],
            "recall": m["recall"],
            "f1": m["f1"],
            "recall_retention": recall_retention(m["recall"], recall_graph),
        })
    return pd.DataFrame(rows)


def choose_t_clear(sweep, target=0.95):
    """Largest t_clear whose recall retention is >= target.

    If no row reaches the target, return the row with the highest retention
    (largest t_clear among ties) and target_met=False, so the report can say
    what was achievable. Returns (t_clear, row_dict, target_met).
    """
    if not 0.0 < target <= 1.0:
        raise ValueError("target must be in (0, 1]")
    ok = sweep[sweep["recall_retention"] >= target - TOL]
    if len(ok):
        row = ok.loc[ok["t_clear"].idxmax()]
        return float(row["t_clear"]), row.to_dict(), True
    valid = sweep.dropna(subset=["recall_retention"])
    if valid.empty:
        raise ValueError("recall retention is undefined everywhere "
                         "(no illicit transactions or zero graph-always recall)")
    best = valid["recall_retention"].max()
    row = valid[np.isclose(valid["recall_retention"], best, rtol=0.0, atol=TOL)]
    row = row.loc[row["t_clear"].idxmax()]
    return float(row["t_clear"]), row.to_dict(), False


# --------------------------------------------------------------------------
# Loading frozen artifacts and thresholds (no refitting anywhere)
# --------------------------------------------------------------------------

def load_stage1(path):
    """Load the frozen Stage 1 artifact written by scripts/e3_calibration.py."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. {REBUILD_HINT}")
    artifact = joblib.load(path)
    for key in ("family", "model", "calibrator", "feature_columns", "d"):
        if key not in artifact:
            raise ValueError(f"{path} is missing '{key}'. {REBUILD_HINT}")
    return artifact


def stage1_p1(artifact, X):
    """Calibrated Stage 1 probability for a DataFrame that has the LOCAL columns."""
    calibrator = SigmoidCalibrator.from_dict(artifact["calibrator"])
    return predict_p1(artifact["model"], calibrator, X[artifact["feature_columns"]])


def save_thresholds(path, family, d, d2, t_clear, target, target_met, extra=None):
    """Write the frozen thresholds (choose them on steps 29-34 only)."""
    if family not in FAMILIES:
        raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
    if not np.isfinite(target) or not (0.0 < target <= 1.0):
        raise ValueError(f"target must be in (0, 1], got {target}")
    validate_thresholds(t_clear, d, d2)
    payload = {
        "family": family,
        "d": float(d),
        "d2": float(d2),
        "t_clear": float(t_clear),
        "recall_retention_target": float(target),
        "target_met_on_validation": bool(target_met),
        "chosen_on_steps": [29, 34],
    }
    if extra:
        payload.update(extra)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def load_thresholds(path):
    """Read frozen thresholds and re-check t_clear <= d."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run the Day 7 threshold step first.")
    payload = json.loads(path.read_text(encoding="utf-8"))
    for key in ("family", "d", "d2", "t_clear", "recall_retention_target"):
        if key not in payload:
            raise ValueError(f"{path} is missing '{key}'")
    if payload["family"] not in FAMILIES:
        raise ValueError(f"{path}: family must be one of {FAMILIES}, got {payload['family']!r}")
    target = payload["recall_retention_target"]
    if not isinstance(target, (int, float)) or not np.isfinite(target) or not (0.0 < target <= 1.0):
        raise ValueError(f"{path}: recall_retention_target must be in (0, 1], got {target!r}")
    validate_thresholds(payload["t_clear"], payload["d"], payload["d2"])
    return payload