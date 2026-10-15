"""Export the frozen results to one JSON file for the web app (Day 9).

    python scripts/export_for_ui.py                  # writes web/data/results.json
    python scripts/export_for_ui.py --family rf

Reads ONLY files that already exist in results/ plus thresholds_<family>.json.
It never reads data/raw, never loads a model and never scores anything, so it
cannot change a result. If a number is wrong in the web app, it is wrong in
results/.

Input files (results/)
    cascade_predictions.csv, e5_summary.csv, e5_per_step.csv, e5_cost_table.csv,
    e5_cost_recall_curve.csv, e5_run_meta.json
    e6_seed_spread.csv  (optional: included when present)

Output
    web/data/results.json   (rounded numbers, null instead of NaN, well under 2 MB)

JSON layout
    meta, kpis, systems, cost, curve, per_step, batches, queue, seed_spread, honesty

Queue rule: every escalated transaction, plus a small seeded sample of cleared
transactions per step (so the app can show both what was cleared and what was
escalated). The sample is deterministic.
"""

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade import load_thresholds  # noqa: E402

PRED_COLUMNS = ["txId", "time_step", "p1", "escalated", "p2", "final_decision", "label",
                "local_only_decision", "graph_always_decision"]
SYSTEM_LABELS = {
    "local_only": "Local-only",
    "graph_always": "Graph-always",
    "cascade": "Cascade",
    "random_control_mean": "Random escalation (mean of 5 seeds)",
}
SYSTEM_ORDER = tuple(SYSTEM_LABELS)
MAX_BYTES = 2_000_000
ROUND = 5


def say(message):
    print(f"[DAY9] {message}")


# --------------------------------------------------------------------------
# JSON-safe conversion
# --------------------------------------------------------------------------

def clean(value):
    """numpy/pandas values -> plain Python; NaN and inf -> None; floats rounded."""
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        f = float(value)
        return round(f, ROUND) if math.isfinite(f) else None
    if value is None or isinstance(value, (int, str)):
        return value
    if value is pd.NA or value is pd.NaT:
        return None
    return str(value)


def records(frame, columns):
    return clean(frame[columns].to_dict(orient="records"))


# --------------------------------------------------------------------------
# Loading and validation
# --------------------------------------------------------------------------

def read_inputs(results_dir):
    results_dir = Path(results_dir)
    needed = ["cascade_predictions.csv", "e5_summary.csv", "e5_per_step.csv",
              "e5_cost_table.csv", "e5_cost_recall_curve.csv", "e5_run_meta.json"]
    missing = [name for name in needed if not (results_dir / name).exists()]
    if missing:
        raise FileNotFoundError(
            f"missing in {results_dir}: {', '.join(missing)}. "
            "Run `python scripts/e5_cascade_test.py --family rf` first (Day 8)."
        )
    data = {
        "preds": pd.read_csv(results_dir / "cascade_predictions.csv"),
        "summary": pd.read_csv(results_dir / "e5_summary.csv"),
        "per_step": pd.read_csv(results_dir / "e5_per_step.csv"),
        "cost": pd.read_csv(results_dir / "e5_cost_table.csv"),
        "curve": pd.read_csv(results_dir / "e5_cost_recall_curve.csv"),
        "meta": json.loads((results_dir / "e5_run_meta.json").read_text(encoding="utf-8")),
        "seed_spread": None,
    }
    spread = results_dir / "e6_seed_spread.csv"
    if spread.exists():
        data["seed_spread"] = pd.read_csv(spread)
    return data


def validate(data, th, test_steps):
    """Fail loudly if the files do not describe the same run."""
    preds, meta = data["preds"], data["meta"]
    missing = [c for c in PRED_COLUMNS if c not in preds.columns]
    if missing:
        raise ValueError(f"cascade_predictions.csv is missing columns {missing}")
    lo, hi = test_steps
    if not preds["time_step"].between(lo, hi).all():
        raise ValueError(f"predictions contain steps outside the test window {lo}-{hi}")
    if len(preds) != int(meta["n_test"]):
        raise ValueError(f"predictions have {len(preds)} rows but the run meta says "
                         f"n_test = {meta['n_test']}")
    rate = float(preds["escalated"].mean())
    if abs(rate - float(meta["test_escalation_rate"])) > 1e-9:
        raise ValueError("escalation rate in the predictions differs from the run meta")
    for key in ("d", "d2", "t_clear"):
        if abs(float(meta["thresholds"][key]) - float(th[key])) > 1e-12:
            raise ValueError(f"{key} in e5_run_meta.json differs from the thresholds file; "
                             "the results are not from the frozen thresholds")
    cascade = data["summary"][data["summary"]["system"] == "cascade"]
    if cascade.empty:
        raise ValueError("e5_summary.csv has no cascade row")
    tp = int(((preds["final_decision"] == 1) & (preds["label"] == 1)).sum())
    if tp != int(cascade["tp"].iloc[0]):
        raise ValueError("cascade true positives differ between predictions and summary")
    if not preds["p2"][preds["escalated"] == 0].isna().all():
        raise ValueError("p2 must be blank for cleared rows")


# --------------------------------------------------------------------------
# Building the JSON sections
# --------------------------------------------------------------------------

def build_systems(summary):
    wanted = summary[summary["system"].isin(SYSTEM_ORDER)].copy()
    out = []
    for system in SYSTEM_ORDER:
        row = wanted[wanted["system"] == system]
        if row.empty:
            continue
        r = row.iloc[0]
        entry = {"system": system, "label": SYSTEM_LABELS[system]}
        for col in ("escalation_rate", "precision", "recall", "f1", "pr_auc",
                    "recall_retention", "tp", "fp", "fn"):
            if col in row.columns:
                value = r[col]
                if col in ("tp", "fp", "fn") and pd.notna(value):
                    value = int(value)
                entry[col] = value
        out.append(entry)
    std = summary[summary["system"] == "random_control_std"]
    if not std.empty:
        for entry in out:
            if entry["system"] == "random_control_mean":
                entry["std"] = {c: std.iloc[0][c] for c in ("precision", "recall", "f1")}
    return clean(out)


def build_batches(preds):
    """One row per test step: the batch picker in the app."""
    rows = []
    for step, g in preds.groupby("time_step", sort=True):
        cascade_pos = g["final_decision"] == 1
        rows.append({
            "time_step": int(step),
            "n": len(g),
            "n_illicit": int((g["label"] == 1).sum()),
            "cleared": int((g["escalated"] == 0).sum()),
            "escalated": int((g["escalated"] == 1).sum()),
            "caught_cascade": int((cascade_pos & (g["label"] == 1)).sum()),
            "false_alarms_cascade": int((cascade_pos & (g["label"] == 0)).sum()),
            "missed_cascade": int(((~cascade_pos) & (g["label"] == 1)).sum()),
            "caught_graph_always": int(((g["graph_always_decision"] == 1)
                                        & (g["label"] == 1)).sum()),
            "caught_local_only": int(((g["local_only_decision"] == 1)
                                      & (g["label"] == 1)).sum()),
        })
    return clean(rows)


def outcome(final_decision, label):
    if final_decision == 1 and label == 1:
        return "caught"
    if final_decision == 1 and label == 0:
        return "false alarm"
    if final_decision == 0 and label == 1:
        return "missed"
    return "correctly cleared"


def build_queue(preds, cleared_per_step, seed=0):
    """All escalated rows plus a seeded sample of cleared rows per step."""
    rng = np.random.default_rng(seed)
    esc = preds[preds["escalated"] == 1]
    parts = [esc]
    for _, g in preds[preds["escalated"] == 0].groupby("time_step", sort=True):
        k = min(int(cleared_per_step), len(g))
        if k:
            parts.append(g.iloc[np.sort(rng.choice(len(g), size=k, replace=False))])
    queue = pd.concat(parts).sort_values(["time_step", "txId"]).reset_index(drop=True)
    out = []
    for r in queue.itertuples():
        out.append({
            "txId": int(r.txId), "time_step": int(r.time_step),
            "p1": r.p1, "escalated": bool(r.escalated),
            "p2": None if pd.isna(r.p2) else r.p2,
            "final_decision": int(r.final_decision), "label": int(r.label),
            "local_only_decision": int(r.local_only_decision),
            "graph_always_decision": int(r.graph_always_decision),
            "outcome": outcome(int(r.final_decision), int(r.label)),
            "why": ("p1 below t_clear: cleared at Stage 1, graph features never used"
                    if not r.escalated else
                    "p1 at or above t_clear: escalated; Stage 2 made the final decision"),
        })
    return clean(out)


def build_seed_spread(frame):
    if frame is None:
        return None
    return clean(frame.to_dict(orient="records"))


def build_honesty(meta, systems):
    by = {s["system"]: s for s in systems}
    test_rate = float(meta["test_escalation_rate"])
    val_rate = meta.get("validation_escalation_rate")
    lines = [
        "Cost is simulated: the graph features are precomputed in this dataset, so cost "
        "is modelled as local = 1 and graph = r times local.",
        f"Evaluation is on the test steps {meta['test_steps'][0]}-{meta['test_steps'][1]}, "
        "scored once with thresholds frozen on earlier steps.",
        "Transaction ids are the dataset's own anonymised ids; no real-world identity is "
        "shown or implied.",
        "After step 42 every system collapses to near-zero F1 (temporal drift); the "
        "cascade does not fix that.",
        "Recall retention is a weak test on this split: local-only recall is already "
        "close to graph-always recall.",
        "The Stage 2 threshold d2 was chosen on 19.6%-illicit calibration steps and "
        "did not transfer well to the 6.5%-illicit test steps.",
        "One chronological split and one model family; results are not a guarantee on "
        "future data.",
    ]
    if val_rate is not None:
        lines.insert(1, f"The share of transactions escalated on test ({test_rate:.1%}) is "
                        f"much lower than on validation ({float(val_rate):.1%}) because the "
                        "illicit share fell between the periods. It is reported, not corrected.")
    if "cascade" in by and "graph_always" in by:
        lines.append(f"The cascade F1 ({by['cascade']['f1']:.3f}) is above graph-always "
                     f"({by['graph_always']['f1']:.3f}) largely because graph-always, at its "
                     "frozen threshold, raises many false alarms; this is not evidence that "
                     "the cascade is a better model.")
    return lines


def build(data, th, cleared_per_step=10):
    meta, preds, summary = data["meta"], data["preds"], data["summary"]
    systems = build_systems(summary)
    by = {s["system"]: s for s in systems}
    casc = by["cascade"]
    n = len(preds)
    escalated = int(preds["escalated"].sum())
    n_illicit = int((preds["label"] == 1).sum())
    cost_rows = data["cost"].copy()
    keep_cost = ["r", "graph_always_cost", "cascade_cost", "relative_savings"]

    frozen = data["curve"][data["curve"]["is_frozen_t_clear"].astype(bool)]
    curve_cols = ["t_clear", "escalation_rate", "cascade_precision", "cascade_recall",
                  "cascade_f1", "recall_retention", "random_recall_mean", "random_f1_mean"]
    per_step = {}
    for system in SYSTEM_ORDER:
        g = data["per_step"][data["per_step"]["system"] == system].sort_values("time_step")
        per_step[system] = records(g, ["time_step", "precision", "recall", "f1"])

    return clean({
        "meta": {
            "title": "Cost-aware selective escalation for illicit Bitcoin transaction detection",
            "family": meta["family"],
            "thresholds": {k: th[k] for k in ("d", "d2", "t_clear")},
            "windows": {
                "stage1_trained_on": meta["stage1_trained_on_steps"],
                "calibrated_on": meta["stage1_calibrated_on_steps"],
                "stage2_final_trained_on": meta["stage2_final_trained_on_steps"],
                "test": meta["test_steps"],
            },
            "n_test": n,
            "cost_label": "simulated",
            "dataset": "Elliptic Bitcoin dataset (CC BY-NC-ND 4.0)",
            "recall_retention_target": th.get("recall_retention_target"),
        },
        "kpis": {
            "transactions": n,
            "illicit_transactions": n_illicit,
            "cleared_at_stage1": n - escalated,
            "escalated": escalated,
            "escalation_rate": escalated / n,
            "validation_escalation_rate": meta.get("validation_escalation_rate"),
            "illicit_caught_cascade": casc["tp"],
            "illicit_caught_graph_always": by["graph_always"]["tp"],
            "recall_cascade": casc["recall"],
            "recall_graph_always": by["graph_always"]["recall"],
            "recall_retention": casc["recall_retention"],
            "precision_cascade": casc["precision"],
            "f1_cascade": casc["f1"],
        },
        "systems": systems,
        "cost": {
            "formula": "graph_always = 1 + r; cascade = 1 + escalation_rate * r; "
                       "savings = 1 - cascade / graph_always",
            "escalation_rate": escalated / n,
            "r_values": [int(v) for v in cost_rows["r"]],
            "rows": records(cost_rows, keep_cost),
            "slider": {"min": 5, "max": 100},
            "label": "simulated",
        },
        "curve": {
            "points": records(data["curve"], curve_cols),
            "local_only_recall": float(data["curve"]["local_only_recall"].iloc[0]),
            "graph_always_recall": float(data["curve"]["graph_always_recall"].iloc[0]),
            "frozen_point": (records(frozen, curve_cols)[0] if len(frozen) else None),
            "note": "Descriptive sweep on the test steps; the frozen t_clear is marked "
                    "and was not chosen from this curve.",
        },
        "per_step": per_step,
        "batches": build_batches(preds),
        "queue": build_queue(preds, cleared_per_step),
        "seed_spread": build_seed_spread(data["seed_spread"]),
        "honesty": build_honesty(meta, systems),
    })


def write_json(payload, out_path):
    text = json.dumps(payload, separators=(",", ":"), allow_nan=False)
    if len(text.encode("utf-8")) > MAX_BYTES:
        raise ValueError(f"results.json would be {len(text) / 1e6:.1f} MB (limit "
                         f"{MAX_BYTES / 1e6:.0f} MB); lower --cleared-per-step")
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text, encoding="utf-8")
    return len(text.encode("utf-8"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=("rf", "lr"), default="rf")
    ap.add_argument("--results-dir", default="results")
    ap.add_argument("--thresholds-dir", default=".")
    ap.add_argument("--out", default="web/data/results.json")
    ap.add_argument("--cleared-per-step", type=int, default=10,
                    help="cleared transactions sampled per step for the queue")
    args = ap.parse_args()
    if args.cleared_per_step < 0:
        raise SystemExit("--cleared-per-step must be >= 0")

    th = load_thresholds(Path(args.thresholds_dir) / f"thresholds_{args.family}.json")
    data = read_inputs(args.results_dir)
    if data["meta"]["family"] != args.family:
        raise SystemExit(f"results are for {data['meta']['family']!r}, not {args.family!r}")
    validate(data, th, tuple(data["meta"]["test_steps"]))
    payload = build(data, th, args.cleared_per_step)
    size = write_json(payload, args.out)
    say(f"saved {args.out} ({size / 1e3:.0f} kB): {len(payload['queue'])} queue rows, "
        f"{len(payload['batches'])} batches, seed spread "
        f"{'included' if payload['seed_spread'] else 'not found (run Day 9 e6 first)'}")
    say("Done. Nothing was recomputed; every number comes from results/.")


if __name__ == "__main__":
    main()