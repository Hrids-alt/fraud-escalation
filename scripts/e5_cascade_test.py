"""E5: the one and only evaluation of the cascade on the test steps 35-49.

SPEC.md sections 6-9, Day 8. Run from the repo root:

    python scripts/e5_cascade_test.py --family rf

What it does
  1. loads the FROZEN Stage 1 artifact (never refits it) and thresholds_<family>.json
  2. checks they agree (family, d, t_clear <= d) and that the split windows are right
  3. trains the final Stage 2 on steps 1-34 (same family, ALL features, same seed)
  4. scores steps 35-49 once: p1 (frozen Stage 1 + calibrator) and p2
  5. evaluates four systems on the same rows:
       local-only (p1 >= d), graph-always (p2 >= d2), the cascade,
       random escalation at the cascade's escalation rate (5 fixed seeds)
  6. writes tables, a cost-vs-recall curve, per-step metrics and the predictions

Nothing here chooses or changes a threshold. d, d2 and t_clear come from the
thresholds file and stay exactly as frozen on Day 7. The cost-vs-recall curve
sweeps t_clear on the test steps for DESCRIPTION only; the frozen t_clear is
marked on it and is never replaced by a value read off the curve.

Outputs (results/)
    e5_summary.csv              one row per system (random control: one row per seed + mean)
    e5_per_step.csv             per-step metrics for every system
    e5_cost_table.csv           simulated relative cost for r = 5, 20, 100
    e5_cost_recall_curve.csv    descriptive curve: cascade vs random vs the two baselines
    e5_run_meta.json            thresholds used, windows, seeds, hash of the thresholds file
    cascade_predictions.csv     txId, time_step, p1, escalated, p2, final_decision, label, ...
    figures/e5_cost_recall.png, figures/e5_per_step_f1.png

The script refuses to overwrite cascade_predictions.csv without --force, so the
test result cannot be quietly regenerated after looking at it.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cascade import (  # noqa: E402
    graph_always,
    load_stage1,
    load_thresholds,
    local_only,
    random_control,
    run_cascade,
    stage1_p1,
    sweep_t_clear,
    t_clear_grid,
)
from src.evaluate import (  # noqa: E402
    cost_table,
    illicit_metrics,
    per_step_metrics,
    recall_retention,
)
from src.load_data import build_dataset  # noqa: E402
from src.models import feature_columns, make_model  # noqa: E402
from src.splits import make_splits  # noqa: E402

RANDOM_SEEDS = (0, 1, 2, 3, 4)
R_VALUES = (5, 20, 100)
SYSTEMS = ("local_only", "graph_always", "cascade", "random_control")
METRIC_COLS = ["precision", "recall", "f1"]


def say(message):
    print(f"[DAY8] {message}")


def p_illicit(model, X):
    """P(illicit) from a fitted model (class 1 column)."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------
# Checks that must pass BEFORE the test steps are scored
# --------------------------------------------------------------------------

def check_frozen_inputs(family, artifact, thresholds):
    """Stage 1 artifact and thresholds file must describe the same frozen setup."""
    if thresholds["family"] != family:
        raise ValueError(f"thresholds file is for {thresholds['family']!r}, expected {family!r}")
    if artifact["family"] != family:
        raise ValueError(f"Stage 1 artifact is for {artifact['family']!r}, expected {family!r}")
    if abs(float(artifact["d"]) - float(thresholds["d"])) > 1e-12:
        raise ValueError(
            f"d differs: artifact {artifact['d']} vs thresholds file {thresholds['d']}. "
            "Thresholds are frozen; do not edit either file."
        )
    if float(thresholds["t_clear"]) > float(thresholds["d"]) + 1e-12:
        raise ValueError("t_clear must be <= d")


def check_windows(cfg, splits):
    """train_final must be exactly steps 1-34 and test exactly 35-49."""
    sp = cfg["splits"]
    for name in ("train_final", "test"):
        lo, hi = (int(v) for v in sp[name])
        steps = set(int(v) for v in splits[name]["time_step"].unique())
        if steps != set(range(lo, hi + 1)):
            raise ValueError(f"{name} must contain exactly steps {lo}-{hi}, got {sorted(steps)}")
    if int(sp["train_final"][1]) >= int(sp["test"][0]):
        raise ValueError("train_final must end before the test window starts")


def check_artifact_windows(artifact, cfg):
    """The frozen Stage 1 must have been built from the configured windows."""
    sp = cfg["splits"]
    want = ([int(v) for v in sp["train_stage1"]], [int(v) for v in sp["calib"]])
    got = (list(artifact["train_steps"]), list(artifact["calib_steps"]))
    if got != want:
        raise ValueError(
            f"Stage 1 artifact was built on {got} but config.yaml says {want}. "
            "Rebuild with `python scripts/e3_calibration.py` (then redo Day 7)."
        )


# --------------------------------------------------------------------------
# Pure evaluation logic (no files, no training): unit-tested with toy data
# --------------------------------------------------------------------------

def predictions_frame(frame, p1, p2, th):
    """One row per test transaction with every system's decision.

    `frame` needs txId, time_step, label. p2 is kept only where the cascade
    escalated (it is NaN elsewhere, as the cascade never reads it there).
    """
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    out = run_cascade(p1, p2, th["t_clear"], th["d2"])
    esc = out["escalated"]
    return pd.DataFrame({
        "txId": frame["txId"].to_numpy(),
        "time_step": frame["time_step"].to_numpy(),
        "p1": p1,
        "escalated": esc.astype(int),
        "p2": np.where(esc, p2, np.nan),
        "final_decision": out["pred"],
        "label": frame["label"].to_numpy(),
        "local_only_decision": local_only(p1, th["d"]),
        "graph_always_decision": graph_always(p2, th["d2"]),
    })


def evaluate_systems(frame, p1, p2, th, seeds=RANDOM_SEEDS):
    """Metrics for the four systems on the same rows.

    Returns (summary, per_step, decisions) where decisions maps system -> array
    (random control: dict seed -> array). PR-AUC is reported where a single score
    exists (local-only: p1, graph-always: p2); the cascade and the random control
    make mixed decisions with no single score, so their PR-AUC is NaN.
    """
    y = frame["label"].to_numpy()
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)

    casc = run_cascade(p1, p2, th["t_clear"], th["d2"])
    rate = casc["escalation_rate"]
    decisions = {
        "local_only": local_only(p1, th["d"]),
        "graph_always": graph_always(p2, th["d2"]),
        "cascade": casc["pred"],
    }
    scores = {"local_only": p1, "graph_always": p2, "cascade": None}
    rand = {s: random_control(p1, p2, th["d"], th["d2"], rate, s) for s in seeds}

    ga_recall = illicit_metrics(y, decisions["graph_always"])["recall"]
    rows = []

    def add(system, seed, pred, score, esc_rate):
        m = illicit_metrics(y, pred, score)
        rows.append({
            "system": system, "seed": seed, "escalation_rate": esc_rate,
            "n": m["n"], "n_illicit": m["n_illicit"], "n_flagged": m["n_flagged"],
            "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            "precision": m["precision"], "recall": m["recall"], "f1": m["f1"],
            "pr_auc": m["pr_auc"],
            "recall_retention": recall_retention(m["recall"], ga_recall),
        })

    add("local_only", "", decisions["local_only"], scores["local_only"], 0.0)
    add("graph_always", "", decisions["graph_always"], scores["graph_always"], 1.0)
    add("cascade", "", decisions["cascade"], None, rate)
    for s in seeds:
        add("random_control", s, rand[s]["pred"], None, rand[s]["escalation_rate"])
    summary = pd.DataFrame(rows)

    mean_row = summary[summary["system"] == "random_control"][
        ["escalation_rate", "precision", "recall", "f1", "recall_retention"]].mean()
    std_row = summary[summary["system"] == "random_control"][
        ["precision", "recall", "f1", "recall_retention"]].std(ddof=1)
    extra = [{"system": "random_control_mean", "seed": "mean", **mean_row.to_dict()},
             {"system": "random_control_std", "seed": "std", **std_row.to_dict()}]
    summary = pd.concat([summary, pd.DataFrame(extra)], ignore_index=True)

    # per-step metrics
    base = frame[["time_step", "label"]].copy()
    parts = []
    for name, pred in decisions.items():
        step = per_step_metrics(base.assign(pred=pred))
        parts.append(step.assign(system=name, seed=""))
    rand_steps = pd.concat(
        [per_step_metrics(base.assign(pred=rand[s]["pred"])).assign(seed=s) for s in seeds])
    r_mean = rand_steps.groupby("time_step", as_index=False)[METRIC_COLS].mean()
    parts.append(r_mean.assign(system="random_control_mean", seed="mean"))
    per_step = pd.concat(parts, ignore_index=True)
    per_step = per_step[["system", "seed", "time_step", "precision", "recall", "f1"]]
    return summary, per_step, decisions


def cost_recall_curve(frame, p1, p2, th, seeds=RANDOM_SEEDS, step=0.01):
    """Descriptive curve on the test steps: cascade vs random at each escalation rate.

    One row per t_clear in 0..d. NOT used to choose anything.
    """
    y = frame["label"].to_numpy()
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    sweep = sweep_t_clear(y, p1, p2, th["d"], th["d2"], t_clear_grid(th["d"], step))
    rand_recall, rand_f1 = [], []
    for rate in sweep["escalation_rate"]:
        ms = [illicit_metrics(y, random_control(p1, p2, th["d"], th["d2"], rate, s)["pred"])
              for s in seeds]
        rand_recall.append(float(np.mean([m["recall"] for m in ms])))
        rand_f1.append(float(np.mean([m["f1"] for m in ms])))
    sweep = sweep.rename(columns={"precision": "cascade_precision", "recall": "cascade_recall",
                                  "f1": "cascade_f1"})
    sweep["random_recall_mean"] = rand_recall
    sweep["random_f1_mean"] = rand_f1
    lo = illicit_metrics(y, local_only(p1, th["d"]))
    ga = illicit_metrics(y, graph_always(p2, th["d2"]))
    sweep["local_only_recall"] = lo["recall"]
    sweep["graph_always_recall"] = ga["recall"]
    sweep["is_frozen_t_clear"] = np.isclose(sweep["t_clear"], th["t_clear"], atol=1e-9)
    return sweep


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------

def plot_cost_recall(curve, th, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    ax.plot(curve["escalation_rate"], curve["cascade_recall"], marker="o", ms=3, lw=2,
            label="Cascade (sweep of t_clear)")
    ax.plot(curve["escalation_rate"], curve["random_recall_mean"], lw=2, ls="--",
            label="Random escalation (mean of 5 seeds)")
    ax.axhline(curve["local_only_recall"].iloc[0], color="grey", lw=1, ls=":",
               label="Local-only")
    ax.axhline(curve["graph_always_recall"].iloc[0], color="black", lw=1, ls=":",
               label="Graph-always")
    frozen = curve[curve["is_frozen_t_clear"]]
    if len(frozen):
        ax.scatter(frozen["escalation_rate"], frozen["cascade_recall"], s=90, zorder=5,
                   facecolors="none", edgecolors="red", linewidths=2,
                   label=f"Frozen t_clear = {th['t_clear']:.2f}")
    ax.set_xlabel("escalation rate (share of transactions sent to Stage 2)")
    ax.set_ylabel("illicit recall on steps 35-49")
    ax.set_title("Cost versus recall on the test steps (descriptive)")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_per_step(per_step, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.4))
    names = {"local_only": "Local-only", "graph_always": "Graph-always",
             "cascade": "Cascade", "random_control_mean": "Random escalation (mean)"}
    for key, label in names.items():
        g = per_step[per_step["system"] == key].sort_values("time_step")
        ax.plot(g["time_step"], g["f1"], marker="o", ms=3, lw=1.8, label=label)
    ax.set_xlabel("time step")
    ax.set_ylabel("illicit F1")
    ax.set_title("Per-step illicit F1 on the test steps")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Runner
# --------------------------------------------------------------------------

def run(family, cfg, splits, cols_all, artifact, th, out_dir, thresholds_path, force=False):
    out_dir = Path(out_dir)
    pred_path = out_dir / "cascade_predictions.csv"
    if pred_path.exists() and not force:
        raise SystemExit(
            f"{pred_path} already exists. The test steps are evaluated once; "
            "pass --force only if you are re-running for a documented reason."
        )
    check_frozen_inputs(family, artifact, th)
    check_artifact_windows(artifact, cfg)
    check_windows(cfg, splits)

    train, test = splits["train_final"], splits["test"]
    seed = int(artifact["seed"])
    say(f"{family}: thresholds d={th['d']:.2f} d2={th['d2']:.2f} t_clear={th['t_clear']:.2f} "
        "(frozen, read from file)")
    say(f"{family}: training the final Stage 2 (ALL features) on steps "
        f"{train.time_step.min()}-{train.time_step.max()}, {len(train)} rows (seed {seed})")
    stage2 = make_model(family, seed).fit(train[cols_all], train["label"].to_numpy())

    say(f"{family}: scoring test steps {test.time_step.min()}-{test.time_step.max()} "
        f"({len(test)} rows) once")
    p1 = stage1_p1(artifact, test)
    p2 = p_illicit(stage2, test[cols_all])

    summary, per_step, _ = evaluate_systems(test, p1, p2, th)
    curve = cost_recall_curve(test, p1, p2, th)
    preds = predictions_frame(test, p1, p2, th)

    rate = float(preds["escalated"].mean())
    costs = cost_table(rate, R_VALUES)
    ga_recall = float(summary.loc[summary["system"] == "graph_always", "recall"].iloc[0])
    casc_recall = float(summary.loc[summary["system"] == "cascade", "recall"].iloc[0])
    rand_recall = float(summary.loc[summary["system"] == "random_control_mean", "recall"].iloc[0])
    costs["cascade_recall"] = casc_recall
    costs["recall_retention"] = recall_retention(casc_recall, ga_recall)
    costs["random_recall_same_cost"] = rand_recall
    costs["note"] = "simulated cost: graph features are precomputed, so cost is modelled"

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)
    summary.assign(family=family).to_csv(out_dir / "e5_summary.csv", index=False)
    per_step.assign(family=family).to_csv(out_dir / "e5_per_step.csv", index=False)
    costs.assign(family=family).to_csv(out_dir / "e5_cost_table.csv", index=False)
    curve.assign(family=family).to_csv(out_dir / "e5_cost_recall_curve.csv", index=False)
    plot_cost_recall(curve, th, out_dir / "figures" / "e5_cost_recall.png")
    plot_per_step(per_step, out_dir / "figures" / "e5_per_step_f1.png")
    meta = {
        "family": family, "seed": seed, "random_seeds": list(RANDOM_SEEDS),
        "thresholds": {k: th[k] for k in ("d", "d2", "t_clear")},
        "thresholds_file": str(thresholds_path), "thresholds_sha256": sha256(thresholds_path),
        "stage1_trained_on_steps": artifact["train_steps"],
        "stage1_calibrated_on_steps": artifact["calib_steps"],
        "stage2_final_trained_on_steps": [int(train.time_step.min()), int(train.time_step.max())],
        "test_steps": [int(test.time_step.min()), int(test.time_step.max())],
        "n_test": int(len(test)), "test_escalation_rate": rate,
        "validation_escalation_rate": th.get("validation_escalation_rate"),
        "note": "d2 was frozen on a validation copy of Stage 2 (steps 1-28); the final "
                "Stage 2 is trained on 1-34 and uses the same d2. Not retuned.",
    }
    (out_dir / "e5_run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    preds.to_csv(pred_path, index=False)

    show = summary[summary["system"].isin(
        ["local_only", "graph_always", "cascade", "random_control_mean"])]
    for r in show.itertuples():
        say(f"{family} {r.system:<20} precision={r.precision:.3f} recall={r.recall:.3f} "
            f"F1={r.f1:.3f} retention={r.recall_retention:.3f}")
    say(f"{family}: test escalation rate {rate:.3f} (validation "
        f"{th.get('validation_escalation_rate', float('nan')):.3f}); "
        "a gap is expected because the illicit share fell from 19.6% to 6.5%. Report it, do not retune.")
    for c in costs.itertuples():
        say(f"{family}: simulated cost r={c.r:g}: cascade {c.cascade_cost:.2f} vs graph-always "
            f"{c.graph_always_cost:.2f} -> savings {c.relative_savings:.1%}")
    say(f"saved results to {out_dir}")
    return {"summary": summary, "per_step": per_step, "curve": curve,
            "costs": costs, "preds": preds}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--family", choices=("rf", "lr"), default="rf")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--thresholds-dir", default=".")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true", help="overwrite the saved test result")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    thresholds_path = Path(args.thresholds_dir) / f"thresholds_{args.family}.json"
    th = load_thresholds(thresholds_path)
    artifact = load_stage1(Path(args.artifacts) / f"stage1_{args.family}.joblib")
    pred_path = Path(args.out) / "cascade_predictions.csv"
    if pred_path.exists() and not args.force:
        raise SystemExit(f"{pred_path} already exists. The test steps are evaluated once; "
                         "pass --force only for a documented re-run.")

    say("loading dataset...")
    df = build_dataset(args.data)
    splits = make_splits(df, cfg)
    cols_all = feature_columns(df, "all")
    run(args.family, cfg, splits, cols_all, artifact, th, args.out, thresholds_path,
        force=args.force)
    say("Done. Paste every [DAY8] line into the chat, then commit the results and tag v2-rq2.")


if __name__ == "__main__":
    main()