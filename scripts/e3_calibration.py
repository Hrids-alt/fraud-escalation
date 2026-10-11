"""E3: train Stage 1, calibrate it, choose the standalone threshold d.

SPEC.md Section 6, steps 1-3. Run from the repo root:

    python scripts/e3_calibration.py                 # both families
    python scripts/e3_calibration.py --family rf

Uses ONLY steps 1-28 (train_stage1) and 29-34 (calib). The test steps are
never scored or evaluated here, and nothing is retrained afterwards.

For each family (Stage 1 = LOCAL features):
  1. fit on steps 1-28
  2. fit a sigmoid (Platt) calibrator on steps 29-34
  3. report Brier score before/after and a reliability diagram
  4. choose d = threshold maximising illicit F1 on steps 29-34 (calibrated p1)
  5. save the frozen Stage 1 model, calibrator and d (artifacts/), plus
     human-readable results (results/)

Outputs
    artifacts/stage1_<family>.joblib   model + calibrator + d  (do NOT commit)
    results/stage1_<family>.json       calibrator, d, headline numbers
    results/e3_calibration.csv         one row per family
    results/e3_reliability.csv         reliability points before/after
    results/figures/e3_reliability_<family>.png

Optional config.yaml key (defaults shown):
    threshold_grid: {start: 0.01, stop: 0.99, step: 0.01}
"""

import argparse
import json
import sys
from pathlib import Path
import sklearn

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import (  # noqa: E402
    brier,
    calibrate_sigmoid,
    choose_d,
    fit_sigmoid,
    predict_p1,
    reliability_data,
)
from src.drift import INK, INK_2, INK_MUTED, SURFACE, GRID  # noqa: E402
from src.evaluate import illicit_metrics  # noqa: E402
from src.load_data import build_dataset  # noqa: E402
from src.models import feature_columns, make_model  # noqa: E402
from src.splits import make_splits  # noqa: E402

FAMILY_LABEL = {"rf": "Random forest", "lr": "Logistic regression"}
BLUE = "#2a78d6"


def say(message):
    print(f"[DAY6] {message}")


def threshold_grid(cfg):
    g = cfg.get("threshold_grid") or {}
    start, stop, step = (float(g.get(k, d))
                         for k, d in (("start", 0.01), ("stop", 0.99), ("step", 0.01)))
    return np.round(np.arange(start, stop + step / 2, step), 6)


def plot_reliability(before, after, brier_before, brier_after, title, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.8, 5.8), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.plot([0, 1], [0, 1], color="#c3c2b7", lw=1.5, label="Perfect calibration")
    for frame, colour, label in (
        (before, INK_MUTED, f"Before (Brier {brier_before:.3f})"),
        (after, BLUE, f"After sigmoid (Brier {brier_after:.3f})"),
    ):
        ax.plot(frame["mean_predicted"], frame["observed_rate"], color=colour,
                lw=2, marker="o", ms=6, markeredgecolor=SURFACE,
                markeredgewidth=1.2, label=label)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.set_xlabel("mean predicted probability of illicit", color=INK_2)
    ax.set_ylabel("observed share illicit", color=INK_2)
    ax.grid(color=GRID, lw=1)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(length=0, colors=INK_MUTED)
    ax.legend(loc="upper left", frameon=False, labelcolor=INK_2, fontsize=10)
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, color=INK, y=0.99)
    fig.text(0.01, 0.005, "Equal-count bins on steps 29-34 (the calibration "
             "steps, so 'after' is in-sample).", fontsize=8, color=INK_MUTED)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)


def run_family(family, seed, splits, cols, grid, n_bins, out_dir, art_dir):
    train, calib = splits["train_stage1"], splits["calib"]
    y_train = train["label"].to_numpy()
    y_cal = calib["label"].to_numpy()

    say(f"--- {family}: fitting Stage 1 on steps "
        f"{train.time_step.min()}-{train.time_step.max()} (seed {seed}) ---")
    model = make_model(family, seed)
    model.fit(train[cols], y_train)

    p_raw = predict_p1(model, None, calib[cols])
    calibrator = calibrate_sigmoid(model, calib[cols], y_cal)
    p_cal = calibrator.transform(p_raw)

    prevalence = float(y_cal.mean())
    b_raw, b_cal = brier(y_cal, p_raw), brier(y_cal, p_cal)
    b_ref = prevalence * (1 - prevalence)
    say(f"{family}: calibrator a={calibrator.a:.4f} b={calibrator.b:.4f} "
        "(a=1, b=0 would mean already calibrated)")
    say(f"{family}: calib steps illicit share = {prevalence:.4f} "
        f"| train_stage1 illicit share = {float(y_train.mean()):.4f}")
    say(f"{family}: mean predicted P(illicit) raw={p_raw.mean():.4f} "
        f"calibrated={p_cal.mean():.4f} observed={prevalence:.4f}")
    say(f"{family}: Brier raw={b_raw:.4f} calibrated={b_cal:.4f} "
        f"| reference (always predict the base rate)={b_ref:.4f}")

    # Out-of-sample look inside the calibration steps (no test data involved).
    steps = np.sort(calib["time_step"].unique())
    half = len(steps) // 2
    for fit_steps, eval_steps in ((steps[:half], steps[half:]),
                                  (steps[half:], steps[:half])):
        m_fit = calib["time_step"].isin(fit_steps).to_numpy()
        cal = fit_sigmoid(p_raw[m_fit], y_cal[m_fit])
        raw_b = brier(y_cal[~m_fit], p_raw[~m_fit])
        cal_b = brier(y_cal[~m_fit], cal.transform(p_raw[~m_fit]))
        say(f"{family}: fit on steps {fit_steps.min()}-{fit_steps.max()}, "
            f"evaluate on {eval_steps.min()}-{eval_steps.max()}: "
            f"Brier raw={raw_b:.4f} -> calibrated={cal_b:.4f}")

    ap_raw = float(average_precision_score(y_cal, p_raw))
    ap_cal = float(average_precision_score(y_cal, p_cal))
    say(f"{family}: PR-AUC on calib steps raw={ap_raw:.4f} calibrated={ap_cal:.4f} "
        "(must be equal: calibration keeps the ranking)")

    rel_before = reliability_data(y_cal, p_raw, n_bins)
    rel_after = reliability_data(y_cal, p_cal, n_bins)
    say(f"{family}: reliability (bin, n, mean_pred_before, observed_before, "
        "mean_pred_after, observed_after):")
    for b, a in zip(rel_before.itertuples(), rel_after.itertuples()):
        say(f"{family} {b.bin:>2} {b.n:>5} {b.mean_predicted:.3f} "
            f"{b.observed_rate:.3f} {a.mean_predicted:.3f} {a.observed_rate:.3f}")

    d, table = choose_d(y_cal, p_cal, grid)
    at_d = illicit_metrics(y_cal, (p_cal >= d).astype(int), p_cal)
    say(f"{family}: d = {d:.2f} (max illicit F1 on steps 29-34): "
        f"precision={at_d['precision']:.3f} recall={at_d['recall']:.3f} "
        f"F1={at_d['f1']:.3f} flagged={at_d['n_flagged']}/{at_d['n']}")

    plot_reliability(
        rel_before, rel_after, b_raw, b_cal,
        f"{FAMILY_LABEL[family]}: Stage 1 (local features) reliability",
        out_dir / "figures" / f"e3_reliability_{family}.png")
    say(f"saved results/figures/e3_reliability_{family}.png")

    art_dir.mkdir(parents=True, exist_ok=True)
    
    
    artifact = {
        "family": family,
        "seed": seed,
        "feature_columns": cols,
        "model": model,
        "calibrator": calibrator.as_dict(),
        "d": d,
        "train_steps": [
             int(train.time_step.min()),
             int(train.time_step.max()),
        ],
        "calib_steps": [
             int(calib.time_step.min()),
             int(calib.time_step.max()),
        ],
        "metadata": {
            "training_window": "steps 1-28",
            "calibration_window": "steps 29-34",
            "brier_raw": b_raw,
            "brier_calibrated": b_cal,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
        },
    }

    joblib.dump(
         artifact,
        art_dir / f"stage1_{family}.joblib",
    )


    row = {
        "family": family, "seed": seed, "n_calib": len(y_cal),
        "calib_illicit_share": prevalence,
        "train_illicit_share": float(y_train.mean()),
        "mean_p_raw": float(p_raw.mean()), "mean_p_cal": float(p_cal.mean()),
        "brier_raw": b_raw, "brier_cal": b_cal, "brier_reference": b_ref,
        "a": calibrator.a, "b": calibrator.b, "d": d,
        "precision_at_d": at_d["precision"], "recall_at_d": at_d["recall"],
        "f1_at_d": at_d["f1"], "pr_auc_raw": ap_raw, "pr_auc_cal": ap_cal,
    }
    (out_dir / f"stage1_{family}.json").write_text(
        json.dumps(row, indent=2), encoding="utf-8")

    rel = pd.concat([
        rel_before.assign(stage="before", family=family),
        rel_after.assign(stage="after", family=family),
    ])
    return row, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--family", choices=("lr", "rf", "both"), default="both")
    ap.add_argument("--seed", type=int, default=None,
                    help="defaults to 'seed' in config.yaml")
    ap.add_argument("--n-bins", type=int, default=10)
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed = int(cfg["seed"]) if args.seed is None else args.seed
    grid = threshold_grid(cfg)
    out_dir, art_dir = Path(args.out), Path(args.artifacts)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    say("loading dataset...")
    df = build_dataset(args.data)
    splits = make_splits(df, cfg)
    assert splits["calib"]["time_step"].max() < splits["test"]["time_step"].min()
    cols = feature_columns(df, "local")
    say(f"Stage 1 features: LOCAL ({len(cols)} columns); threshold grid "
        f"{grid.min():.2f}-{grid.max():.2f} ({len(grid)} values)")
    for name in ("train_stage1", "calib"):
        s = splits[name]
        say(f"{name}: steps {s.time_step.min()}-{s.time_step.max()}, "
            f"{len(s)} rows, illicit {int((s.label == 1).sum())}")

    families = ("rf", "lr") if args.family == "both" else (args.family,)
    rows, rels = [], []
    for family in families:
        row, rel = run_family(family, seed, splits, cols, grid, args.n_bins,
                              out_dir, art_dir)
        rows.append(row)
        rels.append(rel)

    pd.DataFrame(rows).to_csv(out_dir / "e3_calibration.csv", index=False)
    pd.concat(rels, ignore_index=True).to_csv(
        out_dir / "e3_reliability.csv", index=False)
    say("saved results/e3_calibration.csv and results/e3_reliability.csv")
    say("Done. Paste every [DAY6] line into the chat.")


if __name__ == "__main__":
    main()