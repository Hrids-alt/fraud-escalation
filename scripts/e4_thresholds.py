"""E4 (Day 7): choose and freeze d2 and t_clear for the cascade.

SPEC.md Section 6, step 4-5. Run from the repo root, after Day 6:

    python scripts/e4_thresholds.py                  # random forest (default)
    python scripts/e4_thresholds.py --family lr      # secondary comparison
    python scripts/e4_thresholds.py --family both

Uses ONLY steps 1-28 (train_stage1) and 29-34 (calib). Test steps 35-49 are
never read, and Stage 1 is loaded from artifacts/ and never refitted.

For each family:
  1. load the frozen Stage 1 artifact (model, calibrator, d)
  2. train a VALIDATION copy of Stage 2 (same family, ALL features) on 1-28
  3. score steps 29-34 with both stages (p1 calibrated, p2 raw)
  4. choose d2 = threshold maximising illicit F1 of Stage 2 on steps 29-34
     (same rule as d), then keep it fixed
  5. sweep t_clear from 0 to d on steps 29-34 and choose the largest value
     whose recall retention (cascade recall / graph-always recall) is at
     least the target (default 0.95); if the target cannot be met, report the
     best retention achieved
  6. write the sweep table and the frozen thresholds

Outputs
    results/e4_validation_sweep.csv   one row per (family, t_clear)
    thresholds_<family>.json          d, d2, t_clear, target, whether met

A thresholds file is a FREEZE: the script refuses to overwrite one unless you
pass --force. Do not use --force after you have looked at test results.

Optional config.yaml keys (defaults shown):
    recall_target: 0.95
    threshold_grid: {start: 0.01, stop: 0.99, step: 0.01}
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.calibration import choose_d  # noqa: E402
from src.cascade import (  # noqa: E402
    choose_t_clear,
    graph_always,
    load_stage1,
    local_only,
    save_thresholds,
    stage1_p1,
    sweep_t_clear,
    t_clear_grid,
)
from src.evaluate import illicit_metrics  # noqa: E402
from src.load_data import build_dataset  # noqa: E402
from src.models import feature_columns, make_model  # noqa: E402
from src.splits import make_splits  # noqa: E402

TEST_START = 35


def say(message):
    print(f"[DAY7] {message}")


def threshold_grid(cfg):
    """Decision-threshold grid from config.yaml, validated.

    Requires finite values, 0 < step, 0 <= start <= stop <= 1 and a non-empty grid.
    """
    g = cfg.get("threshold_grid") or {}
    try:
        start, stop, step = (float(g.get(k, d))
                             for k, d in (("start", 0.01), ("stop", 0.99), ("step", 0.01)))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"threshold_grid values must be numbers: {exc}") from exc
    if not all(np.isfinite(v) for v in (start, stop, step)):
        raise ValueError("threshold_grid values must be finite")
    if step <= 0:
        raise ValueError(f"threshold_grid step must be positive, got {step}")
    if not (0.0 <= start <= stop <= 1.0):
        raise ValueError(f"threshold_grid needs 0 <= start <= stop <= 1, got {start}, {stop}")
    grid = np.round(np.arange(start, stop + step / 2, step), 6)
    grid = grid[(grid >= 0.0) & (grid <= 1.0)]
    if grid.size == 0:
        raise ValueError("threshold_grid is empty")
    return grid


def validate_cli_values(target, t_step):
    """Fail early, with a clear message, on a bad --target or --t-step."""
    if not np.isfinite(target) or not (0.0 < target <= 1.0):
        raise ValueError(f"recall-retention target must be in (0, 1], got {target}")
    if not np.isfinite(t_step) or not (0.0 < t_step <= 1.0):
        raise ValueError(f"t_clear grid step must be finite and in (0, 1], got {t_step}")


def windows_from_cfg(cfg):
    """(train_stage1, calib) step windows from config.yaml, validated."""
    try:
        sp = cfg["splits"]
        windows = (tuple(int(v) for v in sp["train_stage1"]), tuple(int(v) for v in sp["calib"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("config.yaml needs splits.train_stage1 and splits.calib "
                         "as [first_step, last_step]") from exc
    if any(len(w) != 2 or w[0] > w[1] for w in windows):
        raise ValueError(f"each window must be [first_step, last_step], got {windows}")
    return windows


def check_window(frame, name, lo, hi):
    """The frame must contain exactly the time steps lo..hi, no gaps and no extras."""
    steps = set(int(v) for v in frame["time_step"].unique())
    expected = set(range(lo, hi + 1))
    if steps != expected:
        missing = sorted(expected - steps)
        extra = sorted(steps - expected)
        raise ValueError(f"{name} must contain exactly steps {lo}-{hi}; missing {missing}, unexpected {extra}")


def p_illicit(model, X):
    """P(illicit) from a fitted model (class 1 column)."""
    return model.predict_proba(X)[:, list(model.classes_).index(1)]


def compute_thresholds(family, artifact, splits, cols_all, grid, t_step, target, seed,
                       windows=((1, 28), (29, 34))):
    """Choose d2 and t_clear on the calibration steps. No file I/O.

    `splits` needs only 'train_stage1' and 'calib'. `windows` gives the exact
    steps each must contain. Returns a dict with the frozen values, the sweep
    table and validation reference metrics.
    """
    validate_cli_values(target, t_step)
    train, calib = splits["train_stage1"], splits["calib"]
    (tr_lo, tr_hi), (ca_lo, ca_hi) = windows
    if tr_hi >= ca_lo or ca_hi >= TEST_START:
        raise ValueError("windows must be ordered: train < calibration < test (step 35)")
    check_window(train, "train_stage1", tr_lo, tr_hi)
    check_window(calib, "calib", ca_lo, ca_hi)
    if artifact["family"] != family:
        raise ValueError(f"artifact family is {artifact['family']!r}, expected {family!r}")
    windows = ([int(train["time_step"].min()), int(train["time_step"].max())],
               [int(calib["time_step"].min()), int(calib["time_step"].max())])
    if (artifact["train_steps"], artifact["calib_steps"]) != windows:
        raise ValueError(
            "the Stage 1 artifact was built from different steps than the current "
            "splits. Rebuild it with `python scripts/e3_calibration.py`."
        )

    y_cal = calib["label"].to_numpy()
    d = float(artifact["d"])
    p1 = stage1_p1(artifact, calib)

    stage2 = make_model(family, seed).fit(train[cols_all], train["label"].to_numpy())
    p2 = p_illicit(stage2, calib[cols_all])

    d2, _ = choose_d(y_cal, p2, grid)
    d2 = float(d2)

    sweep = sweep_t_clear(y_cal, p1, p2, d, d2, t_clear_grid(d, t_step))
    t_clear, row, met = choose_t_clear(sweep, target)

    return {
        "d": d,
        "d2": d2,
        "t_clear": t_clear,
        "target": float(target),
        "target_met": bool(met),
        "chosen_row": row,
        "sweep": sweep,
        "local_only": illicit_metrics(y_cal, local_only(p1, d)),
        "graph_always": illicit_metrics(y_cal, graph_always(p2, d2)),
        "n_calib": int(len(y_cal)),
    }


def run_family(family, cfg, splits, cols_all, grid, args):
    art_path = Path(args.artifacts) / f"stage1_{family}.joblib"
    out_path = Path(args.thresholds_dir) / f"thresholds_{family}.json"
    if out_path.exists() and not args.force:
        raise SystemExit(
            f"{out_path} already exists. Thresholds are frozen; pass --force only if "
            "you are redoing Day 7 before any test evaluation."
        )

    say(f"--- {family}: loading frozen Stage 1 from {art_path} ---")
    artifact = load_stage1(art_path)
    seed = int(artifact["seed"])
    if seed != int(cfg["seed"]):
        say(f"note: artifact seed {seed} differs from config seed {cfg['seed']}; using the artifact seed")

    say(f"{family}: training the validation copy of Stage 2 (ALL features) on steps 1-28...")
    windows = windows_from_cfg(cfg)
    res = compute_thresholds(family, artifact, splits, cols_all, grid, args.t_step,
                             args.target, seed, windows=windows)

    sweep = res["sweep"]
    show = sweep[(sweep["t_clear"] * 100).round().astype(int) % 5 == 0]
    say(f"{family}: validation sweep on steps 29-34 (every 0.05):")
    for r in show.itertuples():
        say(f"{family} t_clear={r.t_clear:.2f} escalation={r.escalation_rate:.3f} "
            f"precision={r.precision:.3f} recall={r.recall:.3f} F1={r.f1:.3f} "
            f"retention={r.recall_retention:.3f}")

    lo, ga = res["local_only"], res["graph_always"]
    say(f"{family}: validation reference, local-only at d={res['d']:.2f}: "
        f"precision={lo['precision']:.3f} recall={lo['recall']:.3f} F1={lo['f1']:.3f}")
    say(f"{family}: validation reference, graph-always at d2={res['d2']:.2f}: "
        f"precision={ga['precision']:.3f} recall={ga['recall']:.3f} F1={ga['f1']:.3f}")

    row = res["chosen_row"]
    say(f"{family}: d = {res['d']:.2f} (frozen on Day 6), d2 = {res['d2']:.2f}")
    say(f"{family}: t_clear = {res['t_clear']:.2f} "
        f"(escalation {row['escalation_rate']:.3f}, retention {row['recall_retention']:.3f}, "
        f"target {res['target']:.2f} {'met' if res['target_met'] else 'NOT met: best achievable'})")
    if res["t_clear"] == 0.0:
        say(f"{family}: WARNING t_clear = 0 means everything is escalated, so the "
            "cascade saves nothing on the validation steps. Report this honestly.")

    extra = {
        "stage2_validation_trained_on_steps": [1, 28],
        "d2_rule": "max illicit F1 of Stage 2 on steps 29-34 (validation copy)",
        "validation_escalation_rate": float(row["escalation_rate"]),
        "validation_recall_retention": float(row["recall_retention"]),
        "n_calib": res["n_calib"],
    }
    save_thresholds(out_path, family, res["d"], res["d2"], res["t_clear"],
                    res["target"], res["target_met"], extra=extra)
    say(f"{family}: saved {out_path}")
    return sweep.assign(family=family)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--family", choices=("rf", "lr", "both"), default="rf")
    ap.add_argument("--target", type=float, default=None,
                    help="recall-retention target; defaults to recall_target in config.yaml or 0.95")
    ap.add_argument("--t-step", type=float, default=0.01, help="t_clear grid step")
    ap.add_argument("--out", default="results")
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--thresholds-dir", default=".")
    ap.add_argument("--force", action="store_true", help="overwrite frozen thresholds")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if args.target is None:
        args.target = float(cfg.get("recall_target", 0.95))
    validate_cli_values(args.target, args.t_step)
    grid = threshold_grid(cfg)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    say("loading dataset...")
    df = build_dataset(args.data)
    splits = make_splits(df, cfg)
    cols_all = feature_columns(df, "all")
    stage_splits = {"train_stage1": splits["train_stage1"], "calib": splits["calib"]}
    for name, s in stage_splits.items():
        say(f"{name}: steps {s.time_step.min()}-{s.time_step.max()}, {len(s)} rows, "
            f"illicit {int((s.label == 1).sum())}")
    say("test steps are not used in this script")

    families = ("rf", "lr") if args.family == "both" else (args.family,)
    tables = [run_family(f, cfg, stage_splits, cols_all, grid, args) for f in families]

    sweep_path = out_dir / "e4_validation_sweep.csv"
    new = pd.concat(tables, ignore_index=True)
    if sweep_path.exists():
        old = pd.read_csv(sweep_path)
        old = old[~old["family"].isin(new["family"].unique())]
        new = pd.concat([old, new], ignore_index=True)
    new.to_csv(sweep_path, index=False)
    say(f"saved {sweep_path}")
    say("Done. Paste every [DAY7] line into the chat, then commit thresholds_*.json "
        "and results/e4_validation_sweep.csv.")


if __name__ == "__main__":
    main()