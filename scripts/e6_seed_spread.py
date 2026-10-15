"""E6: seed spread of the final Stage 2 on the test steps 35-49 (Day 9).

SPEC.md sections 6-9. Run from the repo root, after Day 8:

    python scripts/e6_seed_spread.py --family rf

Question answered: how much do the Day 8 numbers move when only the random seed
of the final Stage 2 changes?

What it does
  1. loads the FROZEN Stage 1 artifact and thresholds_<family>.json (never refits Stage 1)
  2. runs the same input checks as Day 8
  3. for each seed (default 42, 43, 44): trains the final Stage 2 (ALL features) on
     steps 1-34, scores steps 35-49, and evaluates the same four systems as Day 8
  4. writes one CSV with a row per (seed, system) plus mean and std rows

What it does NOT do
  * It does not change d, d2 or t_clear. They stay exactly as frozen on Day 7.
  * It does not choose or tune anything from the test results.
  * It does not touch any Day 8 file (cascade_predictions.csv, e5_*.csv, e5_run_meta.json).
  * Stage 1 and its calibrator are the same for every seed, so only Stage 2 varies.

Output (results/)
    e6_seed_spread.csv

The script refuses to overwrite e6_seed_spread.csv without --force.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_e5():
    spec = importlib.util.spec_from_file_location(
        "e5_cascade_test", ROOT / "scripts" / "e5_cascade_test.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e5 = _load_e5()

from src.cascade import load_stage1, load_thresholds, stage1_p1  # noqa: E402
from src.load_data import build_dataset  # noqa: E402
from src.models import feature_columns, make_model  # noqa: E402
from src.splits import make_splits  # noqa: E402

DEFAULT_SEEDS = (42, 43, 44)
SYSTEMS = ("local_only", "graph_always", "cascade", "random_control_mean")
VALUE_COLS = ["escalation_rate", "precision", "recall", "f1", "recall_retention"]
OUT_NAME = "e6_seed_spread.csv"


def say(message):
    print(f"[DAY9] {message}")


def parse_seeds(text):
    """'42,43,44' -> (42, 43, 44). At least two distinct integers."""
    try:
        seeds = tuple(int(part) for part in str(text).split(",") if part.strip() != "")
    except ValueError as exc:
        raise ValueError(f"--seeds must be comma-separated integers, got {text!r}") from exc
    if len(set(seeds)) != len(seeds):
        raise ValueError(f"--seeds must be distinct, got {seeds}")
    if len(seeds) < 2:
        raise ValueError("give at least two seeds to measure a spread")
    return seeds


def spread_table(per_seed):
    """per_seed: DataFrame with columns seed, system and VALUE_COLS.

    Returns the same rows plus, for every system, a 'mean' and a 'std' row
    (sample std, ddof=1) across seeds.
    """
    rows = [per_seed]
    for system, g in per_seed.groupby("system", sort=False):
        mean = g[VALUE_COLS].mean()
        std = g[VALUE_COLS].std(ddof=1)
        rows.append(pd.DataFrame([{"seed": "mean", "system": system, **mean.to_dict()},
                                  {"seed": "std", "system": system, **std.to_dict()}]))
    return pd.concat(rows, ignore_index=True)


def run(family, cfg, splits, cols_all, artifact, th, out_dir, seeds=DEFAULT_SEEDS,
        force=False):
    out_dir = Path(out_dir)
    out_path = out_dir / OUT_NAME
    if out_path.exists() and not force:
        raise SystemExit(f"{out_path} already exists; pass --force to rebuild it.")

    # Same input checks as Day 8, before any test row is scored.
    e5.check_frozen_inputs(family, artifact, th)
    e5.check_artifact_windows(artifact, cfg)
    e5.check_windows(cfg, splits)

    train, test = splits["train_final"], splits["test"]
    say(f"{family}: thresholds d={th['d']:.2f} d2={th['d2']:.2f} t_clear={th['t_clear']:.2f} "
        "(frozen, unchanged)")
    p1 = stage1_p1(artifact, test)          # frozen Stage 1: identical for every seed

    rows = []
    for seed in seeds:
        say(f"{family}: seed {seed}: training final Stage 2 on steps "
            f"{train.time_step.min()}-{train.time_step.max()}")
        stage2 = make_model(family, int(seed)).fit(train[cols_all], train["label"].to_numpy())
        p2 = e5.p_illicit(stage2, test[cols_all])
        summary, _, _ = e5.evaluate_systems(test, p1, p2, th)
        keep = summary[summary["system"].isin(SYSTEMS)]
        for r in keep.itertuples():
            rows.append({"seed": int(seed), "system": r.system,
                         "escalation_rate": r.escalation_rate, "precision": r.precision,
                         "recall": r.recall, "f1": r.f1,
                         "recall_retention": r.recall_retention})
    per_seed = pd.DataFrame(rows)
    table = spread_table(per_seed)
    table.insert(0, "family", family)

    out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_path, index=False)

    for system in SYSTEMS:
        m = table[(table["system"] == system) & (table["seed"] == "mean")].iloc[0]
        s = table[(table["system"] == system) & (table["seed"] == "std")].iloc[0]
        say(f"{family} {system:<20} F1 {m['f1']:.3f} +/- {s['f1']:.3f} | recall "
            f"{m['recall']:.3f} +/- {s['recall']:.3f} | precision {m['precision']:.3f} "
            f"+/- {s['precision']:.3f}")
    return table


def check_against_day8(table, results_dir, seed):
    """If the Day 8 seed is among the seeds, its cascade F1 should match e5_summary.csv."""
    path = Path(results_dir) / "e5_summary.csv"
    if not path.exists():
        return None
    day8 = pd.read_csv(path)
    day8 = day8[day8["system"] == "cascade"]
    mine = table[(table["system"] == "cascade") & (table["seed"].astype(str) == str(seed))]
    if day8.empty or mine.empty:
        return None
    diff = abs(float(day8["f1"].iloc[0]) - float(mine["f1"].iloc[0]))
    return diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--family", choices=("rf", "lr"), default="rf")
    ap.add_argument("--seeds", default=",".join(str(s) for s in DEFAULT_SEEDS))
    ap.add_argument("--artifacts", default="artifacts")
    ap.add_argument("--thresholds-dir", default=".")
    ap.add_argument("--out", default="results")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    th = load_thresholds(Path(args.thresholds_dir) / f"thresholds_{args.family}.json")
    artifact = load_stage1(Path(args.artifacts) / f"stage1_{args.family}.joblib")
    out_path = Path(args.out) / OUT_NAME
    if out_path.exists() and not args.force:
        raise SystemExit(f"{out_path} already exists; pass --force to rebuild it.")

    say("loading dataset...")
    df = build_dataset(args.data)
    splits = make_splits(df, cfg)
    cols_all = feature_columns(df, "all")
    table = run(args.family, cfg, splits, cols_all, artifact, th, args.out, seeds,
                force=args.force)

    diff = check_against_day8(table, args.out, int(artifact["seed"]))
    if diff is not None:
        if diff < 1e-6:
            say(f"seed {artifact['seed']} reproduces the Day 8 cascade F1 (difference {diff:.2e}).")
        else:
            say(f"WARNING seed {artifact['seed']} cascade F1 differs from Day 8 by {diff:.4f}. "
                "Check library versions or n_jobs before quoting both.")
    say(f"saved {out_path}. Thresholds were not changed. Paste every [DAY9] line into the chat.")


if __name__ == "__main__":
    main()