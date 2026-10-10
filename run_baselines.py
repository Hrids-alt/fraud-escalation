"""RQ1 baselines: local-only (Day 3) and all-features (Day 4).

Trains on steps 1-34 (train_final), evaluates once on steps 35-49 (test).
Both families (lr, rf) are run on the SAME feature set per invocation, so the
local vs all comparison is made within each family (SPEC.md section 5).

Run from the repo root:
    python run_baselines.py --features local
    python run_baselines.py --features all

Outputs (in results/):
    baselines_<features>.csv           one row per model/seed (overall test metrics)
    baselines_per_step_<features>.csv  per-time-step metrics on the test steps

Decision threshold: 0.5 on the model's probability, for the standalone RQ1
baselines only. PR-AUC is threshold-free. (The cascade uses its own frozen
threshold d chosen on the calibration steps; that comes later.)
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from src.evaluate import illicit_metrics, per_step_metrics
from src.load_data import build_dataset
from src.models import FEATURE_SETS, feature_columns, make_model
from src.splits import make_splits

THRESHOLD = 0.5
SUMMARY_COLS = ["precision", "recall", "f1", "pr_auc"]


def score_and_evaluate(model, test_df, X_test):
    score = model.predict_proba(X_test)[:, 1]
    pred = (score >= THRESHOLD).astype(int)
    overall = illicit_metrics(test_df["label"], pred, score)
    frame = test_df[["time_step", "label"]].copy()
    frame["pred"] = pred
    frame["score"] = score
    per_step = per_step_metrics(frame, score_col="score")
    return overall, per_step


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--features", choices=FEATURE_SETS, required=True)
    ap.add_argument("--rf-seeds", type=int, default=3)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    with open(args.config, encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    base_seed = int(cfg["seed"])
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 200)

    print("Loading dataset...")
    df = build_dataset(args.data)
    splits = make_splits(df, cfg)
    train, test = splits["train_final"], splits["test"]
    cols = feature_columns(df, args.features)
    print(f"feature set: {args.features} ({len(cols)} columns)")
    print(f"train_final: {len(train):,} rows, illicit {int((train['label'] == 1).sum()):,}")
    print(f"test:        {len(test):,} rows, illicit {int((test['label'] == 1).sum()):,}")

    X_train, y_train = train[cols], train["label"].to_numpy()
    X_test, y_test = test[cols], test["label"].to_numpy()

    rows, step_frames = [], []
    for family in ("lr", "rf"):
        seeds = [base_seed] if family == "lr" else [base_seed + i for i in range(args.rf_seeds)]
        for seed in seeds:
            print(f"fitting {family} (seed {seed})...")
            model = make_model(family, seed)
            model.fit(X_train, y_train)
            overall, per_step = score_and_evaluate(model, test, X_test)
            rows.append({"features": args.features, "family": family, "seed": seed, **overall})
            per_step.insert(0, "seed", seed)
            per_step.insert(0, "family", family)
            per_step.insert(0, "features", args.features)
            step_frames.append(per_step)

    results = pd.DataFrame(rows)
    results.to_csv(out_dir / f"baselines_{args.features}.csv", index=False)
    pd.concat(step_frames, ignore_index=True).to_csv(
        out_dir / f"baselines_per_step_{args.features}.csv", index=False
    )

    print(f"\n=== TEST RESULTS, {args.features.upper()} features (illicit class, mean over seeds) ===")
    summary = results.groupby("family")[SUMMARY_COLS].mean().round(3)
    spread = results.groupby("family")[SUMMARY_COLS].std().round(3)
    print(summary.to_string())
    if (results.groupby("family").size() > 1).any():
        print("\nstd over seeds (random forest only has more than one seed):")
        print(spread.to_string())

    print("\n=== SANITY CHECKS ===")
    prevalence = float((y_test == 1).mean())
    allgood = illicit_metrics(y_test, np.zeros_like(y_test))
    accuracy = (allgood["tp"] + allgood["tn"]) / allgood["n"]
    print(
        f"everything-licit baseline: accuracy {accuracy:.3f}, recall {allgood['recall']:.3f} "
        f"(shows why accuracy is useless)"
    )

    rng = np.random.default_rng(base_seed)
    y_shuffled = rng.permutation(y_train)
    for family in ("lr", "rf"):
        model = make_model(family, base_seed)
        model.fit(X_train, y_shuffled)
        overall, _ = score_and_evaluate(model, test, X_test)
        print(
            f"shuffled-label {family}: F1 {overall['f1']:.3f}, PR-AUC {overall['pr_auc']:.3f} "
            f"(should be near chance; test illicit share = {prevalence:.3f})"
        )

    print(f"\nsaved: {out_dir / f'baselines_{args.features}.csv'}")
    print(f"saved: {out_dir / f'baselines_per_step_{args.features}.csv'}")
    print("Done. Paste this whole output into the chat.")


if __name__ == "__main__":
    main()