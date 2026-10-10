
"""E2: per-time-step drift for RQ1 (LOCAL vs ALL features).

Run from the repository root:
    python scripts/e2_drift.py

Reuses the existing baseline CSVs; does not fit or tune models.
"""

from pathlib import Path


import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.drift import METRICS, plot_drift


RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"

FEATURE_FILES = {
    "LOCAL": RESULTS / "baselines_per_step_local.csv",
    "ALL": RESULTS / "baselines_per_step_all.csv",
}

LOW_SUPPORT = 10
FAMILIES = ("rf", "lr")
COUNT_COLUMNS = ("tp", "fp", "fn", "tn")


def say(message):
    print(f"[DAY5] {message}")


def load_per_step(path, feature_set):
    """Load and validate the existing per-step baseline results."""
    if not path.exists():
        raise FileNotFoundError(
            f"Missing {path}. Run the existing baseline script first."
        )

    df = pd.read_csv(path)
    required = {
        "family", "seed", "time_step", "n", "n_illicit",
        "n_flagged", "tp", "fp", "fn", "tn",
        "precision", "recall", "f1",
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{path.name} is missing columns: {sorted(missing)}")

    df["feature_set"] = feature_set
    df["family"] = df["family"].str.lower()

    if df["time_step"].min() < 35 or df["time_step"].max() > 49:
        raise ValueError("Expected test time steps 35–49 only.")

    return df


def aggregate_feature_set(df):
    """Aggregate per-step metrics across seeds for each model family."""
    summaries = {}
    output_rows = []

    for family in FAMILIES:
        part = df[df["family"] == family].copy()
        if part.empty:
            raise ValueError(f"No rows found for family {family!r}.")

        # Each seed must cover the same test time steps.
        step_sets = [
            set(group["time_step"].astype(int))
            for _, group in part.groupby("seed")
        ]
        if any(steps != step_sets[0] for steps in step_sets[1:]):
            raise ValueError(f"Inconsistent time steps across {family} seeds.")

        grouped = part.groupby("time_step", sort=True)
        summary = grouped.agg(
            n_labelled=("n", "first"),
            n_illicit=("n_illicit", "first"),
        )

        for metric in METRICS:
            summary[f"{metric}_mean"] = grouped[metric].mean()
            summary[f"{metric}_std"] = grouped[metric].std(ddof=0)

        summary.index.name = "time_step"
        summaries[family] = summary

        for step, row in summary.iterrows():
            record = {
                "family": family,
                "feature_set": df["feature_set"].iloc[0],
                "time_step": int(step),
                "n_labelled": int(row["n_labelled"]),
                "n_illicit": int(row["n_illicit"]),
            }
            for metric in METRICS:
                record[f"{metric}_mean"] = row[f"{metric}_mean"]
                record[f"{metric}_std"] = row[f"{metric}_std"]
            output_rows.append(record)

    return summaries, output_rows


def overall_from_counts(df):
    """Calculate pooled P/R/F1 per seed, then summarize across seeds."""
    rows = []

    for (family, seed), group in df.groupby(["family", "seed"]):
        counts = {col: int(group[col].sum()) for col in COUNT_COLUMNS}
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]

        precision = tp / (tp + fp) if tp + fp else np.nan
        recall = tp / (tp + fn) if tp + fn else np.nan
        f1 = (
            2 * precision * recall / (precision + recall)
            if np.isfinite(precision) and np.isfinite(recall)
            and precision + recall > 0
            else 0.0 if precision == 0 and recall == 0 else np.nan
        )

        rows.append({
            "family": family,
            "seed": seed,
            "n_test_rows": int(group["n"].sum()),
            "n_test_illicit": int(group["n_illicit"].sum()),
            "precision": precision,
            "recall": recall,
            "f1": f1,
        })

    per_seed = pd.DataFrame(rows)
    summary_rows = []

    for family, group in per_seed.groupby("family"):
        row = {
            "family": family,
            "feature_set": df["feature_set"].iloc[0],
            "n_seeds": int(group["seed"].nunique()),
            "n_test_rows": int(group["n_test_rows"].iloc[0]),
            "n_test_illicit": int(group["n_test_illicit"].iloc[0]),
        }
        for metric in METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=0)
        summary_rows.append(row)

    return summary_rows


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)

    loaded = {
        name: load_per_step(path, name)
        for name, path in FEATURE_FILES.items()
    }

    all_step_rows = []
    all_overall_rows = []
    summaries = {family: {} for family in FAMILIES}

    for feature_set, df in loaded.items():
        feature_summaries, step_rows = aggregate_feature_set(df)
        all_step_rows.extend(step_rows)
        all_overall_rows.extend(overall_from_counts(df))

        for family in FAMILIES:
            summaries[family][feature_set] = feature_summaries[family]

    per_step = pd.DataFrame(all_step_rows)
    overall = pd.DataFrame(all_overall_rows)

    per_step.to_csv(RESULTS / "e2_per_step.csv", index=False)
    overall.to_csv(RESULTS / "e2_overall.csv", index=False)

    for family in FAMILIES:
        label = {
            "rf": "Random forest",
            "lr": "Logistic regression",
        }[family]

        plot_drift(
            summaries[family],
            f"{label}: illicit-class metrics by test time step "
            "(trained on steps 1–34; threshold 0.5)",
            FIGURES / f"e2_drift_{family}.png",
            low_support=LOW_SUPPORT,
        )
        say(f"Saved results/figures/e2_drift_{family}.png")

    say("Pooled test metrics (means and standard deviations across seeds):")
    print(overall.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

    for family in FAMILIES:
        local = summaries[family]["LOCAL"]
        all_features = summaries[family]["ALL"]
        difference = all_features["f1_mean"] - local["f1_mean"]

        say(
            f"{family}: ALL has higher mean F1 in "
            f"{int((difference > 0).sum())}/{len(difference)} test steps; "
            f"mean step-level F1 difference (ALL - LOCAL) = "
            f"{difference.mean():+.3f}"
        )

        low_support_steps = local.index[
            local["n_illicit"] < LOW_SUPPORT
        ].tolist()
        say(
            f"{family}: steps with fewer than {LOW_SUPPORT} illicit rows: "
            f"{low_support_steps}"
        )

    say("Saved results/e2_per_step.csv and results/e2_overall.csv")


if __name__ == "__main__":
    main()
