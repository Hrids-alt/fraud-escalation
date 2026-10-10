"""Per-time-step drift metrics and plots (guide E2, RQ1).

Pure functions. Illicit (label 1) is the positive class. Nothing here
trains a model or touches thresholds: it only slices predictions that
already exist by time_step.

Names are deliberately different from evaluate.per_step_metrics so the
two cannot be confused.

    step_metrics_table(time_step, y_true, y_pred) -> DataFrame (one row per step)
    pooled_metrics(table)                         -> dict (precision, recall, f1)
    aggregate_seeds(tables)                       -> DataFrame (mean/std over seeds)
    plot_drift(summary_by_feature_set, title, out_path)
"""

import warnings

import numpy as np
import pandas as pd

METRICS = ("precision", "recall", "f1")

# Reference palette, categorical slots 1 and 2 (validated adjacent pair).
COLOURS = {"LOCAL": "#2a78d6", "ALL": "#eb6834"}
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"


def _prf(tp, fp, fn):
    """Precision, recall, F1 from counts; NaN when undefined.

    precision is NaN if nothing was flagged (tp + fp == 0).
    recall and F1 are NaN if the step has no illicit rows (tp + fn == 0).
    """
    tp, fp, fn = (np.asarray(x, dtype=float) for x in (tp, fp, fn))
    flagged = tp + fp
    illicit = tp + fn
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where(flagged > 0, tp / flagged, np.nan)
        recall = np.where(illicit > 0, tp / illicit, np.nan)
        f1 = np.where(illicit > 0, 2 * tp / (2 * tp + fp + fn), np.nan)
    return precision, recall, f1


def step_metrics_table(time_step, y_true, y_pred):
    """Illicit-class precision / recall / F1 for every time step.

    y_true and y_pred must be 0/1 (labelled nodes only; -1 is rejected).
    Returns a DataFrame indexed by time_step with columns
    n_labelled, n_illicit, n_flagged, tp, fp, fn, precision, recall, f1.
    """
    df = pd.DataFrame(
        {
            "time_step": np.asarray(time_step),
            "y": np.asarray(y_true),
            "p": np.asarray(y_pred),
        }
    )
    if not (len(df) > 0):
        raise ValueError("empty input")
    if not df["y"].isin([0, 1]).all():
        raise ValueError("y_true must contain only 0/1 (labelled nodes only)")
    if not df["p"].isin([0, 1]).all():
        raise ValueError("y_pred must contain only 0/1")

    df["tp"] = ((df["y"] == 1) & (df["p"] == 1)).astype(int)
    df["fp"] = ((df["y"] == 0) & (df["p"] == 1)).astype(int)
    df["fn"] = ((df["y"] == 1) & (df["p"] == 0)).astype(int)
    df["n_labelled"] = 1
    df["n_illicit"] = (df["y"] == 1).astype(int)
    df["n_flagged"] = (df["p"] == 1).astype(int)

    cols = ["n_labelled", "n_illicit", "n_flagged", "tp", "fp", "fn"]
    out = df.groupby("time_step")[cols].sum()
    out["precision"], out["recall"], out["f1"] = _prf(
        out["tp"], out["fp"], out["fn"]
    )
    return out


def pooled_metrics(table):
    """Metrics over all steps in the table, from summed counts."""
    tp, fp, fn = (int(table[c].sum()) for c in ("tp", "fp", "fn"))
    p, r, f = _prf(tp, fp, fn)
    return {
        "precision": float(p),
        "recall": float(r),
        "f1": float(f),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def aggregate_seeds(tables):
    """Mean and std (ddof=0) over seeds for each step and metric.

    All tables must cover the same time steps. With one table, std is 0.
    """
    if not tables:
        raise ValueError("no tables to aggregate")
    index = tables[0].index
    for t in tables[1:]:
        if not t.index.equals(index):
            raise ValueError("tables cover different time steps")

    out = pd.DataFrame(index=index)
    out["n_illicit"] = tables[0]["n_illicit"]
    out["n_labelled"] = tables[0]["n_labelled"]
    for m in METRICS:
        stack = np.vstack([t[m].to_numpy(dtype=float) for t in tables])
        # A step whose metric is undefined for every seed stays NaN.
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            out[f"{m}_mean"] = np.nanmean(stack, axis=0)
            out[f"{m}_std"] = np.nanstd(stack, axis=0)
    out["n_seeds"] = len(tables)
    return out


def plot_drift(summary, title, out_path, low_support=10):
    """Per-step precision / recall / F1 for LOCAL vs ALL, plus the number
    of illicit rows per step.

    summary: {"LOCAL": DataFrame, "ALL": DataFrame} from aggregate_seeds.
    Steps with fewer than `low_support` illicit rows are marked on the
    bottom panel because their metrics are very noisy.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in ("LOCAL", "ALL") if n in summary]
    if not names:
        raise ValueError("summary must contain LOCAL and/or ALL")

    plt.rcParams.update(
        {
            "text.color": INK,
            "axes.labelcolor": INK_2,
            "xtick.color": INK_MUTED,
            "ytick.color": INK_MUTED,
        }
    )

    fig = plt.figure(figsize=(12, 6.4), facecolor=SURFACE)
    grid = fig.add_gridspec(2, 3, height_ratios=[3, 1], hspace=0.28, wspace=0.12)
    top_axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    bar_axes = [
        fig.add_subplot(grid[1, i], sharex=top_axes[i]) for i in range(3)
    ]

    for ax in top_axes + bar_axes:
        ax.set_facecolor(SURFACE)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color("#c3c2b7")
        ax.grid(axis="y", color=GRID, lw=1)
        ax.set_axisbelow(True)
        ax.tick_params(length=0)

    for ax, metric in zip(top_axes, METRICS):
        for name in names:
            s = summary[name]
            x = s.index.to_numpy()
            y = s[f"{metric}_mean"].to_numpy()
            sd = s[f"{metric}_std"].to_numpy()
            colour = COLOURS[name]
            ax.fill_between(x, y - sd, y + sd, color=colour, alpha=0.10, lw=0)
            ax.plot(
                x, y, color=colour, lw=2, label=name, marker="o", ms=5,
                markeredgecolor=SURFACE, markeredgewidth=1.2,
            )
        ax.set_title(metric.upper() if metric == "f1" else metric.capitalize(),
                     fontsize=12, loc="left", color=INK)
        ax.set_ylim(-0.02, 1.02)
        plt.setp(ax.get_xticklabels(), visible=False)
        if metric != "precision":
            plt.setp(ax.get_yticklabels(), visible=False)

    handles, labels = top_axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=len(names),
               frameon=False, fontsize=11, labelcolor=INK_2,
               bbox_to_anchor=(0.9, 1.0))

    ref = summary[names[0]]
    x = ref.index.to_numpy()
    n_ill = ref["n_illicit"].to_numpy()
    low = n_ill < low_support
    y_max = max(float(n_ill.max()), 1.0)
    for i, ax in enumerate(bar_axes):
        ax.bar(x[~low], n_ill[~low], width=0.7, color=INK_MUTED)
        if low.any():
            ax.bar(x[low], n_ill[low], width=0.7, color="#c3c2b7",
                   hatch="///", edgecolor=INK_MUTED, linewidth=0)
        ax.set_ylim(0, y_max * 1.1)
        ax.set_xticks(x[::2])
        ax.tick_params(labelsize=8)
        if i > 0:
            plt.setp(ax.get_yticklabels(), visible=False)
    top_axes[0].set_ylabel("score (threshold 0.5)", fontsize=10)
    bar_axes[0].set_ylabel("illicit rows", fontsize=10)
    bar_axes[1].set_xlabel(
        f"time step (hatched bars: fewer than {low_support} illicit rows, "
        "so that step's metrics are very noisy)", fontsize=10)

    fig.suptitle(title, x=0.01, ha="left", fontsize=13, color=INK, y=1.0)
    fig.savefig(out_path, dpi=150, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)