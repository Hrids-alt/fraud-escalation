"""Load the Elliptic Bitcoin dataset into clean, validated DataFrames.

Column conventions (SPEC.md Section 3):
    txId, time_step, local_1..local_N, agg_1..agg_M, label

Label coding:
    raw "1"       -> illicit = 1
    raw "2"       -> licit = 0
    raw "unknown" -> unlabelled = -1

Design rule: every assumption about the file layout is checked at load
time and raises a clear error, so a wrong assumption cannot silently
corrupt later results.
"""

from pathlib import Path

import pandas as pd


FEATURES_FILE = "elliptic_txs_features.csv"
CLASSES_FILE = "elliptic_txs_classes.csv"
EDGES_FILE = "elliptic_txs_edgelist.csv"

EXPECTED_FEATURE_COLS = 167   # txId + time step + 93 local + 72 aggregated
EXPECTED_N_TX = 203_769
EXPECTED_N_EDGES = 234_355

# Set only after verifying the actual feature layout.
N_LOCAL = 93
N_AGG = 72

TIME_STEP_MIN = 1
TIME_STEP_MAX = 49

LABEL_MAP = {
    "1": 1,
    "2": 0,
    "unknown": -1,
}

NON_FEATURE_COLUMNS = ("txId", "time_step", "label")


def _require_file(path):
    if not Path(path).exists():
        raise FileNotFoundError(f"File not found: {path}")


def feature_column_names(n_cols):
    """Return verified column names for the features file."""
    if n_cols != EXPECTED_FEATURE_COLS:
        raise ValueError(
            f"Features file has {n_cols} columns, "
            f"but {EXPECTED_FEATURE_COLS} were expected. "
            "Check the dataset format before continuing."
        )

    expected_total = 2 + N_LOCAL + N_AGG

    if expected_total != n_cols:
        raise ValueError(
            f"N_LOCAL={N_LOCAL} and N_AGG={N_AGG} do not match "
            f"the observed column count of {n_cols}."
        )

    return (
        ["txId", "time_step"]
        + [f"local_{i}" for i in range(1, N_LOCAL + 1)]
        + [f"agg_{i}" for i in range(1, N_AGG + 1)]
    )


def load_features(data_dir):
    """Load the transaction features file (no header in the raw CSV)."""
    path = Path(data_dir) / FEATURES_FILE
    _require_file(path)

    df = pd.read_csv(path, header=None)
    df.columns = feature_column_names(df.shape[1])

    n_missing = int(df.isna().sum().sum())
    if n_missing:
        raise ValueError(f"Features file contains {n_missing} missing values.")

    if df["txId"].duplicated().any():
        raise ValueError("Duplicate txId values in the features file.")

    steps = df["time_step"]
    if not (steps == steps.round()).all():
        raise ValueError("time_step contains non-integer values.")

    df["txId"] = df["txId"].astype("int64")
    df["time_step"] = df["time_step"].astype("int64")

    if df["time_step"].min() < TIME_STEP_MIN or df["time_step"].max() > TIME_STEP_MAX:
        raise ValueError(
            f"time_step outside [{TIME_STEP_MIN}, {TIME_STEP_MAX}]: "
            f"found {df['time_step'].min()}..{df['time_step'].max()}."
        )

    return df


def load_classes(data_dir):
    """Load transaction classes and map them to project labels."""
    path = Path(data_dir) / CLASSES_FILE
    _require_file(path)

    df = pd.read_csv(path, dtype=str)

    expected_columns = {"txId", "class"}
    if set(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected classes columns: {list(df.columns)}. "
            "Expected: ['txId', 'class']."
        )

    df = df.rename(columns={"class": "raw_class"})

    if df["raw_class"].isna().any():
        raise ValueError("Classes file contains empty class values.")

    df["raw_class"] = df["raw_class"].str.strip()

    unexpected = sorted(set(df["raw_class"].unique()) - set(LABEL_MAP))
    if unexpected:
        raise ValueError(f"Unexpected class values found: {unexpected}")

    df["txId"] = pd.to_numeric(df["txId"]).astype("int64")

    if df["txId"].duplicated().any():
        raise ValueError("Duplicate txId values in the classes file.")

    df["label"] = df["raw_class"].map(LABEL_MAP).astype(int)

    return df[["txId", "raw_class", "label"]]


def load_edges(data_dir):
    """Load the directed transaction edge list."""
    path = Path(data_dir) / EDGES_FILE
    _require_file(path)

    df = pd.read_csv(path)

    expected_columns = {"txId1", "txId2"}
    if set(df.columns) != expected_columns:
        raise ValueError(
            f"Unexpected edge columns: {list(df.columns)}. "
            "Expected: ['txId1', 'txId2']."
        )

    if df.isna().any().any():
        raise ValueError("Edge list contains missing values.")

    df = df[["txId1", "txId2"]].astype("int64")

    return df


def build_dataset(data_dir):
    """Combine transaction features with labels.

    All transactions are retained.
    label == -1 means the transaction is unlabelled.
    """
    features = load_features(data_dir)
    classes = load_classes(data_dir)[["txId", "label"]]

    n_orphan = int((~classes["txId"].isin(features["txId"])).sum())
    if n_orphan:
        raise ValueError(
            f"{n_orphan} txIds in the classes file are not in the features file."
        )

    dataset = features.merge(
        classes,
        on="txId",
        how="left",
        validate="one_to_one",
    )

    if dataset["label"].isna().any():
        missing = int(dataset["label"].isna().sum())
        raise ValueError(
            f"{missing} transactions have no corresponding class entry."
        )

    dataset["label"] = dataset["label"].astype(int)

    return dataset


def local_columns(df):
    """Return all local feature columns."""
    return [column for column in df.columns if column.startswith("local_")]


def agg_columns(df):
    """Return all aggregated feature columns."""
    return [column for column in df.columns if column.startswith("agg_")]


def all_feature_columns(df):
    """Return LOCAL + AGG columns (the ALL feature set). Never includes
    txId, time_step or label."""
    return local_columns(df) + agg_columns(df)


def edge_step_report(dataset, edges):
    """Check whether edges stay within one time step.

    Returns a dict with the number of edges, edges whose endpoint is not
    in the dataset, and edges that connect two different time steps.
    """
    step_of = dataset.set_index("txId")["time_step"]
    step_1 = edges["txId1"].map(step_of)
    step_2 = edges["txId2"].map(step_of)

    missing = step_1.isna() | step_2.isna()
    cross = (~missing) & (step_1 != step_2)

    return {
        "n_edges": int(len(edges)),
        "n_missing_endpoint": int(missing.sum()),
        "n_cross_step": int(cross.sum()),
    }


def dataset_summary(dataset, edges):
    """Headline counts used for the Day 1 verification."""
    return {
        "n_transactions": int(len(dataset)),
        "n_edges": int(len(edges)),
        "n_illicit": int((dataset["label"] == 1).sum()),
        "n_licit": int((dataset["label"] == 0).sum()),
        "n_unlabelled": int((dataset["label"] == -1).sum()),
        "n_local_features": len(local_columns(dataset)),
        "n_agg_features": len(agg_columns(dataset)),
        "time_step_min": int(dataset["time_step"].min()),
        "time_step_max": int(dataset["time_step"].max()),
    }