"""Models and feature sets (SPEC.md sections 3 and 5).

Two families only, so every comparison can be made within one family:
    "lr"  logistic regression (features scaled, class_weight="balanced")
    "rf"  random forest       (class_weight="balanced")

Feature sets:
    "local" = local_* columns
    "all"   = local_* + agg_* columns
time_step, txId and label are NEVER features; feature_columns() enforces it.
"""
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

FAMILIES = ("lr", "rf")
FEATURE_SETS = ("local", "all")
FORBIDDEN_FEATURES = ("txId", "time_step", "label")


def feature_columns(df, feature_set):
    """Column names for the LOCAL or ALL feature set."""
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"feature_set must be one of {FEATURE_SETS}, got {feature_set!r}")

    local = [c for c in df.columns if c.startswith("local_")]
    agg = [c for c in df.columns if c.startswith("agg_")]
    cols = local if feature_set == "local" else local + agg

    if not cols:
        raise ValueError("no feature columns found; was the dataset loaded with src.load_data?")
    bad = [c for c in cols if c in FORBIDDEN_FEATURES]
    if bad:
        raise ValueError(f"forbidden columns in feature set: {bad}")
    return cols


def make_model(family, seed):
    """Fresh, unfitted model of the given family."""
    if family == "lr":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
        )
    if family == "rf":
        return RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            n_jobs=-1,
            random_state=seed,
        )
    raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")