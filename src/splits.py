"""Chronological splits for the Elliptic cascade (SPEC.md Section 4).

make_splits(df, cfg) -> dict of DataFrames, LABELLED nodes only:
    train_stage1  Stage 1 training, and the validation copy of Stage 2
    calib         calibration of Stage 1; choice of d and t_clear
    train_final   final training of standalone RQ1 models and final Stage 2
    test          evaluated once, at the very end

assert_chronological(splits) raises AssertionError on any leakage.

Expected config shape (config.yaml):
    splits:
      train_stage1: [1, 28]
      calib: [29, 34]
      train_final: [1, 34]
      test: [35, 49]
"""

SPLIT_NAMES = ("train_stage1", "calib", "train_final", "test")


def _get_window(cfg, name):
    """Return (first_step, last_step), both inclusive, for one split."""
    try:
        first, last = cfg["splits"][name]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"cfg['splits']['{name}'] must be a [first_step, last_step] pair"
        ) from exc

    first, last = int(first), int(last)
    if first > last:
        raise ValueError(
            f"split '{name}' has first step {first} > last step {last}"
        )
    return first, last


def make_splits(df, cfg):
    """Cut the dataset into the four chronological splits.

    Unlabelled nodes (label == -1) are dropped here, so no later step can
    train or evaluate on them (SPEC rule 5). time_step is used only to
    cut; it stays a plain column and is never a feature.
    """
    for column in ("time_step", "label"):
        if column not in df.columns:
            raise ValueError(f"df is missing required column '{column}'")

    allowed_labels = {-1, 0, 1}

    if df["label"].isna().any():
        raise ValueError("label contains missing values")

    unexpected = set(df["label"].unique()) - allowed_labels
    if unexpected:
        raise ValueError(f"Unexpected label values: {unexpected}")

    labelled = df[df["label"].isin([0, 1])].copy()

    splits = {}
    for name in SPLIT_NAMES:
        first, last = _get_window(cfg, name)
        in_window = labelled["time_step"].between(first, last)
        splits[name] = labelled.loc[in_window].copy()

    assert_chronological(splits)
    return splits


def assert_chronological(splits):
    """Raise AssertionError if any training step is not strictly earlier
    than the steps it is validated or tested on."""
    missing = [name for name in SPLIT_NAMES if name not in splits]
    if missing:
        raise AssertionError(f"splits is missing: {missing}")

    for name in SPLIT_NAMES:
        if splits[name].empty:
            raise AssertionError(f"split '{name}' is empty")

    first = {n: int(splits[n]["time_step"].min()) for n in SPLIT_NAMES}
    last = {n: int(splits[n]["time_step"].max()) for n in SPLIT_NAMES}

    must_be_earlier = [
        ("train_stage1", "calib"),
        ("calib", "test"),
        ("train_final", "test"),
    ]
    for earlier, later in must_be_earlier:
        if last[earlier] >= first[later]:
            raise AssertionError(
                f"Leakage: '{earlier}' reaches step {last[earlier]} but "
                f"'{later}' starts at step {first[later]}."
            )

    steps_final = set(splits["train_final"]["time_step"].unique())
    steps_parts = (
        set(splits["train_stage1"]["time_step"].unique())
        | set(splits["calib"]["time_step"].unique())
    )
    if steps_final != steps_parts:
        raise AssertionError(
            "'train_final' must cover exactly the steps of 'train_stage1' "
            "plus 'calib' (SPEC Section 4)."
        )
    if "txId" not in splits["train_final"].columns:
        raise AssertionError("Splits must contain the 'txId' column.")

    ids_final = set(splits["train_final"]["txId"])
    ids_parts = (
        set(splits["train_stage1"]["txId"])
      | set(splits["calib"]["txId"])
    )

    if ids_final != ids_parts:
        raise AssertionError(
           "'train_final' must contain exactly the transaction IDs "
           "from 'train_stage1' plus 'calib'."
        )