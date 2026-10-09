"""Tests for src/splits.py (SPEC Section 4 and Section 11 split-ordering test).

Toy tests always run. The real-data test checks the split counts against
the Day 1 output and is skipped if the dataset or config.yaml is missing.

    pytest tests/test_splits.py -v -s
"""

import os
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.splits import assert_chronological, make_splits  # noqa: E402

CFG = {
    "splits": {
        "train_stage1": [1, 28],
        "calib": [29, 34],
        "train_final": [1, 34],
        "test": [35, 49],
    }
}


def _toy_df():
    """49 steps x 3 rows: one illicit, one licit, one unlabelled each."""
    rows = []
    for step in range(1, 50):
        base = step * 10
        rows += [
            {"txId": base + 1, "time_step": step, "label": 1, "local_1": 0.0},
            {"txId": base + 2, "time_step": step, "label": 0, "local_1": 0.0},
            {"txId": base + 3, "time_step": step, "label": -1, "local_1": 0.0},
        ]
    return pd.DataFrame(rows)


def test_step_ranges_are_as_configured():
    splits = make_splits(_toy_df(), CFG)
    got = {
        name: (int(s["time_step"].min()), int(s["time_step"].max()))
        for name, s in splits.items()
    }
    assert got == {
        "train_stage1": (1, 28),
        "calib": (29, 34),
        "train_final": (1, 34),
        "test": (35, 49),
    }


def test_unlabelled_dropped_and_counts_known_by_hand():
    splits = make_splits(_toy_df(), CFG)
    for s in splits.values():
        assert set(s["label"].unique()) <= {0, 1}
    # 2 labelled rows per step (1 illicit + 1 licit)
    assert {n: len(s) for n, s in splits.items()} == {
        "train_stage1": 56,   # 28 steps
        "calib": 12,          # 6 steps
        "train_final": 68,    # 34 steps
        "test": 30,           # 15 steps
    }
    assert {n: int((s["label"] == 1).sum()) for n, s in splits.items()} == {
        "train_stage1": 28,
        "calib": 6,
        "train_final": 34,
        "test": 15,
    }


def test_windows_do_not_share_rows():
    splits = make_splits(_toy_df(), CFG)
    ids = {n: set(s["txId"]) for n, s in splits.items()}
    assert ids["train_stage1"].isdisjoint(ids["calib"])
    assert ids["train_stage1"].isdisjoint(ids["test"])
    assert ids["calib"].isdisjoint(ids["test"])
    assert ids["train_final"].isdisjoint(ids["test"])
    assert ids["train_final"] == ids["train_stage1"] | ids["calib"]


@pytest.mark.parametrize(
    "target, source",
    [
        ("train_stage1", "calib"),
        ("train_stage1", "test"),
        ("train_final", "test"),
        ("calib", "test"),
    ],
)
def test_leak_into_earlier_split_is_detected(target, source):
    """Split-ordering test: a later-step row inside an earlier split must fail."""
    splits = make_splits(_toy_df(), CFG)
    leaked_row = splits[source].tail(1)
    splits[target] = pd.concat([splits[target], leaked_row])
    with pytest.raises(AssertionError):
        assert_chronological(splits)


def test_overlapping_config_is_rejected():
    bad_cfg = {"splits": dict(CFG["splits"], train_stage1=[1, 30])}
    with pytest.raises(AssertionError):
        make_splits(_toy_df(), bad_cfg)


def test_inconsistent_final_train_is_rejected():
    bad_cfg = {"splits": dict(CFG["splits"], train_final=[1, 30])}
    with pytest.raises(AssertionError):
        make_splits(_toy_df(), bad_cfg)


def test_missing_or_malformed_config_raises_value_error():
    with pytest.raises(ValueError):
        make_splits(_toy_df(), {"splits": {"calib": [29, 34]}})
    with pytest.raises(ValueError):
        make_splits(_toy_df(), {"splits": dict(CFG["splits"], test=[49, 35])})


def test_missing_required_column_raises_value_error():
    with pytest.raises(ValueError):
        make_splits(_toy_df().drop(columns=["time_step"]), CFG)


def test_real_split_counts_match_day1():
    yaml = pytest.importorskip("yaml")
    config_path = ROOT / "config.yaml"
    data_dir = Path(os.environ.get("ELLIPTIC_DIR", ROOT / "data" / "raw"))
    if not config_path.exists():
        pytest.skip("config.yaml not found")
    if not (data_dir / "elliptic_txs_features.csv").exists():
        pytest.skip(f"Dataset not found in {data_dir}")

    from src import load_data as ld

    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    splits = make_splits(ld.build_dataset(data_dir), cfg)

    expected = {  # (labelled rows, illicit rows) from the Day 1 output
        "train_stage1": (25207, 2542),
        "calib": (4687, 920),
        "train_final": (29894, 3462),
        "test": (16670, 1083),
    }
    for name, (n_rows, n_illicit) in expected.items():
        s = splits[name]
        got = (len(s), int((s["label"] == 1).sum()))
        print(f"[DAY2] {name}: labelled={got[0]}, illicit={got[1]}, "
              f"illicit_share={got[1] / got[0]:.4f}")
        assert got == (n_rows, n_illicit)