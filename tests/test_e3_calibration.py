
"""Tests for scripts/e3_calibration.py (Day 6: Stage 1 training and calibration).

Synthetic data only; no real Elliptic dataset required.

Checks:
  1. Training and calibration use only their allowed chronological windows.
  2. Frozen artifacts record the expected setup and metadata.
  3. Rebuilding with the same seed gives reproducible predictions and d.
  4. The standalone threshold d is preserved; t_clear is not stored here.
  5. Results files and threshold-grid configuration behave as expected.

Run:
    python -m pytest tests/test_e3_calibration.py -q
"""

import importlib.util
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "e3_calibration.py"
sys.path.insert(0, str(ROOT))

from src.calibration import SigmoidCalibrator  # noqa: E402

SEED = 42
N_FEATURES = 8
FAMILIES = ("rf", "lr")


def _load_script():
    spec = importlib.util.spec_from_file_location("e3_calibration", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load calibration script: {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e3 = _load_script()


def _synthetic_frame():
    """Create labelled synthetic rows over time steps 1-49."""
    rng = np.random.default_rng(0)
    n_steps, per_step = 49, 40
    n = n_steps * per_step

    X = rng.normal(size=(n, N_FEATURES))
    y = (
        X[:, 0] + 0.5 * rng.normal(size=n) > 1.2
    ).astype(int)

    df = pd.DataFrame(
        X, columns=[f"local_{i + 1}" for i in range(N_FEATURES)]
    )
    df.insert(0, "time_step", np.repeat(np.arange(1, n_steps + 1), per_step))
    df.insert(0, "txId", np.arange(n))
    df["label"] = y
    return df


def _stage1_splits(df):
    """Supply only the permitted Stage 1 training and calibration windows."""
    return {
        "train_stage1": (
            df[df["time_step"].between(1, 28)].reset_index(drop=True)
        ),
        "calib": (
            df[df["time_step"].between(29, 34)].reset_index(drop=True)
        ),
    }


def _run(family, base_dir):
    """Run one family using synthetic data and isolated output directories."""
    df = _synthetic_frame()
    splits = _stage1_splits(df)
    cols = [c for c in df.columns if c.startswith("local_")]

    out_dir = Path(base_dir) / "results"
    art_dir = Path(base_dir) / "artifacts"
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    grid = e3.threshold_grid({})
    row, rel = e3.run_family(
        family, SEED, splits, cols, grid, 10, out_dir, art_dir
    )
    artifact = joblib.load(art_dir / f"stage1_{family}.joblib")

    return {
        "row": row,
        "rel": rel,
        "artifact": artifact,
        "cols": cols,
        "out_dir": out_dir,
        "art_dir": art_dir,
        "df": df,
        "train_steps": set(splits["train_stage1"]["time_step"].unique()),
        "calib_steps": set(splits["calib"]["time_step"].unique()),
    }


@pytest.fixture(scope="module", params=FAMILIES)
def built(request, tmp_path_factory):
    """Build one synthetic artifact per model family."""
    family = request.param
    result = _run(
        family, tmp_path_factory.mktemp(f"e3_{family}")
    )
    result["family"] = family
    return result


# ---- 1. Temporal boundaries -----------------------------------------------

def test_stage1_receives_only_allowed_training_and_calibration_steps(built):
    train_steps = built["train_steps"]
    calib_steps = built["calib_steps"]

    assert train_steps
    assert min(train_steps) >= 1
    assert max(train_steps) <= 28

    assert calib_steps
    assert min(calib_steps) >= 29
    assert max(calib_steps) <= 34

    assert train_steps.isdisjoint(range(35, 50))
    assert calib_steps.isdisjoint(range(35, 50))


def test_recorded_windows_end_before_test_steps(built):
    artifact = built["artifact"]

    assert artifact["train_steps"] == [1, 28]
    assert artifact["calib_steps"] == [29, 34]
    assert max(
        artifact["train_steps"] + artifact["calib_steps"]
    ) < 35


# ---- 2. Artifact metadata and outputs -------------------------------------

def test_artifact_records_setup_and_metadata(built):
    artifact = built["artifact"]

    assert artifact["family"] == built["family"]
    assert artifact["seed"] == SEED
    assert artifact["feature_columns"] == built["cols"]

    metadata = artifact["metadata"]
    assert metadata["training_window"] == "steps 1-28"
    assert metadata["calibration_window"] == "steps 29-34"

    for key in (
        "brier_raw",
        "brier_calibrated",
        "numpy_version",
        "pandas_version",
        "sklearn_version",
    ):
        assert key in metadata, f"metadata is missing {key}"

    assert 0 <= metadata["brier_raw"] <= 1
    assert 0 <= metadata["brier_calibrated"] <= 1


def test_results_files_are_written(built):
    family = built["family"]
    out_dir = built["out_dir"]

    assert (out_dir / f"stage1_{family}.json").exists()
    assert (
        out_dir / "figures" / f"e3_reliability_{family}.png"
    ).exists()

    saved = json.loads(
        (out_dir / f"stage1_{family}.json").read_text(encoding="utf-8")
    )
    assert saved["family"] == family
    assert saved["seed"] == SEED


def test_calibration_keeps_the_ranking(built):
    row = built["row"]
    assert row["pr_auc_cal"] == pytest.approx(
        row["pr_auc_raw"], abs=1e-6
    )


# ---- 3. Reproducibility ----------------------------------------------------
@pytest.mark.slow

@pytest.mark.parametrize("family", FAMILIES)
def test_rebuild_twice_gives_same_predictions_and_threshold(
    family, tmp_path
):
    first = _run(family, tmp_path / "first")
    second = _run(family, tmp_path / "second")

    # Fixed input rows from calibration steps.
    df = first["df"]
    X = df.loc[df["time_step"].between(29, 34), first["cols"]]

    model1 = first["artifact"]["model"]
    model2 = second["artifact"]["model"]

    p1 = model1.predict_proba(X)[:, 1]
    p2 = model2.predict_proba(X)[:, 1]
    np.testing.assert_allclose(p1, p2, rtol=0, atol=1e-9)

    calibrator1 = SigmoidCalibrator.from_dict(
        first["artifact"]["calibrator"]
    )
    calibrator2 = SigmoidCalibrator.from_dict(
        second["artifact"]["calibrator"]
    )

    calibrated1 = calibrator1.transform(p1)
    calibrated2 = calibrator2.transform(p2)
    np.testing.assert_allclose(
        calibrated1, calibrated2, rtol=0, atol=1e-9
    )

    assert first["artifact"]["d"] == pytest.approx(
        second["artifact"]["d"], abs=1e-9
    )


# ---- 4. Frozen threshold protocol ------------------------------------------

def test_artifact_preserves_d_but_not_t_clear(built):
    artifact = built["artifact"]

    # d is the standalone Stage 1 threshold selected on calibration steps.
    assert "d" in artifact
    assert 0.0 <= artifact["d"] <= 1.0

    # t_clear is a separate cascade-routing threshold selected on Day 7.
    assert "t_clear" not in artifact
    assert "t_clear" not in artifact["metadata"]


# ---- 5. Threshold grid -----------------------------------------------------

def test_threshold_grid_defaults():
    grid = e3.threshold_grid({})

    assert grid[0] == pytest.approx(0.01)
    assert grid[-1] == pytest.approx(0.99)
    assert len(grid) == 99


def test_threshold_grid_reads_config():
    grid = e3.threshold_grid(
        {"threshold_grid": {"start": 0.1, "stop": 0.5, "step": 0.1}}
    )

    assert list(grid) == pytest.approx([0.1, 0.2, 0.3, 0.4, 0.5])
