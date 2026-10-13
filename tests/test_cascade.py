"""Tests for src/cascade.py (Day 7).

Everything is hand-computed on a 10-row toy example, so no dataset is needed.

    row     0     1     2     3     4     5     6     7     8     9
    y       1     1     1     0     0     0     0     0     0     0
    p1    0.90  0.50  0.10  0.80  0.20  0.05  0.40  0.02  0.60  0.01
    p2    0.95  0.70  0.80  0.30  0.10  0.10  0.20  0.10  0.90  0.10

With d = 0.5, d2 = 0.5, t_clear = 0.3:
    escalated (p1 >= 0.3): rows 0, 1, 3, 6, 8      -> 5 of 10, rate 0.5
    cleared: rows 2, 4, 5, 7, 9 (row 2 is a missed illicit)
    Stage 2 on escalated: row0 1, row1 1, row3 0, row6 0, row8 1 (false positive)
    cascade pred = [1,1,0,0,0,0,0,0,1,0]  -> tp 2, fp 1, fn 1
    precision = recall = 2/3

    graph-always (p2 >= 0.5): rows 0, 1, 2, 8 -> tp 3, fp 1, recall 1.0
    recall retention = (2/3) / 1 = 2/3

    local-only (p1 >= 0.5): rows 0, 1, 3, 8 -> tp 2, fp 2, fn 1
"""
import json

import joblib
import numpy as np
import pandas as pd
import pytest

from src.calibration import SigmoidCalibrator, predict_p1
from src.cascade import (
    choose_t_clear,
    graph_always,
    load_stage1,
    load_thresholds,
    local_only,
    random_control,
    route,
    run_cascade,
    save_thresholds,
    stage1_p1,
    sweep_t_clear,
    t_clear_grid,
    validate_thresholds,
)
from src.evaluate import illicit_metrics
from src.models import make_model

Y = np.array([1, 1, 1, 0, 0, 0, 0, 0, 0, 0])
P1 = np.array([0.90, 0.50, 0.10, 0.80, 0.20, 0.05, 0.40, 0.02, 0.60, 0.01])
P2 = np.array([0.95, 0.70, 0.80, 0.30, 0.10, 0.10, 0.20, 0.10, 0.90, 0.10])
D, D2, T = 0.5, 0.5, 0.3


# ---- routing ----------------------------------------------------------------

def test_route_boundary_is_escalated():
    mask = route([0.05, 0.10, 0.30, 0.60, 0.95], 0.30)
    assert mask.tolist() == [False, False, True, True, True]  # p1 == t_clear escalates


def test_route_extremes():
    assert route(P1, 0.0).all()                 # t_clear 0: escalate everything
    assert not route(P1, 1.0).any()             # the largest toy p1 is 0.90


def test_route_rejects_nan():
    with pytest.raises(ValueError):
        route([0.1, float("nan")], 0.3)


# ---- input validation (probabilities and thresholds) -----------------------------

@pytest.mark.parametrize("bad", [[0.1, 1.2], [-0.1, 0.5], [0.1, float("inf")], [float("-inf"), 0.2]])
def test_out_of_range_or_infinite_probabilities_are_rejected(bad):
    with pytest.raises(ValueError):
        route(bad, 0.3)
    with pytest.raises(ValueError):
        local_only(bad, 0.5)
    with pytest.raises(ValueError):
        graph_always(bad, 0.5)


def test_floating_point_overshoot_is_tolerated_but_clipped():
    p2 = P2.copy()
    p2[0] = 1.0 + 1e-12                           # e.g. an average of 300 tree outputs
    out = run_cascade(P1, p2, T, D2)
    assert out["pred"][0] == 1
    assert graph_always(p2, D2)[0] == 1


def test_p2_may_be_nan_only_where_cleared_but_not_out_of_range():
    p2 = P2.copy()
    p2[2] = np.nan                                # row 2 is cleared at T = 0.3
    assert run_cascade(P1, p2, T, D2)["pred"][2] == 0
    p2[2] = 1.5                                   # out of range is never accepted
    with pytest.raises(ValueError):
        run_cascade(P1, p2, T, D2)


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan"), float("inf")])
def test_invalid_d2_is_rejected_everywhere(bad):
    with pytest.raises(ValueError):
        run_cascade(P1, P2, T, bad)
    with pytest.raises(ValueError):
        graph_always(P2, bad)
    with pytest.raises(ValueError):
        random_control(P1, P2, D, bad, 0.5, seed=0)
    with pytest.raises(ValueError):
        validate_thresholds(0.1, 0.5, d2=bad)
    with pytest.raises(ValueError):
        sweep_t_clear(Y, P1, P2, D, bad, [0.0, 0.3])


@pytest.mark.parametrize("bad", [-0.1, 1.1, float("nan")])
def test_invalid_t_clear_and_d_are_rejected(bad):
    with pytest.raises(ValueError):
        route(P1, bad)
    with pytest.raises(ValueError):
        local_only(P1, bad)
    with pytest.raises(ValueError):
        random_control(P1, P2, bad, D2, 0.5, seed=0)


# ---- the cascade on the toy example ---------------------------------------------

def test_cascade_toy_escalation_count_is_five():
    out = run_cascade(P1, P2, T, D2)
    assert out["escalated"].sum() == 5
    assert out["escalated"].tolist() == [True, True, False, True, False,
                                         False, True, False, True, False]
    assert out["escalation_rate"] == pytest.approx(0.5)


def test_cascade_toy_predictions_and_metrics():
    out = run_cascade(P1, P2, T, D2)
    assert out["pred"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]
    m = illicit_metrics(Y, out["pred"])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 6)
    assert m["precision"] == pytest.approx(2 / 3)
    assert m["recall"] == pytest.approx(2 / 3)


def test_cleared_rows_are_always_licit_even_if_p2_is_high():
    # Row 2 has p1 = 0.10 < t_clear, so it is cleared although p2 = 0.80.
    out = run_cascade(P1, P2, T, D2)
    assert out["pred"][2] == 0


def test_p2_is_not_read_for_cleared_rows():
    p2 = P2.copy()
    cleared = ~route(P1, T)
    p2[cleared] = np.nan                         # graph features never computed
    out = run_cascade(P1, p2, T, D2)
    assert out["pred"].tolist() == [1, 1, 0, 0, 0, 0, 0, 0, 1, 0]


def test_nan_p2_for_an_escalated_row_is_an_error():
    p2 = P2.copy()
    p2[0] = np.nan                               # row 0 is escalated
    with pytest.raises(ValueError):
        run_cascade(P1, p2, T, D2)


def test_t_clear_zero_equals_graph_always():
    out = run_cascade(P1, P2, 0.0, D2)
    assert out["escalation_rate"] == 1.0
    assert out["pred"].tolist() == graph_always(P2, D2).tolist()


# ---- baselines -------------------------------------------------------------------

def test_graph_always_toy():
    pred = graph_always(P2, D2)
    assert pred.tolist() == [1, 1, 1, 0, 0, 0, 0, 0, 1, 0]
    m = illicit_metrics(Y, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (3, 1, 0)
    assert m["recall"] == 1.0


def test_local_only_toy():
    pred = local_only(P1, D)
    assert pred.tolist() == [1, 1, 0, 1, 0, 0, 0, 0, 1, 0]
    m = illicit_metrics(Y, pred)
    assert (m["tp"], m["fp"], m["fn"]) == (2, 2, 1)


# ---- random escalation control ------------------------------------------------------

def test_random_control_escalates_exactly_the_requested_share():
    out = random_control(P1, P2, D, D2, 0.5, seed=1)
    assert out["escalated"].sum() == 5
    assert out["escalation_rate"] == pytest.approx(0.5)


def test_random_control_is_reproducible_and_seed_dependent():
    a = random_control(P1, P2, D, D2, 0.5, seed=7)
    b = random_control(P1, P2, D, D2, 0.5, seed=7)
    assert a["escalated"].tolist() == b["escalated"].tolist()
    masks = {tuple(random_control(P1, P2, D, D2, 0.5, seed=s)["escalated"]) for s in range(20)}
    assert len(masks) > 1                        # different seeds pick different rows


def test_random_control_extremes_match_the_baselines():
    assert random_control(P1, P2, D, D2, 1.0, seed=0)["pred"].tolist() == graph_always(P2, D2).tolist()
    assert random_control(P1, P2, D, D2, 0.0, seed=0)["pred"].tolist() == local_only(P1, D).tolist()


def test_random_control_non_escalated_keep_stage1_decision():
    out = random_control(P1, P2, D, D2, 0.5, seed=3)
    keep = ~out["escalated"]
    assert out["pred"][keep].tolist() == local_only(P1, D)[keep].tolist()
    esc = out["escalated"]
    assert out["pred"][esc].tolist() == graph_always(P2, D2)[esc].tolist()


def test_random_control_validates_rate():
    with pytest.raises(ValueError):
        random_control(P1, P2, D, D2, 1.5, seed=0)


# ---- thresholds: t_clear <= d --------------------------------------------------------

def test_validate_thresholds_accepts_t_clear_up_to_d():
    validate_thresholds(0.0, 0.5)
    validate_thresholds(0.3, 0.5)
    validate_thresholds(0.5, 0.5)


@pytest.mark.parametrize("t_clear", [0.51, 0.9, -0.1])
def test_validate_thresholds_rejects_t_clear_above_d_or_negative(t_clear):
    with pytest.raises(ValueError):
        validate_thresholds(t_clear, 0.5)


def test_t_clear_grid_never_exceeds_d_and_starts_at_zero():
    g = t_clear_grid(0.39)
    assert g[0] == 0.0
    assert g.max() <= 0.39 + 1e-9
    assert g[-1] == pytest.approx(0.39)
    assert len(g) == 40


def test_sweep_refuses_a_grid_value_above_d():
    with pytest.raises(ValueError):
        sweep_t_clear(Y, P1, P2, D, D2, [0.0, 0.3, 0.6])


# ---- the validation sweep and choosing t_clear -------------------------------------------

def test_sweep_toy_values():
    s = sweep_t_clear(Y, P1, P2, D, D2, [0.0, 0.3, 0.5]).set_index("t_clear")
    # t_clear 0: everything escalated, identical to graph-always
    assert s.loc[0.0, "escalation_rate"] == 1.0
    assert s.loc[0.0, "recall_retention"] == pytest.approx(1.0)
    # t_clear 0.3: the hand-computed cascade above
    assert s.loc[0.3, "escalation_rate"] == pytest.approx(0.5)
    assert s.loc[0.3, "recall"] == pytest.approx(2 / 3)
    assert s.loc[0.3, "recall_retention"] == pytest.approx(2 / 3)
    # t_clear 0.5: rows with p1 >= 0.5 are rows 0, 1, 3, 8 -> 4 of 10
    assert s.loc[0.5, "escalation_rate"] == pytest.approx(0.4)
    assert s.loc[0.5, "recall_retention"] == pytest.approx(2 / 3)


def test_choose_t_clear_picks_the_largest_value_meeting_the_target():
    s = sweep_t_clear(Y, P1, P2, D, D2, [0.0, 0.3, 0.5])
    t, row, met = choose_t_clear(s, target=0.6)
    assert (t, met) == (0.5, True)
    t, row, met = choose_t_clear(s, target=0.95)
    assert (t, met) == (0.0, True)               # only escalating everything keeps 100%
    assert row["recall_retention"] == pytest.approx(1.0)


def test_choose_t_clear_reports_when_the_target_is_not_reached():
    sweep = pd.DataFrame({
        "t_clear": [0.1, 0.2, 0.3],
        "escalation_rate": [0.9, 0.7, 0.5],
        "recall_retention": [0.80, 0.90, 0.90],
    })
    t, row, met = choose_t_clear(sweep, target=0.95)
    assert met is False
    assert t == 0.3                              # best retention, largest t_clear among ties
    assert row["recall_retention"] == pytest.approx(0.90)


def test_choose_t_clear_rejects_bad_target_and_undefined_sweep():
    s = sweep_t_clear(Y, P1, P2, D, D2, [0.0, 0.3])
    with pytest.raises(ValueError):
        choose_t_clear(s, target=0.0)
    undefined = pd.DataFrame({"t_clear": [0.1], "recall_retention": [float("nan")]})
    with pytest.raises(ValueError):
        choose_t_clear(undefined, target=0.95)


def test_sweep_retention_is_nan_without_illicit_rows():
    s = sweep_t_clear(np.zeros(10, dtype=int), P1, P2, D, D2, [0.0, 0.3])
    assert s["recall_retention"].isna().all()


# ---- frozen artifacts and thresholds --------------------------------------------------------

def test_load_stage1_missing_file_gives_a_helpful_error(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        load_stage1(tmp_path / "stage1_rf.joblib")
    assert "scripts/e3_calibration.py" in str(exc.value)


def test_load_stage1_rejects_an_incomplete_artifact(tmp_path):
    joblib.dump({"family": "rf"}, tmp_path / "bad.joblib")
    with pytest.raises(ValueError):
        load_stage1(tmp_path / "bad.joblib")


def test_stage1_p1_matches_predict_p1(tmp_path):
    rng = np.random.default_rng(0)
    cols = ["local_1", "local_2", "local_3"]
    X = pd.DataFrame(rng.normal(size=(300, 3)), columns=cols)
    y = (X["local_1"] + 0.3 * rng.normal(size=300) > 0.7).astype(int).to_numpy()
    model = make_model("lr", 0).fit(X, y)
    calibrator = SigmoidCalibrator(1.0, 0.0)
    path = tmp_path / "stage1_lr.joblib"
    joblib.dump({"family": "lr", "model": model, "calibrator": calibrator.as_dict(),
                 "feature_columns": cols, "d": 0.5}, path)

    artifact = load_stage1(path)
    extra = X.assign(txId=np.arange(300), time_step=1)   # extra columns must be ignored
    expected = predict_p1(model, calibrator, X[cols])
    np.testing.assert_allclose(stage1_p1(artifact, extra), expected, atol=1e-12)


def test_thresholds_roundtrip(tmp_path):
    path = tmp_path / "thresholds.json"
    save_thresholds(path, "rf", d=0.39, d2=0.45, t_clear=0.12, target=0.95, target_met=True)
    loaded = load_thresholds(path)
    assert loaded["t_clear"] == pytest.approx(0.12)
    assert loaded["d"] == pytest.approx(0.39)
    assert loaded["d2"] == pytest.approx(0.45)
    assert loaded["chosen_on_steps"] == [29, 34]


def test_save_thresholds_refuses_an_invalid_d2(tmp_path):
    with pytest.raises(ValueError):
        save_thresholds(tmp_path / "t.json", "rf", d=0.39, d2=1.5, t_clear=0.1,
                        target=0.95, target_met=True)


def test_load_thresholds_rejects_an_invalid_d2(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"family": "rf", "d": 0.5, "d2": -1, "t_clear": 0.2}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


def test_save_thresholds_refuses_t_clear_above_d(tmp_path):
    with pytest.raises(ValueError):
        save_thresholds(tmp_path / "t.json", "rf", d=0.39, d2=0.45, t_clear=0.40,
                        target=0.95, target_met=True)


def test_load_thresholds_rejects_a_tampered_file(tmp_path):
    path = tmp_path / "thresholds.json"
    path.write_text(json.dumps({"family": "rf", "d": 0.3, "d2": 0.5, "t_clear": 0.4}),
                    encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


def test_load_thresholds_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_thresholds(tmp_path / "nope.json")


# ---- explicit validation gaps ---------------------------------------------------------------

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf"), -0.1, 1.1])
def test_random_control_rejects_non_finite_or_out_of_range_rate(bad):
    with pytest.raises(ValueError):
        random_control(P1, P2, D, D2, bad, seed=0)


@pytest.mark.parametrize("d", [float("nan"), float("inf"), -0.1, 1.5])
def test_t_clear_grid_rejects_invalid_d(d):
    with pytest.raises(ValueError):
        t_clear_grid(d)


@pytest.mark.parametrize("step", [float("nan"), float("inf"), 0.0, -0.01, 2.0])
def test_t_clear_grid_rejects_invalid_step(step):
    with pytest.raises(ValueError):
        t_clear_grid(0.5, step=step)


@pytest.mark.parametrize("bad_y", [
    [0, 2, 0], [0, -1, 1], [0, 0.5, 1], [0, float("nan"), 1], [[0, 1], [1, 0]],
])
def test_sweep_rejects_invalid_labels(bad_y):
    n = len(np.asarray(bad_y).ravel()) if np.ndim(bad_y) == 1 else 2
    p = np.linspace(0.1, 0.9, n)
    with pytest.raises(ValueError):
        sweep_t_clear(bad_y, p, p, D, D2, [0.0, 0.3])


@pytest.mark.parametrize("grid", [
    [], [float("nan")], [0.0, float("inf")], [-0.1], [0.0, 0.6], ["a", 0.1], [None],
])
def test_sweep_rejects_invalid_grid_values(grid):
    with pytest.raises(ValueError):
        sweep_t_clear(Y, P1, P2, D, D2, grid)


@pytest.mark.parametrize("d", [float("nan"), float("inf"), -0.1, 1.5])
def test_sweep_rejects_invalid_d_even_with_a_valid_grid(d):
    with pytest.raises(ValueError):
        sweep_t_clear(Y, P1, P2, d, D2, [0.0])


def test_load_thresholds_validates_family_and_target(tmp_path):
    base = {"family": "rf", "d": 0.5, "d2": 0.5, "t_clear": 0.2, "recall_retention_target": 0.95}
    path = tmp_path / "t.json"
    path.write_text(json.dumps(base), encoding="utf-8")
    assert load_thresholds(path)["family"] == "rf"
    for change in ({"family": "svm"}, {"family": None}, {"recall_retention_target": 0},
                   {"recall_retention_target": 1.5}, {"recall_retention_target": "high"},
                   {"recall_retention_target": None}):
        path.write_text(json.dumps({**base, **change}), encoding="utf-8")
        with pytest.raises(ValueError):
            load_thresholds(path)
    missing = dict(base)
    del missing["recall_retention_target"]
    path.write_text(json.dumps(missing), encoding="utf-8")
    with pytest.raises(ValueError):
        load_thresholds(path)


def test_save_thresholds_validates_family_and_target(tmp_path):
    with pytest.raises(ValueError):
        save_thresholds(tmp_path / "a.json", "svm", d=0.4, d2=0.5, t_clear=0.1,
                        target=0.95, target_met=True)
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError):
            save_thresholds(tmp_path / "b.json", "rf", d=0.4, d2=0.5, t_clear=0.1,
                            target=bad, target_met=True)