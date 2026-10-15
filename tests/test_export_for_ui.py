"""Tests for scripts/export_for_ui.py (Day 9). Synthetic results only: no data, no models."""
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


e5 = _load("e5_cascade_test", "e5_cascade_test.py")
ex = _load("export_for_ui", "export_for_ui.py")

from src.evaluate import cost_table  # noqa: E402

TH = {"family": "rf", "d": 0.4, "d2": 0.3, "t_clear": 0.4, "recall_retention_target": 0.95}


def _make_results(tmp_path, seed=3, with_spread=False):
    """Write the same files scripts/e5_cascade_test.py writes, from random scores."""
    rng = np.random.default_rng(seed)
    steps = np.repeat(np.arange(35, 50), 60)
    n = len(steps)
    label = (rng.random(n) < 0.1).astype(int)
    p1 = np.clip(0.15 + 0.5 * label + rng.normal(0, 0.2, n), 0, 1)
    p2 = np.clip(0.1 + 0.6 * label + rng.normal(0, 0.2, n), 0, 1)
    frame = pd.DataFrame({"txId": np.arange(1000, 1000 + n), "time_step": steps, "label": label})

    summary, per_step, _ = e5.evaluate_systems(frame, p1, p2, TH)
    curve = e5.cost_recall_curve(frame, p1, p2, TH, step=0.05)
    preds = e5.predictions_frame(frame, p1, p2, TH)
    rate = float(preds["escalated"].mean())
    costs = cost_table(rate, (5, 20, 100))
    casc_recall = float(summary.loc[summary["system"] == "cascade", "recall"].iloc[0])
    costs["cascade_recall"] = casc_recall

    out = tmp_path / "results"
    out.mkdir()
    summary.to_csv(out / "e5_summary.csv", index=False)
    per_step.to_csv(out / "e5_per_step.csv", index=False)
    costs.to_csv(out / "e5_cost_table.csv", index=False)
    curve.to_csv(out / "e5_cost_recall_curve.csv", index=False)
    preds.to_csv(out / "cascade_predictions.csv", index=False)
    meta = {"family": "rf", "thresholds": {k: TH[k] for k in ("d", "d2", "t_clear")},
            "stage1_trained_on_steps": [1, 28], "stage1_calibrated_on_steps": [29, 34],
            "stage2_final_trained_on_steps": [1, 34], "test_steps": [35, 49],
            "n_test": n, "test_escalation_rate": rate, "validation_escalation_rate": 0.2}
    (out / "e5_run_meta.json").write_text(json.dumps(meta), encoding="utf-8")
    if with_spread:
        pd.DataFrame({"family": ["rf"], "seed": [42], "system": ["cascade"],
                      "f1": [0.8]}).to_csv(out / "e6_seed_spread.csv", index=False)
    return out


@pytest.fixture()
def results(tmp_path):
    return _make_results(tmp_path)


def _payload(results_dir, cleared=5):
    data = ex.read_inputs(results_dir)
    ex.validate(data, TH, (35, 49))
    return ex.build(data, TH, cleared_per_step=cleared)


def test_payload_has_all_sections(results):
    payload = _payload(results)
    assert list(payload) == ["meta", "kpis", "systems", "cost", "curve", "per_step",
                             "batches", "queue", "seed_spread", "honesty"]
    assert payload["meta"]["cost_label"] == "simulated"
    assert payload["meta"]["windows"]["test"] == [35, 49]
    assert payload["cost"]["r_values"] == [5, 20, 100]
    assert payload["seed_spread"] is None


def test_json_has_no_nan_and_round_trips(results, tmp_path):
    payload = _payload(results)
    out = tmp_path / "web" / "data" / "results.json"
    ex.write_json(payload, out)
    text = out.read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    assert json.loads(text)["kpis"]["transactions"] == 900


def test_nan_becomes_null():
    assert ex.clean(float("nan")) is None
    assert ex.clean(np.float64("inf")) is None
    assert ex.clean({"a": np.int64(3), "b": [np.float32(0.123456789)]}) == {"a": 3, "b": [0.12346]}


def test_steps_stay_inside_the_test_window(results):
    payload = _payload(results)
    assert {b["time_step"] for b in payload["batches"]} == set(range(35, 50))
    assert all(35 <= q["time_step"] <= 49 for q in payload["queue"])
    for rows in payload["per_step"].values():
        assert all(35 <= r["time_step"] <= 49 for r in rows)


def test_queue_has_every_escalated_row_and_a_capped_cleared_sample(results):
    preds = pd.read_csv(results / "cascade_predictions.csv")
    payload = _payload(results, cleared=5)
    queue = pd.DataFrame(payload["queue"])
    assert int(queue["escalated"].sum()) == int(preds["escalated"].sum())
    cleared = queue[~queue["escalated"]]
    assert cleared.groupby("time_step").size().max() <= 5
    assert queue["p2"][~queue["escalated"]].isna().all()
    assert queue["p2"][queue["escalated"]].notna().all()
    assert set(queue["outcome"]) <= {"caught", "false alarm", "missed", "correctly cleared"}


def test_queue_is_deterministic(results):
    a = _payload(results)
    b = _payload(results)
    assert a["queue"] == b["queue"]


def test_zero_cleared_sample_keeps_only_escalated(results):
    payload = _payload(results, cleared=0)
    assert all(q["escalated"] for q in payload["queue"])


def test_batches_add_up_to_the_totals(results):
    payload = _payload(results)
    batches = payload["batches"]
    assert sum(b["n"] for b in batches) == payload["kpis"]["transactions"]
    assert sum(b["escalated"] for b in batches) == payload["kpis"]["escalated"]
    assert sum(b["cleared"] for b in batches) == payload["kpis"]["cleared_at_stage1"]
    assert sum(b["caught_cascade"] for b in batches) == payload["kpis"]["illicit_caught_cascade"]
    assert sum(b["n_illicit"] for b in batches) == payload["kpis"]["illicit_transactions"]


def test_kpis_match_the_summary_file(results):
    payload = _payload(results)
    summary = pd.read_csv(results / "e5_summary.csv").set_index("system")
    assert payload["kpis"]["recall_cascade"] == pytest.approx(summary.loc["cascade", "recall"], abs=1e-5)
    assert payload["kpis"]["recall_retention"] == pytest.approx(
        summary.loc["cascade", "recall_retention"], abs=1e-5)
    assert payload["kpis"]["illicit_caught_cascade"] == int(summary.loc["cascade", "tp"])
    assert isinstance(payload["kpis"]["illicit_caught_cascade"], int)


def test_cost_rows_follow_the_formula(results):
    payload = _payload(results)
    rate = payload["cost"]["escalation_rate"]
    for row in payload["cost"]["rows"]:
        assert row["graph_always_cost"] == pytest.approx(1 + row["r"], abs=1e-4)
        assert row["cascade_cost"] == pytest.approx(1 + rate * row["r"], abs=1e-3)
        assert row["relative_savings"] == pytest.approx(
            1 - row["cascade_cost"] / row["graph_always_cost"], abs=1e-3)


def test_frozen_point_is_marked_on_the_curve(results):
    payload = _payload(results)
    assert payload["curve"]["frozen_point"]["t_clear"] == pytest.approx(TH["t_clear"], abs=1e-4)
    assert len(payload["curve"]["points"]) > 3


def test_honesty_text_is_present(results):
    text = " ".join(_payload(results)["honesty"]).lower()
    for phrase in ("simulated", "anonymised", "drift", "d2", "validation"):
        assert phrase in text


def test_seed_spread_is_included_when_present(tmp_path):
    results = _make_results(tmp_path, with_spread=True)
    payload = _payload(results)
    assert payload["seed_spread"][0]["system"] == "cascade"


# --- guards ---------------------------------------------------------------

def test_missing_inputs_give_a_clear_error(tmp_path):
    (tmp_path / "results").mkdir()
    with pytest.raises(FileNotFoundError, match="e5_cascade_test"):
        ex.read_inputs(tmp_path / "results")


def test_rejects_results_from_other_thresholds(results):
    data = ex.read_inputs(results)
    other = dict(TH, t_clear=0.3)
    with pytest.raises(ValueError, match="t_clear"):
        ex.validate(data, other, (35, 49))


def test_rejects_steps_outside_the_test_window(results):
    data = ex.read_inputs(results)
    data["preds"].loc[0, "time_step"] = 20
    with pytest.raises(ValueError, match="outside the test window"):
        ex.validate(data, TH, (35, 49))


def test_rejects_a_row_count_mismatch(results):
    data = ex.read_inputs(results)
    data["preds"] = data["preds"].iloc[:-1]
    with pytest.raises(ValueError, match="n_test"):
        ex.validate(data, TH, (35, 49))


def test_rejects_p2_on_cleared_rows(results):
    data = ex.read_inputs(results)
    idx = data["preds"].index[data["preds"]["escalated"] == 0][0]
    data["preds"].loc[idx, "p2"] = 0.5
    with pytest.raises(ValueError, match="p2 must be blank"):
        ex.validate(data, TH, (35, 49))


def test_size_limit_is_enforced(results, tmp_path, monkeypatch):
    monkeypatch.setattr(ex, "MAX_BYTES", 1000)
    with pytest.raises(ValueError, match="MB"):
        ex.write_json(_payload(results), tmp_path / "x.json")


def test_export_never_reads_the_dataset_or_thresholds_files_it_does_not_need(results):
    """read_inputs only opens files in the results directory."""
    before = sorted(p.name for p in results.iterdir())
    ex.read_inputs(results)
    assert sorted(p.name for p in results.iterdir()) == before