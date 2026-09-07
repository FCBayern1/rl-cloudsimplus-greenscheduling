"""Judge helpers of the regret instrument gate (reports/EUCRD_REGRET_SIGNAL_PREREG.md §3)."""

import csv
import importlib.util
import os
import sys

import pytest

G1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..",
                                  "g1", "compressed_timecap_s2"))


def _gate():
    sys.path.insert(0, G1)
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    spec = importlib.util.spec_from_file_location(
        "eucrd_regret_gate", os.path.join(G1, "eucrd_regret_gate.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _dump(path, rows):
    cols = ["episode", "step", "slot", "cloudlet_id", "site", "kappa"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow(r)


def test_plan_start_is_step_plus_offset_plus_lag(tmp_path):
    gate = _gate()
    p = tmp_path / "d.csv"
    _dump(p, [
        {"episode": 0, "step": 3, "slot": 0, "cloudlet_id": 11, "site": 2, "kappa": 5},
        {"episode": 0, "step": 3, "slot": 1, "cloudlet_id": -1, "site": 0, "kappa": 0},
        {"episode": 0, "step": 9, "slot": 0, "cloudlet_id": 12, "site": 0, "kappa": 0},
    ])
    plan, stats = gate.schedule_from_decisions(str(p))
    assert plan == {11: [2, 3 + 5 + 1], 12: [0, 9 + 0 + 1]}
    assert stats == {"jobs": 2, "multi_sighting": 0}


def test_repeated_sightings_keep_the_last_decision_and_are_counted(tmp_path):
    gate = _gate()
    p = tmp_path / "d.csv"
    _dump(p, [
        {"episode": 0, "step": 1, "slot": 0, "cloudlet_id": 7, "site": 0, "kappa": 4},
        {"episode": 0, "step": 2, "slot": 0, "cloudlet_id": 7, "site": 1, "kappa": 0},
    ])
    plan, stats = gate.schedule_from_decisions(str(p))
    assert plan == {7: [1, 3]}
    assert stats["multi_sighting"] == 1


def test_verdict_names_every_failing_gate(tmp_path, monkeypatch):
    gate = _gate()
    monkeypatch.setattr(gate, "OUT", str(tmp_path))
    monkeypatch.setattr(gate, "windows", lambda: [0, 1, 2])
    res = gate.judge()                      # no runs on disk: every gate must fail, not crash
    assert res["verdict"].startswith("STOP_REGRET_SIGNAL:")
    for k in ("H1_zero_under_godeye", "H2_alive_under_shrink75", "H4_fixed_state_pairing",
              "H5_no_observation_leak"):
        assert k in res["verdict"]


def test_control_block_differs_only_in_the_source(tmp_path, monkeypatch):
    gate = _gate()
    if not os.path.exists(gate.EVAL_CFG_SRC):
        pytest.skip("evaluation twin config not present")
    monkeypatch.setattr(gate, "CFG", str(tmp_path / "cfg.yml"))
    import yaml
    cfg = yaml.safe_load(open(gate.build_config()))
    for tier in gate.TIERS:
        a, b = cfg[f"rg_{tier}"], cfg[f"rc_{tier}"]
        assert a["crd"]["forecast"]["source"] == "candidate_carbon_regret"
        assert b["crd"]["forecast"]["source"] == "instantaneous_carbon_cf"
        assert sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k)) == [
            "crd", "experiment_name", "simulation_name"]
