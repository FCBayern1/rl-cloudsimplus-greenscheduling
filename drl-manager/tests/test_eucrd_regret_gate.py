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


def _switch_judge():
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "switch_ab_judge", os.path.join(G1, "switch_ab_judge.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _write_calls(path, rows_):
    import json
    with open(path, "w") as f:
        for r in rows_:
            f.write(json.dumps(r) + "\n")


def _call(dw=0.05, dmax=0.2, firing=0.2, nonfiring=0.01, cos=0.99):
    return {"call": 1, "n_valid": 100, "n_firing": 5, "w_delta_mean": dw, "w_delta_max": dmax,
            "w_delta_mean_firing": firing, "w_delta_mean_nonfiring": nonfiring,
            "grad_cosine": cos, "grad_rel_l2": 0.1, "adv_cosine": 0.99,
            "frac_neg_damped": 0.6, "frac_pos_damped": 0.4,
            "w_ratio_neg_mean": 0.97, "w_ratio_pos_mean": 1.02,
            "n_neg_adv": 40, "n_pos_adv": 60,
            "top_changed": [{"delta": dmax, "adv_sign": -1, "firing": True}]}


def test_switch_judge_passes_a_targeted_change(tmp_path, monkeypatch):
    j = _switch_judge()
    monkeypatch.setattr(j, "OUT", str(tmp_path))
    monkeypatch.setattr(j, "DUMP", str(tmp_path / "d.jsonl"))
    _write_calls(j.DUMP, [_call(), _call()])
    res = j.judge()
    assert res["verdict"] == "SWITCH_AB_PASS"
    assert res["pooled"]["targeting_ratio"] == 20.0
    assert res["advantage_sign"]["frac_neg_damped_mean"] == 0.6


def test_switch_judge_names_each_failing_criterion(tmp_path, monkeypatch):
    j = _switch_judge()
    monkeypatch.setattr(j, "OUT", str(tmp_path))
    monkeypatch.setattr(j, "DUMP", str(tmp_path / "d.jsonl"))
    # inert weights, an untargeted change and an unchanged gradient
    _write_calls(j.DUMP, [_call(dw=1e-6, dmax=1e-6, firing=0.01, nonfiring=0.01, cos=1.0)])
    res = j.judge()
    for k in ("W1_weights_differ", "W2_change_is_targeted", "W3_gradient_differs"):
        assert k in res["verdict"]


def test_switch_judge_stops_when_nothing_was_recorded(tmp_path, monkeypatch):
    j = _switch_judge()
    monkeypatch.setattr(j, "OUT", str(tmp_path))
    monkeypatch.setattr(j, "DUMP", str(tmp_path / "missing.jsonl"))
    assert j.judge()["verdict"] == "STOP_SWITCH_AB:no_calls_recorded"


def test_switch_judge_pairs_each_call_against_its_own_null(tmp_path, monkeypatch):
    j = _switch_judge()
    monkeypatch.setattr(j, "OUT", str(tmp_path))
    monkeypatch.setattr(j, "DUMP", str(tmp_path / "d.jsonl"))
    # the pairing is read on the relative L2 (a magnitude diagnostic), never on the cosine
    a = _call(cos=0.99); a["grad_cosine_null"] = 1.0000001; a["grad_rel_l2"] = 0.1
    a["grad_rel_l2_null"] = 0.003
    b = _call(cos=0.9999); b["grad_cosine_null"] = 1.0; b["grad_rel_l2"] = 0.004
    b["grad_rel_l2_null"] = 0.003
    _write_calls(j.DUMP, [a, b])
    res = j.judge()
    p = res["paired_null"]
    assert p["calls"] == 2
    assert p["frac_calls_ab_exceeds_own_null"] == 1.0     # both A/B differences beat their null
    assert p["frac_calls_ab_exceeds_2x_own_null"] == 0.5  # only the first beats it by 2x
    assert p["rel_l2_ab_median"] > p["rel_l2_null_median"]
    assert p["resolvable"] is True
    # a cosine that rounds above 1 is clamped, and W3 is still decided on it
    assert res["pooled"]["grad_cosine_max"] <= 1.0


def test_switch_judge_reports_the_decomposition_without_gating_on_it(tmp_path, monkeypatch):
    j = _switch_judge()
    monkeypatch.setattr(j, "OUT", str(tmp_path))
    monkeypatch.setattr(j, "DUMP", str(tmp_path / "d.jsonl"))
    rows_ = []
    for i in range(3):
        r = _call(cos=0.99999)                      # total gradient barely moves ...
        r.update({"grad_rel_l2": 0.0006, "grad_rel_l2_null": 0.0, "grad_cosine_null": 1.0,
                  "pi_cosine": 0.95, "pi_rel_l2": 0.3, "pi_cosine_null": 1.0,
                  "pi_rel_l2_null": 0.0, "pi_norm_a": 1.0, "pi_norm_b": 1.1,
                  "vf_norm_a": 50.0, "entkl_norm_a": 2.0, "pi_share_of_total_norm": 0.02,
                  "pi_delta_norm": 0.3, "total_delta_norm": 0.3})
        rows_.append(r)
    _write_calls(j.DUMP, rows_)
    res = j.judge()
    d = res["decomposition"]
    assert d["calls"] == 2                            # the first call is excluded
    assert d["pi_resolvable"] is True
    assert d["dilution_ratio_median"] == 0.3 / 0.0006  # ... while the surrogate moves a lot
    assert d["vf_norm_median"] == 50.0
    # the decomposition never changes the frozen verdict
    assert res["gates"]["W3_gradient_differs"] is False
