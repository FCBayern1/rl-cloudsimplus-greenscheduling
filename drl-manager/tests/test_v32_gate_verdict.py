"""Boundary tests for the immutable V3.2 Gate 0--5 judge."""

import copy
import sys
from pathlib import Path

DRL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DRL_ROOT))

from v32_gate_verdict import FAIL, PASS, WAIT, TEMPLATE, evaluate_gates


def _passing_evidence():
    evidence = copy.deepcopy(TEMPLATE)
    evidence["gate1"].update({
        "v32_forecast_gradient_abs": 1e-5,
        "v32_slack_gradient_abs": 1e-5,
    })
    evidence["gate2"].update({
        "synthetic_temporal_delta": 0.05,
        "rollout_temporal_delta": 0.06,
        "forecast_perturbation_above_null": True,
        "gain_monotonic": True,
        "slack_monotonic": True,
    })
    evidence["gate3"].update({
        "late_checkpoint_temporal_deltas_by_seed": {
            "1": [0.01, 0.02], "2": [0.03, 0.01],
        },
        "defer_route_td_residual_ratio": 3.0,
    })
    evidence["gate4"].update({
        "oracle_completion": 0.995,
        "blind_completion": 0.996,
        "oracle_carbon_per_mi": 0.80,
        "blind_carbon_per_mi": 1.00,
        "forecast_gain_slack_behavior": True,
    })
    evidence["gate5"].update({
        "base_oracle_carbon_per_mi": 1.00,
        "anti_forecast_carbon_per_mi": 1.10,
        "anti_forecast_completion": 0.995,
    })
    return evidence


def test_all_preregistered_boundaries_can_pass():
    verdicts = evaluate_gates(_passing_evidence())
    assert {row["status"] for row in verdicts.values()} == {PASS}


def test_gate0_requires_exact_real_seeds_one_and_two():
    evidence = _passing_evidence()
    evidence["gate0"]["result_config_seeds"] = [7, 8]
    assert evaluate_gates(evidence)["gate0"]["status"] == FAIL


def test_gate2_rejects_delta_below_frozen_point_zero_five():
    evidence = _passing_evidence()
    evidence["gate2"]["rollout_temporal_delta"] = 0.049999
    assert evaluate_gates(evidence)["gate2"]["status"] == FAIL


def test_gate4_requires_strictly_more_than_thirteen_percent():
    evidence = _passing_evidence()
    evidence["gate4"]["oracle_carbon_per_mi"] = 0.87
    assert evaluate_gates(evidence)["gate4"]["status"] == FAIL


def test_missing_evidence_waits_instead_of_passing_or_crashing():
    verdicts = evaluate_gates({})
    assert all(row["status"] == WAIT for row in verdicts.values())


def test_string_booleans_and_wrong_seed_labels_cannot_pass():
    evidence = _passing_evidence()
    evidence["gate0"]["same_seed_machine_offset"] = "true"
    evidence["gate3"]["late_checkpoint_temporal_deltas_by_seed"] = {
        "7": [0.1], "8": [0.1],
    }
    verdicts = evaluate_gates(evidence)
    assert verdicts["gate0"]["status"] == FAIL
    assert verdicts["gate3"]["status"] == FAIL
