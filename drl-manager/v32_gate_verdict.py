#!/usr/bin/env python3
"""Deterministic Gate 0--5 verdicts for the pre-registered V3.2 ladder.

The input is one JSON evidence ledger. Missing fields produce WAIT, never PASS.
Thresholds are literals here so a later run cannot move the goalposts by editing
an experiment config. Use ``--write-template`` to emit the accepted schema.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PASS, FAIL, WAIT = "PASS", "FAIL", "WAIT"


TEMPLATE: Dict[str, Any] = {
    "gate0": {
        "result_config_seeds": [1, 2],
        "same_seed_machine_offset": True,
        "arm_diff_only_forecast": True,
    },
    "gate1": {
        "legacy_direct_edge_abs": 0.0,
        "v32_forecast_gradient_abs": 0.0,
        "v32_slack_gradient_abs": 0.0,
        "same_shape_parameter_count_mask": True,
    },
    "gate2": {
        "synthetic_temporal_delta": 0.0,
        "rollout_temporal_delta": 0.0,
        "forecast_perturbation_above_null": False,
        "gain_monotonic": False,
        "slack_monotonic": False,
    },
    "gate3": {
        "late_checkpoint_temporal_deltas_by_seed": {"1": [], "2": []},
        "defer_route_td_residual_ratio": 0.0,
        "all_defer": False,
        "backstop_dominant": False,
        "completion_collapse": False,
    },
    "gate4": {
        "oracle_completion": 0.0,
        "blind_completion": 0.0,
        "oracle_carbon_per_mi": 0.0,
        "blind_carbon_per_mi": 0.0,
        "forecast_gain_slack_behavior": False,
    },
    "gate5": {
        "base_oracle_carbon_per_mi": 0.0,
        "anti_forecast_carbon_per_mi": 0.0,
        "anti_forecast_completion": 0.0,
    },
}


def _missing(block: Dict[str, Any], names: Iterable[str]) -> List[str]:
    return [name for name in names if name not in block]


def _result(status: str, reason: str) -> Dict[str, str]:
    return {"status": status, "reason": reason}


def _true(value: Any) -> bool:
    """JSON booleans only; strings such as ``"false"`` must never pass."""
    return value is True


def _false(value: Any) -> bool:
    return value is False


def evaluate_gates(evidence: Dict[str, Any]) -> Dict[str, Dict[str, str]]:
    out: Dict[str, Dict[str, str]] = {}

    g = evidence.get("gate0", {})
    req = ("result_config_seeds", "same_seed_machine_offset", "arm_diff_only_forecast")
    if miss := _missing(g, req):
        out["gate0"] = _result(WAIT, f"missing {miss}")
    else:
        seeds = list(g["result_config_seeds"])
        ok = (
            len(seeds) == 2
            and set(seeds) == {1, 2}
            and _true(g["same_seed_machine_offset"])
            and _true(g["arm_diff_only_forecast"])
        )
        out["gate0"] = _result(PASS if ok else FAIL, "seed/config integrity")

    g = evidence.get("gate1", {})
    req = (
        "legacy_direct_edge_abs",
        "v32_forecast_gradient_abs",
        "v32_slack_gradient_abs",
        "same_shape_parameter_count_mask",
    )
    if miss := _missing(g, req):
        out["gate1"] = _result(WAIT, f"missing {miss}")
    else:
        ok = (
            float(g["legacy_direct_edge_abs"]) <= 1e-4
            and float(g["v32_forecast_gradient_abs"]) > 1e-8
            and float(g["v32_slack_gradient_abs"]) > 1e-8
            and _true(g["same_shape_parameter_count_mask"])
        )
        out["gate1"] = _result(PASS if ok else FAIL, "direct-edge/shape contract")

    g = evidence.get("gate2", {})
    req = (
        "synthetic_temporal_delta",
        "rollout_temporal_delta",
        "forecast_perturbation_above_null",
        "gain_monotonic",
        "slack_monotonic",
    )
    if miss := _missing(g, req):
        out["gate2"] = _result(WAIT, f"missing {miss}")
    else:
        syn = float(g["synthetic_temporal_delta"])
        real = float(g["rollout_temporal_delta"])
        ok = (
            syn > 0.0
            and real > 0.0
            and min(syn, real) >= 0.05
            and _true(g["forecast_perturbation_above_null"])
            and _true(g["gain_monotonic"])
            and _true(g["slack_monotonic"])
        )
        out["gate2"] = _result(PASS if ok else FAIL, "100k behavior threshold delta>=0.05")

    g = evidence.get("gate3", {})
    req = (
        "late_checkpoint_temporal_deltas_by_seed",
        "defer_route_td_residual_ratio",
        "all_defer",
        "backstop_dominant",
        "completion_collapse",
    )
    if miss := _missing(g, req):
        out["gate3"] = _result(WAIT, f"missing {miss}")
    else:
        series = g["late_checkpoint_temporal_deltas_by_seed"]
        values = (
            [list(series[key]) for key in ("1", "2")]
            if isinstance(series, dict) and set(series) == {"1", "2"}
            else []
        )
        ok = (
            len(values) >= 2
            and all(v and all(float(x) > 0.0 for x in v) for v in values)
            and float(g["defer_route_td_residual_ratio"]) <= 3.0
            and _false(g["all_defer"])
            and _false(g["backstop_dominant"])
            and _false(g["completion_collapse"])
        )
        out["gate3"] = _result(PASS if ok else FAIL, "300k two-seed stability")

    g = evidence.get("gate4", {})
    req = (
        "oracle_completion",
        "blind_completion",
        "oracle_carbon_per_mi",
        "blind_carbon_per_mi",
        "forecast_gain_slack_behavior",
    )
    if miss := _missing(g, req):
        out["gate4"] = _result(WAIT, f"missing {miss}")
    else:
        oracle_c = float(g["oracle_carbon_per_mi"])
        blind_c = float(g["blind_carbon_per_mi"])
        saving = (blind_c - oracle_c) / blind_c if blind_c > 0.0 else float("-inf")
        ok = (
            float(g["oracle_completion"]) >= 0.995
            and float(g["blind_completion"]) >= 0.995
            and saving > 0.13
            and _true(g["forecast_gain_slack_behavior"])
        )
        out["gate4"] = _result(PASS if ok else FAIL, f"iso-completion carbon saving={saving:.4f}")

    g = evidence.get("gate5", {})
    req = (
        "base_oracle_carbon_per_mi",
        "anti_forecast_carbon_per_mi",
        "anti_forecast_completion",
    )
    if miss := _missing(g, req):
        out["gate5"] = _result(WAIT, f"missing {miss}")
    else:
        base = float(g["base_oracle_carbon_per_mi"])
        anti = float(g["anti_forecast_carbon_per_mi"])
        pain = (anti - base) / base if base > 0.0 else float("-inf")
        ok = pain >= 0.10 and float(g["anti_forecast_completion"]) >= 0.995
        out["gate5"] = _result(PASS if ok else FAIL, f"anti-forecast carbon pain={pain:.4f}")

    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence", type=Path)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--write-template", type=Path)
    args = parser.parse_args()
    if args.write_template:
        args.write_template.write_text(json.dumps(TEMPLATE, indent=2) + "\n")
        return
    if not args.evidence:
        parser.error("--evidence is required unless --write-template is used")
    evidence = json.loads(args.evidence.read_text())
    verdicts = evaluate_gates(evidence)
    for gate in (f"gate{i}" for i in range(6)):
        row = verdicts[gate]
        print(f"{gate.upper()}: {row['status']} — {row['reason']}")
    if args.json_out:
        args.json_out.write_text(json.dumps(verdicts, indent=2) + "\n")


if __name__ == "__main__":
    main()
