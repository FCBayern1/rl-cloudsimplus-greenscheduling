"""Pure V3.2 rollout bucketing helpers used by the RLlib callback.

The environment emits bounded job-aligned forecast gain and normalized
time-to-deadline.  This module aligns those decision-time observations with the
sampled per-slot actions and reconstructs the factorized temporal raw log-odds
from RLlib's categorical inputs.  It performs no I/O and is unit-testable
without Ray workers or a simulator.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np


GAIN_EDGES = np.asarray([0.0, 0.01, 0.05, 0.10, 0.25, np.inf], dtype=np.float64)
GAIN_LABELS = ["zero", "0-.01", ".01-.05", ".05-.10", ".10-.25", ">.25"]
SLACK_EDGES_SEC = np.asarray([0.0, 300.0, 900.0, 1800.0, 3600.0, np.inf])
SLACK_LABELS = ["overdue", "0-300", "300-900", "900-1800", "1800-3600", ">3600"]


def new_v32_rollout_accumulator() -> Dict[str, Any]:
    shape = (len(GAIN_LABELS), len(SLACK_LABELS))
    return {
        "schema": "v32_rollout_bins_v1",
        "gain_edges": GAIN_EDGES.tolist(),
        "gain_labels": list(GAIN_LABELS),
        "slack_edges_sec": SLACK_EDGES_SEC.tolist(),
        "slack_labels": list(SLACK_LABELS),
        "real_slot_count": np.zeros(shape, dtype=np.int64),
        "defer_count": np.zeros(shape, dtype=np.int64),
        "gate_logit_sum": np.zeros(shape, dtype=np.float64),
        "gate_logit_count": np.zeros(shape, dtype=np.int64),
        "resolution_carbon_kg_sum": np.zeros(shape, dtype=np.float64),
        "resolution_carbon_count": np.zeros(shape, dtype=np.int64),
        "step_count": 0,
    }


def _bucket(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    # Exact zero has its own pre-registered bin. Positive values use the
    # remaining right-closed ranges; negative slack is "overdue".
    out = np.searchsorted(edges, values, side="left")
    return np.clip(out, 0, len(edges) - 1).astype(np.int64)


def factorized_gate_logit(action_dist_inputs: Any, num_slots: int) -> Optional[np.ndarray]:
    """Recover logit(p_hold) from [route log-probs | defer log-prob]."""
    if action_dist_inputs is None or num_slots <= 0:
        return None
    logits = np.asarray(action_dist_inputs, dtype=np.float64).reshape(-1)
    if logits.size % num_slots != 0:
        return None
    choices = logits.size // num_slots
    if choices < 2:
        return None
    z = logits.reshape(num_slots, choices)
    route = z[:, :-1]
    route_max = np.max(route, axis=1, keepdims=True)
    log_route_mass = (
        route_max[:, 0]
        + np.log(np.exp(route - route_max).sum(axis=1))
    )
    return z[:, -1] - log_route_mass


def accumulate_v32_rollout_step(
    acc: Dict[str, Any],
    observation: Dict[str, Any],
    actions: Any,
    *,
    num_datacenters: int,
    deadline_scale_sec: float,
    action_dist_inputs: Any = None,
    resolution_carbon_kg_by_slot: Any = None,
) -> bool:
    """Accumulate one aligned global decision; return False if not V3.2 obs."""
    inner = observation.get("observation", observation)
    required = {
        "batch_cloudlet_mi",
        "batch_cloudlet_forecast_gain",
        "batch_cloudlet_time_to_deadline",
    }
    if not isinstance(inner, dict) or not required.issubset(inner):
        return False

    mi = np.asarray(inner["batch_cloudlet_mi"], dtype=np.float64).reshape(-1)
    gain = np.asarray(
        inner["batch_cloudlet_forecast_gain"], dtype=np.float64).reshape(-1)
    slack_sec = np.asarray(
        inner["batch_cloudlet_time_to_deadline"], dtype=np.float64).reshape(-1)
    slack_sec = slack_sec * max(1.0, float(deadline_scale_sec))
    action = np.asarray(actions, dtype=np.int64).reshape(-1)
    n = min(mi.size, gain.size, slack_sec.size, action.size)
    if n == 0:
        return False
    gate = factorized_gate_logit(action_dist_inputs, n)
    resolution_carbon = (
        None
        if resolution_carbon_kg_by_slot is None
        else np.asarray(resolution_carbon_kg_by_slot, dtype=np.float64).reshape(-1)
    )
    real = np.flatnonzero(mi[:n] > 0.0)
    if real.size == 0:
        acc["step_count"] += 1
        return True

    gain_bin = _bucket(gain[:n], GAIN_EDGES)
    slack_bin = _bucket(slack_sec[:n], SLACK_EDGES_SEC)
    for slot in real:
        cell = (int(gain_bin[slot]), int(slack_bin[slot]))
        acc["real_slot_count"][cell] += 1
        if int(action[slot]) == int(num_datacenters):
            acc["defer_count"][cell] += 1
        if gate is not None and slot < gate.size and np.isfinite(gate[slot]):
            acc["gate_logit_sum"][cell] += float(gate[slot])
            acc["gate_logit_count"][cell] += 1
        if (
            resolution_carbon is not None
            and slot < resolution_carbon.size
            and np.isfinite(resolution_carbon[slot])
        ):
            acc["resolution_carbon_kg_sum"][cell] += float(
                resolution_carbon[slot])
            acc["resolution_carbon_count"][cell] += 1
    acc["step_count"] += 1
    return True


def finalize_v32_rollout(
    acc: Dict[str, Any], *, forced_route_count: int = 0
) -> Dict[str, Any]:
    counts = np.asarray(acc["real_slot_count"], dtype=np.int64)
    defers = np.asarray(acc["defer_count"], dtype=np.int64)
    logit_sum = np.asarray(acc["gate_logit_sum"], dtype=np.float64)
    logit_count = np.asarray(acc["gate_logit_count"], dtype=np.int64)
    carbon_sum = np.asarray(acc["resolution_carbon_kg_sum"], dtype=np.float64)
    carbon_count = np.asarray(acc["resolution_carbon_count"], dtype=np.int64)
    defer_rate = np.divide(
        defers,
        counts,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=counts > 0,
    )
    gate_mean = np.divide(
        logit_sum,
        logit_count,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=logit_count > 0,
    )
    carbon_mean = np.divide(
        carbon_sum,
        carbon_count,
        out=np.full(counts.shape, np.nan, dtype=np.float64),
        where=carbon_count > 0,
    )
    def _nullable(matrix: np.ndarray):
        return [
            [None if not np.isfinite(value) else float(value) for value in row]
            for row in matrix
        ]
    return {
        "schema": acc["schema"],
        "gain_labels": list(acc["gain_labels"]),
        "slack_labels": list(acc["slack_labels"]),
        "step_count": int(acc["step_count"]),
        "real_slot_count": counts.tolist(),
        "defer_count": defers.tolist(),
        "defer_rate": _nullable(defer_rate),
        "raw_gate_logit_mean": _nullable(gate_mean),
        "raw_gate_logit_count": logit_count.tolist(),
        # Persistence billing at the policy-selected route time. Forced-route
        # resolutions have no observable chosen-DC action here and remain in
        # the separate forced_route_count instead of being misattributed.
        "resolution_carbon_kg_sum": carbon_sum.tolist(),
        "resolution_carbon_kg_mean": _nullable(carbon_mean),
        "resolution_carbon_count": carbon_count.tolist(),
        "resolution_carbon_kg_total": float(carbon_sum.sum()),
        "forced_route_count": int(forced_route_count),
    }


def resolution_carbon_kg_by_slot(
    observation: Dict[str, Any],
    actions: Any,
    *,
    num_datacenters: int,
    green_carbon_factors: Any,
    brown_carbon_factors: Any,
    mi_per_kg_factor: float,
) -> Optional[np.ndarray]:
    """Persistence-billed carbon for slots resolved to a selected DC.

    Deferred/padded/invalid slots are NaN. This mirrors the environment's
    current-green/current-demand effective-factor equation and therefore does
    not inspect future information.
    """
    inner = observation.get("observation", observation)
    required = {
        "batch_cloudlet_mi",
        "dc_current_green_power_w",
        "dc_current_power_w",
    }
    if not isinstance(inner, dict) or not required.issubset(inner):
        return None
    mi = np.asarray(inner["batch_cloudlet_mi"], dtype=np.float64).reshape(-1)
    action = np.asarray(actions, dtype=np.int64).reshape(-1)
    green = np.asarray(
        inner["dc_current_green_power_w"], dtype=np.float64).reshape(-1)
    demand = np.asarray(inner["dc_current_power_w"], dtype=np.float64).reshape(-1)
    gf = np.asarray(green_carbon_factors, dtype=np.float64).reshape(-1)
    bf = np.asarray(brown_carbon_factors, dtype=np.float64).reshape(-1)
    if min(green.size, demand.size, gf.size, bf.size) < num_datacenters:
        return None
    ratio = np.where(
        demand[:num_datacenters] > 1e-9,
        np.minimum(
            1.0,
            np.maximum(green[:num_datacenters], 0.0)
            / np.maximum(demand[:num_datacenters], 1e-9),
        ),
        np.where(green[:num_datacenters] > 0.0, 1.0, 0.0),
    )
    factor = ratio * gf[:num_datacenters] + (1.0 - ratio) * bf[:num_datacenters]
    out = np.full(mi.shape, np.nan, dtype=np.float64)
    n = min(mi.size, action.size)
    for slot in range(n):
        dc = int(action[slot])
        if mi[slot] > 0.0 and 0 <= dc < num_datacenters:
            out[slot] = (
                mi[slot] / max(1e3, float(mi_per_kg_factor)) * factor[dc]
            )
    return out
