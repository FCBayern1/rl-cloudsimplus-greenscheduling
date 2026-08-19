#!/usr/bin/env python3
"""Aggregate one V3.2 training run into Gate-2/3 diagnostic evidence.

Inputs are behavior-neutral artifacts already emitted during training:
``v32_rollout_worker*.jsonl``, ``training_metrics.csv``, and worker monitor
CSVs.  Missing evidence remains JSON null and is printed as NOT AVAILABLE;
the script never manufactures a PASS from absent data.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import numpy as np


MIN_MONOTONIC_CELL_COUNT = 20
ALL_DEFER_THRESHOLD = 0.95
BACKSTOP_DOMINANT_THRESHOLD = 0.50
COMPLETION_FLOOR = 0.995
LATE_FRACTION = 0.40


def _finite(value) -> Optional[float]:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _weighted(values: Iterable[tuple[object, object]]) -> Optional[float]:
    total = 0.0
    count = 0.0
    for value, weight in values:
        value, weight = _finite(value), _finite(weight)
        if value is None or weight is None or weight <= 0:
            continue
        total += value * weight
        count += weight
    return total / count if count > 0 else None


def _late(rows):
    if not rows:
        return []
    start = max(0, int(math.floor(len(rows) * (1.0 - LATE_FRACTION))))
    return rows[start:]


def _sum_matrix(payloads, key, dtype=float):
    matrices = [np.asarray(p[key], dtype=dtype) for p in payloads if key in p]
    return np.sum(matrices, axis=0) if matrices else None


def _rate(num, den, rows, cols) -> Optional[float]:
    if num is None or den is None:
        return None
    n = float(num[np.ix_(rows, cols)].sum())
    d = float(den[np.ix_(rows, cols)].sum())
    return n / d if d > 0 else None


def _monotonic_fraction(counts, defers, *, axis: int) -> Optional[float]:
    if counts is None or defers is None:
        return None
    rates = np.divide(
        defers, counts, out=np.full(counts.shape, np.nan), where=counts > 0)
    good = total = 0
    outer = rates.shape[1 - axis]
    for fixed in range(outer):
        series = rates[:, fixed] if axis == 0 else rates[fixed, :]
        support = counts[:, fixed] if axis == 0 else counts[fixed, :]
        present = [
            i for i in range(len(series))
            if support[i] >= MIN_MONOTONIC_CELL_COUNT and np.isfinite(series[i])
        ]
        for left, right in zip(present, present[1:]):
            total += 1
            good += int(series[right] >= series[left] - 1e-12)
    return good / total if total else None


def load_rollouts(run_dir: Path):
    payloads = []
    for path in sorted(run_dir.glob("v32_rollout_worker*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                payloads.append(json.loads(line))
    return payloads


def summarize_rollouts(payloads):
    counts = _sum_matrix(payloads, "real_slot_count", np.int64)
    defers = _sum_matrix(payloads, "defer_count", np.int64)
    real_total = int(counts.sum()) if counts is not None else 0
    defer_total = int(defers.sum()) if defers is not None else 0
    # Real temporal contrast: high gain (>=.05) vs persistence/no-gain among
    # jobs with at least 900 s slack.  Holding slack support fixed prevents a
    # loose-job mix shift from masquerading as forecast sensitivity.
    high = _rate(defers, counts, list(range(3, 6)), list(range(3, 6)))
    zero = _rate(defers, counts, [0], list(range(3, 6)))
    rollout_delta = high - zero if high is not None and zero is not None else None

    waited = sum(int(p.get("waited_resolution_count", 0)) for p in payloads)
    predicted = sum(int(p.get("wait_predicted_positive_count", 0)) for p in payloads)
    improved = sum(int(p.get("wait_carbon_improved_count", 0)) for p in payloads)
    half = sum(int(p.get("wait_forecast_half_realized_count", 0)) for p in payloads)
    forced = sum(int(p.get("forced_route_count", 0)) for p in payloads)
    deferred_jobs = sum(int(p.get("deferred_job_count", 0)) for p in payloads)
    collisions = sum(int(p.get("job_key_collision_count", 0)) for p in payloads)
    pending = sum(int(p.get("pending_deferral_count", 0)) for p in payloads)
    wait_duration = _weighted(
        (p.get("mean_wait_duration_sec"), p.get("waited_resolution_count", 0))
        for p in payloads)
    predicted_gain = _weighted(
        (p.get("mean_predicted_gain_at_first_defer"), p.get("waited_resolution_count", 0))
        for p in payloads)
    observed_gain = _weighted(
        (p.get("mean_observed_gain_at_resolution"), p.get("waited_resolution_count", 0))
        for p in payloads)
    backstop_ratio = forced / (forced + waited) if forced + waited else None
    return {
        "episode_count": len(payloads),
        "real_slot_count": real_total,
        "defer_count": defer_total,
        "defer_rate": defer_total / real_total if real_total else None,
        "rollout_temporal_delta": rollout_delta,
        "high_gain_loose_slack_defer_rate": high,
        "zero_gain_loose_slack_defer_rate": zero,
        "gain_monotonic_fraction": _monotonic_fraction(counts, defers, axis=0),
        # More slack should permit more deferral (equivalently, defer decreases
        # as urgency increases).
        "slack_monotonic_fraction": _monotonic_fraction(counts, defers, axis=1),
        "deferred_job_count": deferred_jobs,
        "waited_resolution_count": waited,
        "wait_predicted_positive_count": predicted,
        "wait_carbon_improved_count": improved,
        "wait_carbon_improvement_rate": improved / predicted if predicted else None,
        "wait_forecast_half_realized_rate": half / predicted if predicted else None,
        "mean_wait_duration_sec": wait_duration,
        "mean_predicted_gain_at_first_defer": predicted_gain,
        "mean_observed_gain_at_resolution": observed_gain,
        "forced_route_count": forced,
        "backstop_resolution_ratio": backstop_ratio,
        "pending_deferral_count": pending,
        "job_key_collision_count": collisions,
        "all_defer": (defer_total / real_total >= ALL_DEFER_THRESHOLD)
        if real_total else None,
        "backstop_dominant": (backstop_ratio >= BACKSTOP_DOMINANT_THRESHOLD)
        if backstop_ratio is not None else None,
    }


def summarize_training(run_dir: Path):
    path = run_dir / "training_metrics.csv"
    if not path.exists():
        return {"defer_route_td_residual_ratio": None, "advantage_by_wait_sec": {}}
    rows = _late(list(csv.DictReader(path.open())))
    td_defer = _weighted(
        (r.get("global_v32_td_abs_defer"), r.get("global_v32_td_defer_count"))
        for r in rows)
    td_route = _weighted(
        (r.get("global_v32_td_abs_route"), r.get("global_v32_td_route_count"))
        for r in rows)
    ratio = td_defer / td_route if td_defer is not None and td_route not in (None, 0.0) else None
    waits = {}
    for label in ("0_60", "60_300", "300_900", "900_1800", "1800_3600", "gt3600"):
        waits[label.replace("_", "-")] = _weighted(
            (
                r.get(f"global_v32_adv_defer_wait_{label}"),
                r.get(f"global_v32_adv_defer_wait_{label}_count"),
            )
            for r in rows)
    return {
        "late_iteration_count": len(rows),
        "td_abs_defer": td_defer,
        "td_abs_route": td_route,
        "defer_route_td_residual_ratio": ratio,
        "advantage_defer": _weighted(
            (r.get("global_v32_adv_defer"), r.get("global_v32_adv_defer_count"))
            for r in rows),
        "advantage_route": _weighted(
            (r.get("global_v32_adv_route"), r.get("global_v32_adv_route_count"))
            for r in rows),
        "advantage_by_wait_sec": waits,
    }


def summarize_completion(run_dir: Path):
    values = []
    for path in sorted(run_dir.glob("monitor*.csv")):
        rows = _late(list(csv.DictReader(path.open())))
        values.extend(
            v for v in (_finite(r.get("completion_rate_mi")) for r in rows)
            if v is not None)
    if not values:
        return {"late_completion_median": None, "late_completion_p10": None,
                "completion_collapse": None}
    median = float(np.median(values))
    p10 = float(np.percentile(values, 10))
    return {
        "late_completion_median": median,
        "late_completion_p10": p10,
        "completion_collapse": median < COMPLETION_FLOOR,
    }


def load_probe_deltas(patterns):
    paths = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    deltas = []
    for path in sorted(set(paths)):
        data = json.loads(path.read_text())
        block = data.get("job_temporal") or data.get("temporal") or {}
        value = _finite(block.get("delta"))
        if value is not None:
            deltas.append(value)
    return deltas


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--probe-glob", action="append", default=[])
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    rollout = summarize_rollouts(load_rollouts(args.run_dir))
    learner = summarize_training(args.run_dir)
    completion = summarize_completion(args.run_dir)
    payload = {
        "schema": "v32_gate_run_summary_v1",
        "run_dir": str(args.run_dir.resolve()),
        "rollout": rollout,
        "learner": learner,
        "completion": completion,
        "late_checkpoint_temporal_deltas": load_probe_deltas(args.probe_glob),
        "gate3": {
            "defer_route_td_residual_ratio": learner["defer_route_td_residual_ratio"],
            "all_defer": rollout["all_defer"],
            "backstop_dominant": rollout["backstop_dominant"],
            "completion_collapse": completion["completion_collapse"],
        },
    }
    print(json.dumps(payload, indent=2, allow_nan=False))
    print("\nGATE-3 MATERIAL")
    for key, value in payload["gate3"].items():
        print(f"  {key}: {'NOT AVAILABLE' if value is None else value}")
    print(f"  wait_carbon_improvement_rate: "
          f"{rollout['wait_carbon_improvement_rate']}")
    print(f"  advantage_by_wait_sec: {learner['advantage_by_wait_sec']}")
    if args.json_out:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
