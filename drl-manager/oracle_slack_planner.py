#!/usr/bin/env python3
"""Slack-aware per-task oracle — the scenario upper-bound instrument (v2).

Replaces oracle_hold_until_green.py as the track-0 instrument. That script was
falsified twice on v3 (docs/V3_FORECAST_DIAGNOSIS.md §7.4): its hold rule is a
DC-level binary (thresholds 0.3→hold every step, 0.6→never hold, nothing in
between on synchronized green), and its local layer treats a BestFit VM INDEX
as a dispatch_rate RELEASE COUNT, so even its zero-hold arm diverges from the
no-hold baseline.

This planner follows the V3.1 protocol instead:
  global : per-SLOT route-or-defer. A slot defers iff
             (1) total green now < theta * best total green reachable within
                 this job's wait budget (true future, see cheat below),
             (2) wait budget = time_to_deadline - est_runtime - margin > 0,
             (3) deferred backlog below cap.
           Routed slots go to the greenest-now DC with free capacity.
  local  : deterministic drain (max legal dispatch from the live mask),
           matching fixed_local_scheduler=drain.
  future : GODEYE BY DESIGN. The planner reads the wind CSVs directly and
           reconstructs absolute rows (single episode per env => episode
           offset 0, warmup 13 rows, 1 row = 1 sim second in COMPRESSED).
           An upper-bound instrument is allowed to cheat; RL arms are not.

Verdict rule (pre-registered, diagnosis §7.4): compare against the no-defer
baseline (same script, defer disabled, same seed/offsets, drain local). Only
if THIS instrument cannot save >=10% carbon at completion >=99.5% may the
scenario be called forecast-valueless.

Usage:
    .venv/bin/python oracle_slack_planner.py --experiment experiment_v3_1_oracle \
        [--theta 0.7] [--margin 120] [--backlog-cap 400] [--seed 1]
"""
import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import load_config, collect_metrics

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources"
WARMUP_ROWS = 13          # sim_step 0 == CSV row 13 (GreenEnergyProvider warmup)
VM_MIPS = 40000.0


def _arr(obs, key, n):
    v = np.asarray(obs.get(key, []), dtype=np.float64).ravel()
    if v.size < n:
        v = np.concatenate([v, np.zeros(n - v.size)])
    return v[:n]


def load_green_series(cfg) -> np.ndarray:
    """Total green power (W, compressed scale) per absolute CSV row."""
    divisor = float(cfg["compressed_power_divisor"])
    tot = None
    for d in cfg["datacenters"]:
        for t in d.get("turbine_ids") or []:
            p = GATE / f"windProduction/simplified/Turbine_{t}_2021.csv"
            v = np.array([float(r["power_kw"]) for r in csv.DictReader(open(p))])
            tot = v if tot is None else tot + v
    return tot * 1000.0 / divisor


def drain_action(mask) -> int:
    legal = np.flatnonzero(np.asarray(mask, dtype=bool).ravel())
    return int(legal.max()) if legal.size else 0


def run(env, cfg, green: np.ndarray, defer_enabled: bool, theta: float,
        margin: float, backlog_cap: int, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    defer_idx = num_dc
    ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec",
                                       cfg.get("defer_urgency_window_sec", 3600.0))))
    done, t, defers, routes = False, 0, 0, 0
    while not done:
        g = obs["global"]
        row = WARMUP_ROWS + t                       # single episode -> offset 0
        green_now = green[min(row, len(green) - 1)]
        mi = _arr(g, "batch_cloudlet_mi", batch)
        # v31 keys are normalized: time_to_deadline in units of urgency_window
        ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
        present = _arr(g, "batch_cloudlet_deadline_present", batch)
        backlog = int(_arr(g, "global_deferred_count", 1)[0]
                      * float(cfg.get("obs_v31_global_deferred_count_scale", 2000.0)))
        # greenest-now DC with capacity, else greenest overall
        green_dc = _arr(g, "dc_current_green_power_w", num_dc)
        pes_free = _arr(g, "dc_available_pes", num_dc)
        order = np.argsort(-green_dc)
        target = int(order[0])
        for d in order:
            if pes_free[int(d)] >= 1:
                target = int(d)
                break
        actions: List[int] = []
        for i in range(batch):
            if mi[i] <= 0:
                actions.append(target)              # padded slot, value ignored
                continue
            if not defer_enabled:
                actions.append(target); routes += 1
                continue
            runtime = mi[i] / VM_MIPS
            budget = (ttd[i] - runtime - margin) if present[i] > 0.5 else 0.0
            if budget <= 0 or backlog >= backlog_cap:
                actions.append(target); routes += 1
                continue
            horizon = int(min(budget, 3600))
            best_future = green[row:min(row + horizon, len(green))].max(initial=green_now)
            if green_now < theta * best_future:
                actions.append(defer_idx); defers += 1
            else:
                actions.append(target); routes += 1
        local_actions = {dc: drain_action(env.get_local_action_masks(dc))
                         for dc in range(num_dc)}
        obs, _, term, trunc, info = env.step({"global": actions, "local": local_actions})
        done = term or trunc
        t += 1
    m = collect_metrics(info, num_dc)
    m["_defer_slots"], m["_route_slots"] = defers, routes
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_v3_1_oracle")
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--margin", type=float, default=120.0)
    ap.add_argument("--backlog-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()

    cfg = load_config(args.experiment)
    # Standalone-env boot keys (the 10:24 crash): common carries a FIXED
    # py4j_port, which makes the env try to CONNECT instead of auto-launching
    # a fresh gateway on a free port. Same recipe as oracle_hold_until_green.
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    Path("/tmp/oracle_gateway").mkdir(parents=True, exist_ok=True)
    if not cfg.get("obs_v31_features"):
        sys.exit("needs obs_v31_features=true (per-slot deadline features)")
    green = load_green_series(cfg)

    print(f"=== slack-aware oracle vs no-defer baseline ({args.experiment}, "
          f"theta={args.theta}, margin={args.margin}s, cap={args.backlog_cap}) ===")
    results = {}
    for name, defer_on in (("no-defer", False), ("slack-aware", True)):
        env = HierarchicalMultiDCEnv(cfg)
        try:
            results[name] = run(env, cfg, green, defer_on, args.theta,
                                args.margin, args.backlog_cap, args.seed)
        finally:
            env.close()
        r = results[name]
        done_rate = r.get("finished_over_received_rate", r.get("completion_rate", 0))
        print(f"{name:12s} carbon_kg={r.get('total_carbon_kg', 0):.4f} "
              f"completion={done_rate} green_ratio={r.get('green_ratio', 0):.4f} "
              f"defers={r.get('_defer_slots', 0)} routes={r.get('_route_slots', 0)}")
    a, b = results["no-defer"], results["slack-aware"]
    if a.get("total_carbon_kg"):
        dc = (b["total_carbon_kg"] - a["total_carbon_kg"]) / a["total_carbon_kg"]
        print(f"\nDelta carbon = {dc:+.1%}  (verdict rule: <= -10% at completion "
              f">=99.5% keeps the scenario alive; diagnosis §7.4)")


if __name__ == "__main__":
    main()
