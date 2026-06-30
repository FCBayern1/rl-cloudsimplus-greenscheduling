#!/usr/bin/env python3
"""Robustness gate: does the TEMPORAL defer lever help on top of MULTIPLE routing baselines?

For each global routing baseline S (round_robin, green_aware, min_brown_power, ga, pso, ...):
  - base  = S routes WHERE + drain every step (run-on-arrival).
  - defer = S routes WHERE, but while total green < threshold we override to DEFER (run-later);
            drain when green is present.
Same env + same seed for every arm, so the ONLY difference within a pair is the temporal gate.

If Δcarbon<0 / Δgreen_used>0 holds across ALL baselines — including the already-green-AWARE ones
(green_aware, min_brown_power) — then defer adds genuine TEMPORAL value beyond spatial green routing,
and the result is not an artifact of one weak baseline.

Run from drl-manager/:
  .venv/bin/python oracle_baselines_defer.py --schedulers round_robin green_aware min_brown_power ga pso \
      --experiment experiment_multi_5dc_carbon_v2_deferrable_gdpd --route-thresh-w 300
"""
import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

# Silence the env's per-step INFO logging ("Calling Java with global_actions=[...]") — it
# floods the log (54k+ lines) and badly slows GA/PSO. Keep only warnings/errors.
logging.disable(logging.INFO)

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import collect_metrics
from src.baselines.global_schedulers import GLOBAL_SCHEDULERS
from oracle_diurnal_defer import green_total_now

GREEN_DCS = [0, 1, 2]


def run(env, scheduler_name, defer: bool, seed: int, drain: int, thresh: float) -> Dict[str, Any]:
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    defer_idx = num_dc
    sched = GLOBAL_SCHEDULERS[scheduler_name](num_dc, batch)  # fresh (stateful) per arm
    obs, info = env.reset(seed=seed)
    done = False
    n_defer = 0
    pow_day = pow_night = n_day = n_night = 0.0
    while not done:
        g = obs["global"]
        draw = float(np.sum(np.asarray(g.get("dc_current_power_w", []), dtype=float).ravel()[:num_dc]))
        light = green_total_now(g, GREEN_DCS, num_dc) >= thresh
        if light: pow_day += draw; n_day += 1
        else:     pow_night += draw; n_night += 1
        if defer and not light:
            global_action = [defer_idx] * batch           # temporal hold while calm
            n_defer += batch
        else:
            global_action = list(sched.schedule(g))        # baseline decides WHERE
        local_actions = {dc: drain for dc in range(num_dc)}
        obs, _, term, trunc, info = env.step({"global": global_action, "local": local_actions})
        done = term or trunc
    m = collect_metrics(info, num_dc)
    m["_defer"] = n_defer
    m["_day_W"] = pow_day / max(1, n_day)
    m["_night_W"] = pow_night / max(1, n_night)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_deferrable_gdpd")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--schedulers", nargs="+",
                    default=["round_robin", "green_aware", "min_brown_power", "ga", "pso"])
    ap.add_argument("--route-thresh-w", type=float, default=300.0)
    ap.add_argument("--trace", default=None)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_baselines")
    cfg.setdefault("output_dir", "/tmp/oracle_baselines")
    os.makedirs("/tmp/oracle_baselines", exist_ok=True)
    if args.trace:
        cfg["cloudlet_trace_file"] = args.trace
    drain = int(cfg.get("max_dispatch_per_step", 64))

    print(f"=== Defer robustness across baselines ({args.experiment}, thresh={args.route_thresh_w}W, "
          f"trace={cfg.get('cloudlet_trace_file')}, seed={args.seed}) ===")
    env = HierarchicalMultiDCEnv(config=cfg)
    rows = []
    for name in args.schedulers:
        if name not in GLOBAL_SCHEDULERS:
            print(f"  skip unknown scheduler '{name}'"); continue
        print(f"\n--- {name} ---")
        b = run(env, name, defer=False, seed=args.seed, drain=drain, thresh=args.route_thresh_w)
        d = run(env, name, defer=True, seed=args.seed, drain=drain, thresh=args.route_thresh_w)
        rows.append((name, b, d))
        print(f"  base : carbon {b['total_carbon_kg']:.4f}  compl {b.get('completion_rate_mi',0):.3f}  "
              f"green_used {b['green_used_wh']:.0f}  green_ratio {b.get('green_ratio',0):.3f}")
        print(f"  defer: carbon {d['total_carbon_kg']:.4f}  compl {d.get('completion_rate_mi',0):.3f}  "
              f"green_used {d['green_used_wh']:.0f}  green_ratio {d.get('green_ratio',0):.3f}  "
              f"(day {b['_day_W']:.0f}->{d['_day_W']:.0f} night {b['_night_W']:.0f}->{d['_night_W']:.0f})")
    env.close()

    print("\n=== SUMMARY: does defer help on top of each router? ===")
    print(f"{'router':18}{'base_C':>9}{'defer_C':>9}{'ΔC%':>8}{'Δgreen_used':>13}{'base_cmpl':>10}{'defer_cmpl':>11}")
    for name, b, d in rows:
        dc = 100 * (d['total_carbon_kg'] - b['total_carbon_kg']) / max(1e-9, b['total_carbon_kg'])
        dgu = d['green_used_wh'] - b['green_used_wh']
        flag = "✅" if (dc < -1 and dgu > 1) else ("~" if dc < -1 else "✗")
        print(f"{name:18}{b['total_carbon_kg']:>9.4f}{d['total_carbon_kg']:>9.4f}{dc:>7.1f}%{dgu:>12.0f}  "
              f"{b.get('completion_rate_mi',0):>9.3f}{d.get('completion_rate_mi',0):>10.3f}  {flag}")
    print("\n✅ = carbon down AND green_used up (real green capture). ~ = carbon down but no extra green "
          "(likely 'do less'). ✗ = no carbon gain.")


if __name__ == "__main__":
    main()
