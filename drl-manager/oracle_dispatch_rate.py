#!/usr/bin/env python3
"""
NL-3 oracle — validate the new dispatch-rate local agent. 2026-06-18.

Runs in local_dispatch_mode=dispatch_rate. Two scripted policies on the same
env/seed:
  - "drain"      : dispatch max every step (no green timing) → tests whether the
                   new path UNTHROTTLES throughput (completion/util should jump
                   from the legacy ~0.18 / 13% util).
  - "green_timed": per DC — green now → burst-dispatch (capture it); no green now
                   but forecast says green coming → dispatch 0 (HOLD/defer); no
                   green & none coming → drain (run on brown). Tests whether
                   shifting work into green periods reduces waste/carbon.

Decision gates:
  drain completion/util ≫ legacy 0.18 → throughput bottleneck fixed (NL-1+NL-2 work).
  green_timed waste < drain waste (with green/demand ~1× via --green-divisor)
      → the temporal lever finally moves waste → proceed to NL-4 (RL).

Run from drl-manager/:
  .venv/bin/python oracle_dispatch_rate.py --green-divisor 2000 --max-dispatch 50
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import collect_metrics


def _arr(obs, key, n):
    v = np.asarray(obs.get(key, []), dtype=np.float64).ravel()
    return np.concatenate([v, np.zeros(max(0, n - v.size))])[:n]


def greenest_dc(g, num_dc, batch):
    gn = _arr(g, "dc_current_green_power_w", num_dc)
    fut = _arr(g, "dc_future_short_mean", num_dc)
    return [int(np.argmax(gn / (gn.max() + 1e-9) + fut))] * batch


def run(env, policy, max_dispatch, hold_thresh, seed):
    obs, info = env.reset(seed=seed)
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    pow_sum = np.zeros(num_dc); green_sum = np.zeros(num_dc); nstep = 0
    n_hold = 0
    done = False
    while not done:
        g = obs["global"]
        pow_sum += _arr(g, "dc_current_power_w", num_dc)
        green_sum += _arr(g, "dc_current_green_power_w", num_dc)
        nstep += 1
        ga = greenest_dc(g, num_dc, batch)
        la = {}
        gn = _arr(g, "dc_current_green_power_w", num_dc)
        gr = _arr(g, "dc_green_ratio", num_dc)
        fut = _arr(g, "dc_future_short_mean", num_dc)
        for dc in range(num_dc):
            has_green = gn[dc] > 1.0 or gr[dc] > 0.05
            green_coming = fut[dc] > hold_thresh
            if policy == "drain":
                la[dc] = max_dispatch
            else:  # green_timed
                if has_green:
                    la[dc] = max_dispatch           # burst: capture green
                elif green_coming:
                    la[dc] = 0; n_hold += 1         # hold: wait for green
                else:
                    la[dc] = max_dispatch           # no green coming → drain on brown
        obs, _, term, trunc, info = env.step({"global": ga, "local": la})
        done = term or trunc
    m = collect_metrics(info, num_dc)
    m["_hold"] = n_hold
    m["_pow"] = (pow_sum / max(1, nstep)).tolist()
    m["_green"] = (green_sum / max(1, nstep)).tolist()
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_oracle_godeye")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-dispatch", type=int, default=50)
    ap.add_argument("--hold-thresh", type=float, default=0.3)
    ap.add_argument("--green-divisor", type=float, default=None)
    ap.add_argument("--host-mult", type=float, default=1.0,
                    help="scale every datacenter's host_count_* by this (>1 creates "
                         "compute slack so completion can reach ~1.0 → exposes the lever)")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    if args.host_mult != 1.0:
        for dc in cfg.get("datacenters", []):
            for k in list(dc.keys()):
                if k.startswith("host_count"):
                    dc[k] = max(1, int(round(dc[k] * args.host_mult)))
        print(f"[capacity] host_count_* ×{args.host_mult} → "
              f"hosts/DC = {[ {k:v for k,v in dc.items() if k.startswith('host_count')} for dc in cfg.get('datacenters',[]) ]}")
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    os.makedirs("/tmp/oracle_gateway", exist_ok=True)
    cfg["local_dispatch_mode"] = "dispatch_rate"
    cfg["max_dispatch_per_step"] = args.max_dispatch
    if args.green_divisor is not None:
        cfg["compressed_power_divisor"] = args.green_divisor
        print(f"[calibration] compressed_power_divisor → {args.green_divisor}")

    print(f"=== NL-3 oracle: dispatch_rate (max={args.max_dispatch}, seed={args.seed}) ===")
    env = HierarchicalMultiDCEnv(config=cfg)
    print("run 1/2: drain (no green timing) — does throughput unthrottle? ...")
    drain = run(env, "drain", args.max_dispatch, args.hold_thresh, args.seed)
    print("run 2/2: green_timed (hold brown / burst green) — does waste move? ...")
    timed = run(env, "green_timed", args.max_dispatch, args.hold_thresh, args.seed)
    env.close()

    def row(t, m):
        return (f"{t:12} completion={m.get('completion_rate_mi',0):.4f}  waste_ratio={m['waste_ratio']:.4f}  "
                f"carbon_kg={m['total_carbon_kg']:.4f}  carbon_int={m['carbon_intensity']:.4f}  "
                f"green_used={m['green_used_wh']:.0f}")
    print("\n=== RESULT ===")
    print(row("drain", drain))
    print(row("green_timed", timed), f"  (hold steps={timed['_hold']})")
    pw, gr = np.asarray(drain["_pow"]), np.asarray(drain["_green"])
    print(f"\nsupply/demand: demand={pw.sum():.0f}W green={gr.sum():.0f}W → green {gr.sum()/max(pw.sum(),1e-9):.1f}× demand")
    print("\n=== GATES ===")
    print(f"① throughput unthrottled? drain completion={drain.get('completion_rate_mi',0):.3f} "
          f"(legacy ~0.18) → {'✅ JUMPED' if drain.get('completion_rate_mi',0) > 0.35 else '❌ still low'}")
    dw = timed["waste_ratio"] - drain["waste_ratio"]
    print(f"② lever moves waste? Δwaste(timed−drain)={dw:+.4f} → "
          f"{'✅ green-timing REDUCES waste → proceed to NL-4 RL' if dw < -0.005 else '❌ no effect (check green balance / load)'}")


if __name__ == "__main__":
    main()
