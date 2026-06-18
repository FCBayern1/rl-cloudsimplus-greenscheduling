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


def route(g, num_dc, batch, green_capable, feast_aware, fut_w):
    """Spatial: send the whole batch to the greenest DC.
    feast_aware=False -> greenest NOW (green_now only).
    feast_aware=True  -> greenest NOW-OR-SOON, restricted to green-capable DCs
                         (so admitted work piles onto a DC that has/will have a
                         feast surplus, rather than spilling onto brown DCs)."""
    gn = _arr(g, "dc_current_green_power_w", num_dc)
    fut = _arr(g, "dc_future_short_mean", num_dc)
    score = gn / (gn.max() + 1e-9)
    if feast_aware:
        score = score + fut_w * fut
        mask = np.array([1.0 if i in green_capable else 0.0 for i in range(num_dc)])
        if mask.sum() > 0:
            score = np.where(mask > 0, score, -1e9)  # green-capable only
    return [int(np.argmax(score))] * batch


def run(env, policy, max_dispatch, hold_thresh, fut_w, green_capable, seed):
    obs, info = env.reset(seed=seed)
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    pow_sum = np.zeros(num_dc); green_sum = np.zeros(num_dc); nstep = 0
    n_hold = 0
    feast_aware = (policy == "feast")
    done = False
    while not done:
        g = obs["global"]
        gn = _arr(g, "dc_current_green_power_w", num_dc)
        dem = _arr(g, "dc_current_power_w", num_dc)
        fut = _arr(g, "dc_future_short_mean", num_dc)
        pow_sum += dem
        green_sum += gn
        nstep += 1
        ga = route(g, num_dc, batch, green_capable, feast_aware, fut_w)
        la = {}
        for dc in range(num_dc):
            if policy == "spatial":
                la[dc] = max_dispatch                       # drain everywhere, no timing
                continue
            # --- feast policy ---
            if dc not in green_capable:
                la[dc] = max_dispatch                       # brown DC: holding is pointless
            elif gn[dc] > dem[dc]:
                la[dc] = max_dispatch                       # FEAST (green>demand): burst to soak surplus
            elif fut[dc] > hold_thresh:
                la[dc] = 0; n_hold += 1                     # famine but feast COMING soon: HOLD & wait
            else:
                la[dc] = max_dispatch                       # famine, no feast in sight: drain
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
    ap.add_argument("--hold-thresh", type=float, default=0.5,
                    help="normalized dc_future_short_mean above which 'feast is coming' (hold)")
    ap.add_argument("--fut-w", type=float, default=1.0,
                    help="weight on forecast vs green-now in feast-aware routing")
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

    green_capable = {i for i, dc in enumerate(cfg.get("datacenters", [])) if dc.get("turbine_ids")}
    print(f"=== NL-3 feast-packing oracle (max={args.max_dispatch}, seed={args.seed}) ===")
    print(f"green-capable DCs (have turbines): {sorted(green_capable)}")
    env = HierarchicalMultiDCEnv(config=cfg)
    print("run 1/2: spatial baseline (greenest-now routing + drain everywhere) ...")
    spatial = run(env, "spatial", args.max_dispatch, args.hold_thresh, args.fut_w, green_capable, args.seed)
    print("run 2/2: feast-packing (greenest-soon routing + hold-famine-burst-feast at green DCs) ...")
    feast = run(env, "feast", args.max_dispatch, args.hold_thresh, args.fut_w, green_capable, args.seed)
    env.close()

    def row(t, m):
        return (f"{t:14} completion={m.get('completion_rate_mi',0):.4f}  waste_ratio={m['waste_ratio']:.4f}  "
                f"carbon_kg={m['total_carbon_kg']:.4f}  carbon_int={m['carbon_intensity']:.4f}  "
                f"green_used={m['green_used_wh']:.0f}")
    print("\n=== RESULT ===")
    print(row("spatial", spatial))
    print(row("feast-packing", feast), f"  (hold steps={feast['_hold']})")
    pw, gr = np.asarray(spatial["_pow"]), np.asarray(spatial["_green"])
    print(f"\nsupply/demand: demand={pw.sum():.0f}W green={gr.sum():.0f}W → green {gr.sum()/max(pw.sum(),1e-9):.1f}× demand")
    dw = feast["waste_ratio"] - spatial["waste_ratio"]
    dc = feast["total_carbon_kg"] - spatial["total_carbon_kg"]
    dcomp = feast.get("completion_rate_mi", 0) - spatial.get("completion_rate_mi", 0)
    print("\n=== GATE: does the temporal feast-packing lever capture surplus? ===")
    print(f"Δwaste(feast−spatial) = {dw:+.4f}   Δcarbon_kg = {dc:+.4f}   Δcompletion = {dcomp:+.4f}")
    if dw < -0.005 and dcomp > -0.05:
        print("✅ feast-packing REDUCES waste without tanking completion → temporal lever REAL → build NL-4 RL")
    elif dw < -0.005:
        print("⚠️ waste drops BUT completion falls — lever exists but needs deadline-aware SLA to bound holds")
    else:
        print("❌ waste flat → feast surplus is structurally uncapturable (DC0/DC1 capacity-bound at feast) "
              "→ temporal lever dead, value is spatial routing only")


if __name__ == "__main__":
    main()
