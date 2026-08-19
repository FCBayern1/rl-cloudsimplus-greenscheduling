#!/usr/bin/env python3
"""Version-B deferral GATE — does temporal deferral capture green on the diurnal config?

Built for the synchronized-diurnal green world (experiment ..._diurnalB), where the OLD
oracle_hold_until_green collapses: its greedy "all-128-to-the-single-greenest-DC" routing
plus a non-draining bestfit local policy left the sim ~idle (134W, 28% completion), so its
verdict was meaningless.

This oracle isolates the TEMPORAL lever cleanly:
  - ROUTING is identical for both policies: round-robin across the green DCs (derived from the
    config's per-DC turbine_ids, so it follows the topology rather than assuming 0,1,2) to keep
    load balanced and completion high (the QoS half). WHERE doesn't matter for carbon here
    because the green DCs share one synchronized diurnal profile — only WHEN matters.
  - LOCAL dispatch is the only difference:
      * run-on-arrival : dispatch at max rate every step (drain).
      * defer-to-green : while a DC is DARK now but the forecast says green is imminent,
                         HOLD its deferrable backlog (dispatch 0); once green is present,
                         drain at max rate.
  green_oracle_mode=godeye => perfect forecast, so a null result is a physics verdict, not a
  forecast-quality one.

GATE:  defer-to-green should LOWER carbon / RAISE green_used at COMPARABLE completion.
       If completion craters, the lever is QoS-fatal (deadlines too tight for the green period).

Run from drl-manager/:
  .venv/bin/python oracle_diurnal_defer.py --experiment experiment_multi_5dc_carbon_v2_deferrable_gdpd_diurnalB
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import collect_metrics, _convert_local_obs_for_scheduler
from src.baselines.green_dcs import describe_green_dcs, green_capable_dcs, green_dcs_from_env
from oracle_hold_until_green import _arr, dc_has_green_now, dc_green_coming


def roundrobin_routing(batch: int, green_dcs: List[int], start: int) -> List[int]:
    """Spread the batch evenly across the green DCs (identical for both policies)."""
    return [green_dcs[(start + i) % len(green_dcs)] for i in range(batch)]


def leastloaded_green_routing(g, batch: int, green_dcs: List[int], num_dc: int) -> List[int]:
    """OBS-CONDITIONAL (no counter) spread: route this step's cloudlets to the green DC with the
    most available PEs right now. Across steps the obs snapshot updates → it self-balances. This is
    the LEARNABLE equivalent of the round-robin oracle (the net can compute this from dc_available_pes,
    but cannot reproduce round-robin's hidden counter). If this matches round-robin's result, a policy
    the network CAN represent exists → Tier 1 (an optimization/credit fix) is the right direction."""
    avail = _arr(g, "dc_available_pes", num_dc)
    target = green_dcs[int(np.argmax([avail[d] for d in green_dcs]))]
    return [target] * batch


def green_total_now(g, green_dcs, num_dc) -> float:
    gp = _arr(g, "dc_current_green_power_w", num_dc)
    return float(np.sum(gp[green_dcs]))


def run(env, defer: bool, lever: str, seed: int, drain: int, route_thresh_w: float = 1.0,
        routing: str = "roundrobin", green_dcs: List[int] = None) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    num_dc = env.num_datacenters
    # Which DCs are worth routing to is a property of the topology, not a constant:
    # derive it from the config unless the caller already did.
    if green_dcs is None:
        green_dcs = green_dcs_from_env(env)
    batch = env.global_routing_batch_size
    defer_idx = num_dc  # global DEFER action index (route to none)
    done = False
    n_hold = 0
    # night/day split instrumentation: is the deferred work actually shifting power into day?
    pow_day = np.zeros(num_dc); pow_night = np.zeros(num_dc)
    green_sum = np.zeros(num_dc); pow_sum = np.zeros(num_dc)
    n_day = n_night = n_steps = 0
    rr = 0
    while not done:
        g = obs["global"]
        draw = _arr(g, "dc_current_power_w", num_dc)
        pow_sum += draw
        green_sum += _arr(g, "dc_current_green_power_w", num_dc)
        # "light" = an ABOVE-THRESHOLD (windy) moment, not merely green>0. On 1s-compressed
        # real wind the signal flickers, so a threshold (~demand level) concentrates execution
        # into the genuinely windier windows instead of chasing single non-zero seconds.
        light = green_total_now(g, green_dcs, num_dc) >= route_thresh_w
        if light: pow_day += draw; n_day += 1
        else:     pow_night += draw; n_night += 1
        n_steps += 1

        # --- choose actions per lever ---
        if defer and lever == "global" and not light:
            global_action = [defer_idx] * batch      # system-wide dark -> DEFER all (don't route)
            n_hold += batch
            local_actions = {dc: drain for dc in range(num_dc)}
        else:
            if routing == "leastloaded":
                global_action = leastloaded_green_routing(g, batch, green_dcs, num_dc)
            else:
                global_action = roundrobin_routing(batch, green_dcs, rr)
                rr = (rr + batch) % len(green_dcs)
            local_actions = {}
            for dc in range(num_dc):
                if defer and lever == "local" and not dc_has_green_now(g, dc, num_dc):
                    local_actions[dc] = 0; n_hold += 1     # local hold while this DC dark
                else:
                    local_actions[dc] = drain
        obs, _, term, trunc, info = env.step({"global": global_action, "local": local_actions})
        done = term or trunc
    m = collect_metrics(info, num_dc)
    m["_hold_decisions"] = n_hold
    m["_mean_power_per_dc"] = (pow_sum / max(1, n_steps)).tolist()
    m["_mean_green_per_dc"] = (green_sum / max(1, n_steps)).tolist()
    m["_day_draw_W"] = float(pow_day.sum() / max(1, n_day))
    m["_night_draw_W"] = float(pow_night.sum() / max(1, n_night))
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_deferrable_gdpd_diurnalB")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lever", choices=["global", "local"], default="global",
                    help="global = DEFER action (don't route while dark); local = dispatch-0 hold")
    ap.add_argument("--route-thresh-w", type=float, default=1.0,
                    help="route/run only when total green across green DCs >= this (W); "
                         "defer otherwise. ~demand level concentrates work into windy windows.")
    ap.add_argument("--trace", default=None, help="override cloudlet_trace_file (e.g. a heavier workload)")
    ap.add_argument("--routing", choices=["roundrobin", "leastloaded"], default="roundrobin",
                    help="leastloaded = obs-conditional spread (LEARNABLE, no counter) vs round-robin")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    cfg.pop("py4j_port", None)
    if args.trace:
        cfg["cloudlet_trace_file"] = args.trace
        print(f"[workload] cloudlet_trace_file -> {args.trace}")
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_diurnal")
    cfg.setdefault("output_dir", "/tmp/oracle_diurnal")
    os.makedirs("/tmp/oracle_diurnal", exist_ok=True)
    drain = int(cfg.get("max_dispatch_per_step", 64))
    dc_cfgs = cfg.get("datacenters", [])
    green_dcs = green_capable_dcs(dc_cfgs)

    print(f"=== Version-B deferral gate: run-on-arrival vs defer-to-green "
          f"({args.experiment}, seed={args.seed}, drain={drain}, lever={args.lever}) ===")
    print(f"green-capable DCs ({len(green_dcs)}/{len(dc_cfgs)}): "
          f"{describe_green_dcs(green_dcs, dc_cfgs)}")
    env = HierarchicalMultiDCEnv(config=cfg)
    print(f"run 1/2: run-on-arrival (drain every step) ...")
    base = run(env, defer=False, lever=args.lever, seed=args.seed, drain=drain, route_thresh_w=args.route_thresh_w, routing=args.routing, green_dcs=green_dcs)
    print(f"run 2/2: defer-to-green ({args.lever} lever, routing={args.routing}, route when green>={args.route_thresh_w}W) ...")
    deferred = run(env, defer=True, lever=args.lever, seed=args.seed, drain=drain, route_thresh_w=args.route_thresh_w, routing=args.routing, green_dcs=green_dcs)
    env.close()

    def row(tag, m):
        return (f"{tag:16} carbon_kg={m['total_carbon_kg']:.4f}  completion={m.get('completion_rate_mi',0):.4f}  "
                f"green_used={m['green_used_wh']:.0f}  green_waste={m['green_waste_wh']:.0f}  "
                f"green_ratio={m.get('green_ratio',0):.3f}  waste_ratio={m['waste_ratio']:.3f}")
    print("\n=== RESULT ===")
    print(row("run-on-arrival", base))
    print(row("defer-to-green", deferred), f"  (defer decisions={deferred['_hold_decisions']})")
    print(f"\nnight/day power draw (W):  run-on-arrival day={base['_day_draw_W']:.0f} night={base['_night_draw_W']:.0f}"
          f"  |  defer day={deferred['_day_draw_W']:.0f} night={deferred['_night_draw_W']:.0f}")
    print("(defer should push draw INTO day and OUT of night; if night draw stays high, execution isn't being gated)")
    dc = deferred["total_carbon_kg"] - base["total_carbon_kg"]
    dgu = deferred["green_used_wh"] - base["green_used_wh"]
    dcompl = deferred.get("completion_rate_mi", 0) - base.get("completion_rate_mi", 0)
    print(f"\nΔcarbon_kg={dc:+.4f}  Δgreen_used_wh={dgu:+.0f}  Δcompletion={dcompl:+.4f}")

    pw = np.asarray(base["_mean_power_per_dc"]); gr = np.asarray(base["_mean_green_per_dc"])
    print("\n=== SUPPLY/DEMAND per DC (mean W, run-on-arrival = full-throughput reference) ===")
    for i in range(len(pw)):
        ratio = gr[i]/pw[i] if pw[i] > 1e-6 else float('inf')
        print(f"  dc{i}: demand={pw[i]:7.1f}W  green={gr[i]:7.1f}W  green/demand={ratio:6.2f}x")
    print(f"TOTAL demand={pw.sum():.0f}W green={gr.sum():.0f}W -> green {gr.sum()/max(pw.sum(),1e-6):.2f}x demand")

    ok = (dc < -0.005) and (dgu > 1) and (dcompl > -0.10)
    print("\nVERDICT:", "✅ defer LOWERS carbon + RAISES green_used at held completion -> physics works -> build B-RL"
          if ok else
          "❌ defer did not cleanly capture green at held completion -> tune green amplitude/period/duty and re-gate")


if __name__ == "__main__":
    main()
