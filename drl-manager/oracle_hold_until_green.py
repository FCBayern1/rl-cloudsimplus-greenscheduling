#!/usr/bin/env python3
"""
B-oracle (Deferrable-Jobs lever, Option B validation) — 2026-06-18.

Cheap, decisive test of whether the EXISTING local NoAssign hold can reduce
green waste IF it could see the forecast. Runs two scripted policies on the
SAME env/seed and compares waste_ratio / carbon:

  - no-hold baseline : greenest-DC routing + BestFit local (always assign).
  - hold-until-green : same routing; local HOLDS (NoAssign) a DC's waiting
                       cloudlets while the DC has no green now BUT the forecast
                       says green is coming; assigns (BestFit) once green is
                       present. Reads the per-DC green/forecast straight from
                       obs['global'] (the RL local agent can't — this is the
                       upper-bound test of the local-hold lever).

Uses green_oracle_mode=godeye (PERFECT forecast) so a null result means the
lever mechanism itself doesn't help (not a forecast-quality issue).

Decision gate:
  waste_ratio(hold) < waste_ratio(no-hold)  -> local-hold lever works -> build B-RL.
  ~equal                                    -> same-DC limit fatal -> go to Option A.

Run from drl-manager/:
  .venv/bin/python oracle_hold_until_green.py --experiment experiment_multi_5dc_carbon_v2_oracle_godeye
"""
import argparse
import os
import sys
from pathlib import Path
from typing import Dict, Any, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
from src.baselines.evaluate import load_config, collect_metrics, _convert_local_obs_for_scheduler


def _arr(obs, key, n):
    v = np.asarray(obs.get(key, []), dtype=np.float64).ravel()
    if v.size < n:
        v = np.concatenate([v, np.zeros(n - v.size)])
    return v[:n]


def greenest_dc_routing(global_obs: Dict[str, Any], num_dc: int, batch: int) -> List[int]:
    """Route every cloudlet in the batch to the DC greenest now-or-soon."""
    green_now = _arr(global_obs, "dc_current_green_power_w", num_dc)
    fut = _arr(global_obs, "dc_future_short_mean", num_dc)
    score = green_now / (green_now.max() + 1e-9) + fut  # now + imminent
    target = int(np.argmax(score))
    return [target] * batch


def dc_has_green_now(global_obs, dc, num_dc) -> bool:
    gp = _arr(global_obs, "dc_current_green_power_w", num_dc)[dc]
    gr = _arr(global_obs, "dc_green_ratio", num_dc)[dc]
    return gp > 1.0 or gr > 0.05


def dc_green_coming(global_obs, dc, num_dc, thresh) -> bool:
    """Forecast says this DC will be (more) green in the short horizon."""
    return _arr(global_obs, "dc_future_short_mean", num_dc)[dc] > thresh


def bestfit_assign(local_obs, mask) -> int:
    """BestFit VM choice; returns 0 (NoAssign) only if nothing assignable."""
    pes = np.asarray(local_obs.get("vm_available_pes", []), dtype=np.float64).ravel()
    need = local_obs.get("next_cloudlet_pes", 1)
    need = float(np.asarray(need).ravel()[0]) if np.size(need) else 1.0
    best, best_slack = 0, None
    for vm in range(pes.size):
        if vm + 1 < len(mask) and mask[vm + 1] and pes[vm] >= need:
            slack = pes[vm] - need
            if best_slack is None or slack < best_slack:
                best_slack, best = slack, vm + 1
    return best


def run(env, hold: bool, hold_thresh: float, seed: int) -> Dict[str, Any]:
    obs, info = env.reset(seed=seed)
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    done = False
    n_hold_steps = 0
    while not done:
        g = obs["global"]
        global_action = greenest_dc_routing(g, num_dc, batch)
        local_actions = {}
        for dc in range(num_dc):
            mask = env.get_local_action_masks(dc)
            lobs = _convert_local_obs_for_scheduler(obs["local"].get(dc, {}))
            waiting = int(np.asarray(lobs.get("waiting_cloudlets", 0)).ravel()[0]) \
                if "waiting_cloudlets" in lobs else 1
            if hold and waiting > 0 and not dc_has_green_now(g, dc, num_dc) \
                    and dc_green_coming(g, dc, num_dc, hold_thresh):
                local_actions[dc] = 0  # NoAssign = HOLD (wait for green)
                n_hold_steps += 1
            else:
                local_actions[dc] = bestfit_assign(lobs, mask)
        obs, _, term, trunc, info = env.step({"global": global_action, "local": local_actions})
        done = term or trunc
    m = collect_metrics(info, num_dc)
    m["_hold_decisions"] = n_hold_steps
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_multi_5dc_carbon_v2_oracle_godeye")
    ap.add_argument("--config", default=str(Path(__file__).resolve().parent.parent / "config.yml"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--hold-thresh", type=float, default=0.3,
                    help="forecast dc_future_short_mean above which 'green is coming'")
    args = ap.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))[args.experiment]
    # Auto-launch a fresh gateway on a free port (training does the same); the
    # config's fixed py4j_port points at a gateway that isn't running.
    cfg.pop("py4j_port", None)
    # Config keys the training entrypoint normally injects (not in the raw
    # experiment block) — provide sane defaults so the standalone env launches.
    cfg.setdefault("gateway_log_dir", "/tmp/oracle_gateway")
    cfg.setdefault("output_dir", "/tmp/oracle_gateway")
    os.makedirs("/tmp/oracle_gateway", exist_ok=True)

    print(f"=== B-oracle: hold-until-green vs no-hold ({args.experiment}, seed={args.seed}) ===")
    print("launching env + Java gateway...")
    env = HierarchicalMultiDCEnv(config=cfg)
    # Same env, same seed → identical episode; the only difference is the policy.
    print("run 1/2: no-hold baseline ...")
    base = run(env, hold=False, hold_thresh=args.hold_thresh, seed=args.seed)
    print("run 2/2: hold-until-green oracle ...")
    held = run(env, hold=True, hold_thresh=args.hold_thresh, seed=args.seed)
    env.close()

    def row(tag, m):
        return (f"{tag:16} waste_ratio={m['waste_ratio']:.4f}  carbon_kg={m['total_carbon_kg']:.4f}  "
                f"carbon_int={m['carbon_intensity']:.4f}  completion={m.get('completion_rate_mi', 0):.4f}  "
                f"green_used={m['green_used_wh']:.0f}  green_waste={m['green_waste_wh']:.0f}")
    print("\n=== RESULT ===")
    print(row("no-hold", base))
    print(row("hold-until-green", held), f"  (hold decisions={held['_hold_decisions']})")
    dw = held["waste_ratio"] - base["waste_ratio"]
    dc = held["total_carbon_kg"] - base["total_carbon_kg"]
    print(f"\nΔwaste_ratio = {dw:+.4f}   Δcarbon_kg = {dc:+.4f}")
    print("VERDICT:", "✅ hold REDUCES waste → local-hold lever works → build B-RL"
          if dw < -0.005 else
          "❌ waste ~unchanged → same-DC hold limit fatal → go to Option A (global defer)")


if __name__ == "__main__":
    main()
