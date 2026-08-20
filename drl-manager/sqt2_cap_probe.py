#!/usr/bin/env python3
"""SQT2.3 capacity-window probe (Codex 9-cell ruling, 2026-08-19).

Real-simulation verification of the offline FIFO window: capacity in
{208, 224, 240} fleet PEs (VM granularity: per-DC small/medium/large =
5/2/1, 6/2/1, 7/2/1 -> 26/28/30 PEs x 8 DCs) x policy in {nowait, naive,
clairvoyant} over the frozen anchors. Pre-registered selection rule:

    - nowait AND clairvoyant meet the TRIPLE contract at all 10 anchors;
    - naive shows at least one ontime-contract failure;
    - clairvoyant's carbon edge over the surviving blind arms still clears
      the formal thresholds.

If no capacity in the grid separates cleanly, STOP - no further capacity
tuning; the long-trough variant (SQT2.3B) becomes a Codex decision. Every
cell records c@7200, terminal, ontime, backlog, forced, spill and carbon.
"""
import argparse
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sqt2_prescreen import (ANCHORS, TroughIndex, resolve_blind_ck,  # noqa: E402
                            load_frozen_gate, run_episode)

ARMS = ("nowait", "naive", "clairvoyant")
PE_PER_VM = {"s": 2, "m": 4, "l": 8}   # small_vm_pes=2, medium x2, large x4


def scale_fleet(cfg: dict, factor: float):
    """Scale every DC's VM fleet by `factor`, preserving heterogeneity.

    Grid v1 (2026-08-19 00:43) homogenised all DCs to 26-30 PE and died:
    the local agent releases with no admission control, so a tiny fleet
    puts every job into processor sharing and EVERY arm (including the
    nowait control) misses deadlines - capacity pressure that cannot
    discriminate smooth from bursty release. This scaler keeps the
    testbed's DC asymmetry (600/480/296/240/184 PE) and only moves the
    overall level, which is the axis the separation argument needs.
    Returns the resulting fleet PE total."""
    total = 0
    for dc in cfg["datacenters"]:
        for key, tag in (("initial_s_vm_count", "s"),
                         ("initial_m_vm_count", "m"),
                         ("initial_l_vm_count", "l")):
            n = max(1, int(round(dc[key] * factor)))
            dc[key] = n
            total += n * PE_PER_VM[tag]
    return total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, required=True,
                    help="fleet VM scale factor (1.0 = baseline 2520 PE)")
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--anchors", default=",".join(str(a) for a in ANCHORS))
    ap.add_argument("--tag", default=None)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from h1_matched_headroom import ModuleHead
    from src.baselines.evaluate import load_config
    repo = pathlib.Path(__file__).resolve().parent
    art = json.loads((repo / "calib/sqt2_schedule.json").read_text())
    tindex = TroughIndex(art["troughs"])
    hazard_q, _ = load_frozen_gate(repo)
    cfg = load_config("experiment_sqt2_noforecast")
    cfg.pop("py4j_port", None)
    tag = args.tag or f"f{args.scale:g}"
    cfg.setdefault("gateway_log_dir", f"/tmp/sqt2_gateway_{tag}")
    cfg.setdefault("output_dir", f"/tmp/sqt2_gateway_{tag}")
    pathlib.Path(cfg["gateway_log_dir"]).mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = 10000
    fleet_pes = scale_fleet(cfg, args.scale)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    anchors = [int(a) for a in args.anchors.split(",") if a.strip()]
    print(f"[{tag}] fleet={fleet_pes} PE (baseline 2520) arms={arms} "
          f"anchors={anchors}", flush=True)
    brown = [float(d.get("brown_carbon_factor", 0.5)) for d in cfg["datacenters"]]

    head = ModuleHead(resolve_blind_ck(repo))
    records = []
    for arm in arms:
        env = HierarchicalMultiDCEnv(dict(cfg))
        try:
            next_k = 0
            for k in anchors:
                while next_k < k:
                    env.reset(seed=1)
                    next_k += 1
                rec = run_episode(env, cfg, "ppo", arm, tindex, k, head,
                                  hazard_q, brown)
                next_k = k + 1
                rec["scale"] = args.scale
                rec["fleet_pes"] = fleet_pes
                records.append(rec)
                print(f"[{tag} {arm:11s} ep{k:>3}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"c@7200={rec['completion_at_7200']:.4f} "
                      f"term={rec['completion_rate_mi']:.4f} "
                      f"ontime={rec['ontime_mi_share']:.4f} "
                      f"defer={rec['defer_slots']} spill={rec['spill_slots']} "
                      f"forced={rec['deadline_forced_count']} "
                      f"blmax={rec['backlog_max']}", flush=True)
        finally:
            try:
                env.close()
            except Exception:
                pass
    out = args.json_out or f"../local_eval_rt/audit/sqt2_cap_probe_{tag}.json"
    pathlib.Path(out).write_text(json.dumps(
        {"scale": args.scale, "fleet_pes": fleet_pes,
         "records": records}, indent=1))
    print(f"[{tag} DONE]", flush=True)


if __name__ == "__main__":
    main()
