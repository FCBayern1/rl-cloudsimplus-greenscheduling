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

from sqt2_prescreen import (ANCHORS, BLIND_CK, TroughIndex,  # noqa: E402
                            load_frozen_gate, run_episode)

VM_GRID = {208: (5, 2, 1), 224: (6, 2, 1), 240: (7, 2, 1)}
ARMS = ("nowait", "naive", "clairvoyant")


def squeeze(cfg: dict, cap: int) -> dict:
    s, m, l = VM_GRID[cap]
    for dc in cfg["datacenters"]:
        dc["initial_s_vm_count"] = s
        dc["initial_m_vm_count"] = m
        dc["initial_l_vm_count"] = l
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True, choices=sorted(VM_GRID))
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
    cfg.setdefault("gateway_log_dir", f"/tmp/sqt2_gateway_cap{args.cap}")
    cfg.setdefault("output_dir", f"/tmp/sqt2_gateway_cap{args.cap}")
    pathlib.Path(cfg["gateway_log_dir"]).mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = 10000
    cfg = squeeze(cfg, args.cap)
    brown = [float(d.get("brown_carbon_factor", 0.5)) for d in cfg["datacenters"]]

    head = ModuleHead(repo / BLIND_CK)
    records = []
    for arm in ARMS:
        env = HierarchicalMultiDCEnv(dict(cfg))
        try:
            next_k = 0
            for k in ANCHORS:
                while next_k < k:
                    env.reset(seed=1)
                    next_k += 1
                rec = run_episode(env, cfg, "ppo", arm, tindex, k, head,
                                  hazard_q, brown)
                next_k = k + 1
                rec["cap"] = args.cap
                records.append(rec)
                print(f"[CAP{args.cap} {arm:11s} ep{k:>3}] "
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
    out = args.json_out or f"../local_eval_rt/audit/sqt2_cap_probe_{args.cap}.json"
    pathlib.Path(out).write_text(json.dumps(
        {"cap": args.cap, "vm_grid": VM_GRID[args.cap],
         "records": records}, indent=1))
    print(f"[CAP{args.cap} DONE]", flush=True)


if __name__ == "__main__":
    main()
