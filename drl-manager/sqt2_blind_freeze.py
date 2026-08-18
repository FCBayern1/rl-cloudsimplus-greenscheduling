#!/usr/bin/env python3
"""SQT2.2-Clean blind-comparator freeze by carbon/SLA (Codex P0-2, 2026-08-18).

Classification accuracy is not the objective - carbon under the completion
contract is. This runs the PRE-REGISTERED blind candidate set on the
CALIBRATION schedule over the frozen PPO route-only base:

    nowait, naive, hazard@q for q in {0.25, 0.40, 0.50, 0.60}

then freezes the comparator as: among candidates whose every anchor meets
the DUAL completion contract (completion@7200 >= 99.5% AND terminal
>= 99.5%), the one with the lowest median terminal carbon; ties broken by
the offline label accuracy already recorded in calib/sqt2_hazard_freeze.json.
The winner (and its q, if a hazard arm) is written back into that artifact
as comparator_v2 / q_star_carbon. Held-out never re-tunes any of this.

If NO candidate survives the dual contract, comparator_v2 is written as
null with the measured ceilings - that outcome escalates to Codex, it is
never resolved by locally softening a threshold.
"""
import argparse
import json
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from sqt2_prescreen import (ANCHORS, BLIND_CK, CONTRACT, TroughIndex,  # noqa: E402
                            run_episode)

CANDIDATE_QS = (0.25, 0.40, 0.50, 0.60)
ARMS = ("nowait", "naive") + tuple(f"hazard@{q:.2f}" for q in CANDIDATE_QS)


def arm_spec(arm: str):
    """arm label -> (mode, hazard_q)."""
    if arm.startswith("hazard@"):
        return "hazard", float(arm.split("@")[1])
    return arm, 0.5


def freeze_by_carbon(records, accuracies=None, contract: float = CONTRACT):
    """Pure freeze rule: dual-SLA filter, then min median terminal carbon.

    `records` carry an "arm" label. nowait is the control, not a candidate.
    Returns per-candidate stats plus winner (or None if the filter empties).
    """
    stats = {}
    for arm in sorted({r["arm"] for r in records}):
        rs = [r for r in records if r["arm"] == arm]
        ok = all(r["completion_at_7200"] >= contract
                 and r["completion_rate_mi"] >= contract for r in rs)
        stats[arm] = {
            "anchors": len(rs),
            "dual_sla_all_anchors": ok,
            "min_completion_at_7200": min(r["completion_at_7200"] for r in rs),
            "min_terminal_completion": min(r["completion_rate_mi"] for r in rs),
            "median_terminal_carbon": float(np.median(
                [r["total_carbon_kg"] for r in rs])),
            "median_carbon_at_7200": float(np.median(
                [r["carbon_at_7200"] for r in rs]))}
    eligible = [a for a, s in stats.items()
                if a != "nowait" and s["dual_sla_all_anchors"]]
    if not eligible:
        return stats, None
    best = min(eligible, key=lambda a: stats[a]["median_terminal_carbon"])
    ties = [a for a in eligible
            if stats[a]["median_terminal_carbon"]
            == stats[best]["median_terminal_carbon"]]
    if len(ties) > 1 and accuracies:
        def acc(a):
            key = "naive" if a == "naive" else f"hazard@{arm_spec(a)[1]:.2f}"
            return accuracies.get(key, {}).get("acc", 0.0)
        best = max(ties, key=acc)
    return stats, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-out",
                    default="../local_eval_rt/audit/sqt2_blind_freeze_records.json")
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    from h1_matched_headroom import ModuleHead
    from src.baselines.evaluate import load_config
    repo = pathlib.Path(__file__).resolve().parent
    art = json.loads((repo / "calib/sqt2_schedule.json").read_text())
    assert art.get("variant", "cal") == "cal", "comparator freezes on calibration only"
    tindex = TroughIndex(art["troughs"])
    cfg = load_config("experiment_sqt2_noforecast")
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/sqt2_gateway")
    cfg.setdefault("output_dir", "/tmp/sqt2_gateway")
    pathlib.Path("/tmp/sqt2_gateway").mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = 10000
    brown = [float(d.get("brown_carbon_factor", 0.5)) for d in cfg["datacenters"]]

    head = ModuleHead(repo / BLIND_CK)
    records = []
    for arm in ARMS:
        mode, q = arm_spec(arm)
        env = HierarchicalMultiDCEnv(dict(cfg))
        try:
            next_k = 0
            for k in ANCHORS:
                while next_k < k:
                    env.reset(seed=1)
                    next_k += 1
                rec = run_episode(env, cfg, "ppo", mode, tindex, k, head,
                                  q, brown)
                next_k = k + 1
                rec["arm"] = arm
                rec["hazard_q"] = q if mode == "hazard" else None
                records.append(rec)
                print(f"[BFRZ {arm:12s} ep{k:>3}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"c@7200={rec['completion_at_7200']:.4f} "
                      f"term={rec['completion_rate_mi']:.4f} "
                      f"defer={rec['defer_slots']} "
                      f"forced={rec['deadline_forced_count']}", flush=True)
        finally:
            try:
                env.close()
            except Exception:
                pass

    freeze_art_path = repo / "calib/sqt2_hazard_freeze.json"
    freeze_art = json.loads(freeze_art_path.read_text())
    stats, winner = freeze_by_carbon(records, freeze_art.get("accuracies"))
    freeze_art["comparator_v2"] = winner
    freeze_art["comparator_v2_method"] = ("lowest median terminal carbon among "
                                          "candidates meeting the dual SLA at "
                                          "every anchor; tie-break by label acc")
    freeze_art["comparator_v2_stats"] = stats
    if winner and winner.startswith("hazard@"):
        freeze_art["q_star_carbon"] = arm_spec(winner)[1]
    freeze_art_path.write_text(json.dumps(freeze_art, indent=1))
    pathlib.Path(args.records_out).write_text(
        json.dumps({"records": records, "stats": stats,
                    "winner": winner}, indent=1))
    for a, s in stats.items():
        print(f"[BFRZ STAT {a:12s}] dualSLA={s['dual_sla_all_anchors']} "
              f"minC7200={s['min_completion_at_7200']:.4f} "
              f"medCarbon={s['median_terminal_carbon']:.4f}", flush=True)
    if winner is None:
        print("[BFRZ ESCALATE] no blind candidate meets the dual SLA at all "
              "anchors - comparator_v2=null, ruling needed (no local "
              "threshold softening)", flush=True)
    else:
        print(f"[BFRZ FROZEN] comparator_v2={winner}", flush=True)


if __name__ == "__main__":
    main()
