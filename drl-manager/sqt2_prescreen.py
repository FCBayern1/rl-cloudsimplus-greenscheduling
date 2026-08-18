#!/usr/bin/env python3
"""SQT2.2 headroom prescreen (Codex-approved gate ladder, 2026-08-18).

Four temporal gates over the SAME greedy spatial baseline (pick_targets +
SlotAllocator burst spreading), at the ten pre-registered anchor episodes
K = {0,20,40,59,79,99,119,138,158,178}:

    nowait      never defer                      control ceiling (contract!)
    naive       defer iff green==0 and B>0       weak current-state gate
    hazard      defer iff P(trough ends within B | trough age) >= 0.5
                closed-form posterior over the REGISTERED duration mixture -
                sees green-now, slack, trough age, training distribution,
                NEVER the future (strongest blind baseline)
    clairvoyant defer iff true residual <= B     future information

B = time_to_deadline - runtime - 120 with runtime = MI/(PES*MIPS), matching
the latest-start backstop exactly. Verdicts: pair_verdict validity gating,
clairvoyant must beat the BEST of the three blind arms with median relative
improvement >= 8% (prescreen line, not lowered).
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS, _arr, drain_action, load_green_series  # noqa: E402
from teacher_reward_audit import SlotAllocator, episode_offset  # noqa: E402
from h1_matched_headroom import pair_verdict  # noqa: E402
from src.baselines.evaluate import load_config, collect_metrics  # noqa: E402

MIPS = 40000.0
MARGIN_S = 120.0
BACKLOG_CAP = 200
ANCHORS = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)
MIX = ((0.8, 300.0, 1500.0), (0.2, 2700.0, 4500.0))   # registered trough mixture


def hazard_p_end_within(age: float, budget: float, mix=MIX) -> float:
    """P(trough ends within `budget` more seconds | it has lasted `age`).

    Closed form for a mixture of uniforms. Survival of component (lo,hi) at
    age a: 1 if a<=lo; (hi-a)/(hi-lo) if lo<a<hi; 0 else. Conditional finish
    within budget: (min(hi,a+B) - max(a,lo)) / (hi - max(a,lo)).
    """
    if budget <= 0:
        return 0.0
    num = den = 0.0
    for w, lo, hi in mix:
        if age >= hi:
            continue
        surv = 1.0 if age <= lo else (hi - age) / (hi - lo)
        wsurv = w * surv
        left = max(age, lo)
        x = min(hi, age + budget)
        cond = 0.0 if x <= left else (x - left) / (hi - left)
        num += wsurv * cond
        den += wsurv
    return num / den if den > 0 else 1.0


class TroughIndex:
    """Row -> (in_trough, age, residual) lookup from the schedule artifact."""

    def __init__(self, troughs):
        self.iv = [(t["start"], t["start"] + t["dur"]) for t in troughs]

    def query(self, row: int):
        for s, e in self.iv:
            if s <= row < e:
                return True, float(row - s), float(e - row)
        return False, 0.0, 0.0


def gate_flags(mode: str, g, batch: int, ttd_scale: float,
               in_trough: bool, age: float, residual: float) -> np.ndarray:
    mi = _arr(g, "batch_cloudlet_mi", batch)
    pes = _arr(g, "batch_cloudlet_pes", batch)
    ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
    present = _arr(g, "batch_cloudlet_deadline_present", batch)
    backlog = int(_arr(g, "global_deferred_count", 1)[0] * 2000.0)
    flags = np.zeros(batch, dtype=bool)
    if mode == "nowait" or not in_trough or backlog >= BACKLOG_CAP:
        return flags
    for i in range(batch):
        if mi[i] <= 0 or present[i] <= 0.5:
            continue
        runtime = mi[i] / (max(1.0, pes[i]) * MIPS)
        budget = ttd[i] - runtime - MARGIN_S
        if budget <= 0:
            continue
        if mode == "naive":
            flags[i] = True
        elif mode == "hazard":
            flags[i] = hazard_p_end_within(age, budget) >= 0.5
        elif mode == "clairvoyant":
            flags[i] = residual <= budget
    return flags


def run_episode(env, cfg, mode, tindex, episode_index):
    obs, info = env.reset(seed=1)
    off_range = int(cfg.get("green_episode_offset_range", 0) or 0)
    expected = episode_offset(episode_index, off_range)
    actual = int(getattr(env, "_green_episode_offset_rows", 0))
    if actual != expected:
        raise RuntimeError(f"offset misalign ep{episode_index}: {actual}!={expected}")
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec", 3600.0)))
    done, t, defers = False, 0, 0
    compl_7200 = carbon_7200 = None
    while not done:
        g = obs["global"]
        row = WARMUP_ROWS + actual + t
        in_trough, age, residual = tindex.query(row)
        mi = _arr(g, "batch_cloudlet_mi", batch)
        hold = gate_flags(mode, g, batch, ttd_scale, in_trough, age, residual)
        alloc = SlotAllocator(_arr(g, "dc_current_green_power_w", num_dc),
                              _arr(g, "dc_available_pes", num_dc),
                              _arr(g, "dc_queue_sizes", num_dc))
        actions = [int(num_dc) if (hold[i] and mi[i] > 0) else alloc.take_green()
                   for i in range(batch)]
        defers += int(sum(1 for i in range(batch) if hold[i] and mi[i] > 0))
        local = {dc: drain_action(env.get_local_action_masks(dc))
                 for dc in range(num_dc)}
        obs, _, term, trunc, info = env.step({"global": actions, "local": local})
        done = term or trunc
        t += 1
        if t == 7200:
            ges = info.get("global_energy_stats") or {}
            compl_7200 = float(ges.get("completion_rate_mi", 0.0) or 0.0)
            carbon_7200 = float(ges.get("total_carbon_emission_kg", 0.0) or 0.0)
    m = collect_metrics(info, num_dc)
    if compl_7200 is None:
        compl_7200 = float(m.get("completion_rate_mi", 0.0) or 0.0)
        carbon_7200 = float(m.get("total_carbon_kg", 0.0) or 0.0)
    return {"mode": mode, "episode_index": episode_index, "green_offset": actual,
            "steps": t, "defer_slots": defers,
            "total_carbon_kg": float(m.get("total_carbon_kg", 0.0) or 0.0),
            "completion_rate_mi": float(m.get("completion_rate_mi", 0.0) or 0.0),
            "completion_at_7200": compl_7200, "carbon_at_7200": carbon_7200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_sqt2_noforecast")
    ap.add_argument("--arms", default="nowait,naive,hazard,clairvoyant")
    ap.add_argument("--drain-horizon", type=int, default=10000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    repo = pathlib.Path(__file__).resolve().parent
    art = json.loads((repo / "calib/sqt2_schedule.json").read_text())
    tindex = TroughIndex(art["troughs"])
    cfg = load_config(args.experiment)
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/sqt2_gateway")
    cfg.setdefault("output_dir", "/tmp/sqt2_gateway")
    pathlib.Path("/tmp/sqt2_gateway").mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = int(args.drain_horizon)

    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    records = []
    for arm in arms:
        env = HierarchicalMultiDCEnv(dict(cfg))
        try:
            next_k = 0
            for k in ANCHORS:
                while next_k < k:            # fast-forward the offset counter
                    env.reset(seed=1)
                    next_k += 1
                rec = run_episode(env, cfg, arm, tindex, k)
                next_k = k + 1
                records.append(rec)
                print(f"[SQT2PS {arm:11s} ep{k:>3} off={rec['green_offset']:>6}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"compl={rec['completion_rate_mi']:.4f} "
                      f"defer={rec['defer_slots']}", flush=True)
                if arm == "nowait" and rec["completion_rate_mi"] < 0.995:
                    print(f"[SQT2PS ABORT] control below contract at ep{k} "
                          f"({rec['completion_rate_mi']:.4f}) - recipe FAILS "
                          f"honestly, no fallback to t50", flush=True)
                    if args.json_out:
                        pathlib.Path(args.json_out).write_text(
                            json.dumps({"records": records,
                                        "verdict": "CONTROL_CEILING_FAIL"}, indent=1))
                    return
        finally:
            try:
                env.close()
            except Exception:
                pass

    base = {r["episode_index"]: r for r in records if r["mode"] == "nowait"}
    verdicts = {}
    for arm in arms:
        if arm == "nowait":
            continue
        pv = [dict(pair_verdict(r, base[r["episode_index"]]),
                   episode_index=r["episode_index"])
              for r in records if r["mode"] == arm]
        valid = [v for v in pv if v["valid"]]
        neg = sum(1 for v in valid if v["rel_delta_raw"] < 0)
        verdicts[arm] = {
            "valid_pairs": len(valid), "invalid_pairs": len(pv) - len(valid),
            "improve_signs": f"{neg}/{len(valid)}",
            "median_rel_delta": (float(np.median([v["rel_delta_raw"] for v in valid]))
                                 if valid else None)}
        v = verdicts[arm]
        print(f"[SQT2PS VERDICT {arm}] valid {v['valid_pairs']} "
              f"median {v['median_rel_delta'] if v['median_rel_delta'] is not None else float('nan'):+.4f} "
              f"improves {v['improve_signs']}", flush=True)
    blind_best = min((verdicts[a]["median_rel_delta"]
                      for a in ("naive", "hazard") if a in verdicts
                      and verdicts[a]["median_rel_delta"] is not None),
                     default=0.0)
    if "clairvoyant" in verdicts and verdicts["clairvoyant"]["median_rel_delta"] is not None:
        c = verdicts["clairvoyant"]["median_rel_delta"]
        print(f"[SQT2PS FINAL] clairvoyant {c:+.4f} vs best-blind {blind_best:+.4f} "
              f"| passes 8% line: {c <= -0.08} | beats blind: {c < blind_best}",
              flush=True)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(
            json.dumps({"anchors": list(ANCHORS), "records": records,
                        "verdicts": verdicts}, indent=1))


if __name__ == "__main__":
    main()
