#!/usr/bin/env python3
"""SQT2.2-Clean headroom prescreen (Codex four-layer ruling, 2026-08-18).

Layer 1 - two spatial bases, temporal gates share the base's spatial actions:
    ppo     frozen V3.2 blind PPO route-only argmax (defer logit excluded);
            the FORMAL base - conclusions must hold here
    greedy  power-aware sentinel: green headroom net of the job's
            counterfactual increment; trough fallback spreads by brown
            factor + per-step PE ledger + queue, never index order

Layer 2 - four temporal arms over the same base:
    nowait / naive / hazard@q* / clairvoyant, with q* and the blind
    comparator FROZEN by sqt2_hazard_calibrate.py on calibration data only.

Layer 3 - decision boundary locked to training:
    B_eff = min(deadline - now - runtime - 120, 7200 - now - runtime - 120)
    (teacher_reward_audit.effective_budget); NO deferral at or past step
    7200; completion@7200 and terminal completion reported separately.

Layer 4 - direct paired verdict on the ppo base:
    nowait completion@7200 >= 99.5% at all 10 anchors (else abort, no
    fallback); clairvoyant vs nowait median <= -8%; clairvoyant vs frozen
    comparator median <= -5%; >= 8 valid pairs and >= 8/10 negative signs on
    both comparisons; deadline_forced_count / forced share / backlog max
    reported so guardrail-surfing is visible.

Run on the calibration schedule first (protocol shakedown, --schedule cal),
then ONCE on the held-out schedule (--schedule ho) for the formal verdict.
"""
import argparse
import os
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS, _arr, drain_action  # noqa: E402
from teacher_reward_audit import episode_offset, effective_budget  # noqa: E402
from h1_matched_headroom import ModuleHead, pair_verdict  # noqa: E402
from src.baselines.evaluate import load_config, collect_metrics  # noqa: E402

MIPS = 40000.0
MARGIN_S = 120.0
HORIZON_S = 7200.0
BACKLOG_CAP = 200
CONTRACT = 0.995
# SQT2.3 unified release (Codex, 2026-08-19): every gate releases a held job
# through the SAME spatial policy (PPO + shield) strictly BEFORE the Java
# backstop threshold, so no arm's brown work ever rides the backstop's
# greenest-DC router (the confound behind naive's fake 0.5-1.6pp edge).
# The Java backstop stays armed as the final safety net; forced counts are
# reported and should read ~0 in every arm.
RELEASE_EPS_S = 30.0
W_PER_PE_DYN = 2.541          # job counterfactual increment (RS500A W/PE)
ANCHORS = (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)
# 价值检查用的子集(默认不启用,判决跑必须用全部 10 个锚点)。
if os.environ.get("GWO1_ANCHORS", "").strip():
    ANCHORS = tuple(int(x) for x in os.environ["GWO1_ANCHORS"].split(","))
MIX = ((0.8, 300.0, 1500.0), (0.2, 2700.0, 4500.0))   # registered mixture

SCHEDULES = {
    "cal": ("experiment_sqt2_noforecast", "calib/sqt2_schedule.json"),
    "ho": ("experiment_sqt2ho_noforecast", "calib/sqt2ho_schedule.json"),
}
BLIND_CK = ("logs/v32_nofc600_s1/multidc_gtrxl_training/"
            "PPO_multidc_env_70179_00000_0_2026-08-17_01-34-42/checkpoint_000010")


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


class PowerAwareAllocator:
    """Sentinel spatial base (Codex layer 1, 2026-08-18).

    Primary: route to the DC with the largest green headroom NET of the
    job's counterfactual power increment (pes * W_PER_PE_DYN), committing
    both the watts and one PE on a per-step ledger. Trough fallback: among
    DCs with ledger capacity, lowest brown carbon factor first, queue+taken
    as tie-break - burst spreading by preference, not by index order.
    Overflow: least loaded queue overall (completion protection)."""

    def __init__(self, green_w, power_w, pes_free, queue_sizes, brown_factor):
        self.head = np.asarray(green_w, dtype=float) - np.asarray(power_w, dtype=float)
        self.ledger = np.asarray(pes_free, dtype=float).copy()
        self.queue = np.asarray(queue_sizes, dtype=float).copy()
        self.brown = np.asarray(brown_factor, dtype=float)

    def take(self, pes: float = 1.0) -> int:
        need = max(1.0, pes)          # PE ledger charged at the job's width
        dp = need * W_PER_PE_DYN
        cand = np.where((self.ledger >= need) & (self.head >= dp))[0]
        if cand.size:
            d = int(cand[np.argmax(self.head[cand])])
            self.head[d] -= dp
            self.ledger[d] -= need
            self.queue[d] += 1
            return d
        cand = np.where(self.ledger >= need)[0]
        if cand.size:
            score = self.brown[cand] * 1000.0 + self.queue[cand]
            d = int(cand[np.argmin(score)])
            self.ledger[d] -= need
            self.queue[d] += 1
            return d
        d = int(np.argmin(self.queue))
        self.queue[d] += 1
        return d


class SpillShield:
    """Forecast-blind work-conserving capacity shield (Codex ruling, 2026-08-18 pm).

    The frozen PPO's all-DC0 routing is carbon-optimal only with the
    completion constraint relaxed; under the 99.5%@7200 contract it queues
    whale jobs past the horizon. The shield keeps the PPO target whenever
    that DC can IMMEDIATELY fit the job (VM-level free PEs, per-step
    ledger) and otherwise spills to a DC that can: lowest brown carbon
    factor first, then lowest incremental brown power (job increment minus
    remaining green headroom), then shortest queue. If NO DC fits, fall
    back to the PPO choice - the shield redirects, never blocks. It reads
    only current capacity/power/queue: no forecast, no trough state, and
    every temporal arm shares it identically."""

    def __init__(self, avail_pes, green_w, power_w, queue_sizes, brown):
        self.free = np.asarray(avail_pes, dtype=float).copy()
        self.head = (np.asarray(green_w, dtype=float)
                     - np.asarray(power_w, dtype=float))
        self.queue = np.asarray(queue_sizes, dtype=float).copy()
        self.brown = np.asarray(brown, dtype=float)
        self.spills = 0

    def _commit(self, d: int, need: float):
        self.free[d] -= need
        self.head[d] -= need * W_PER_PE_DYN
        self.queue[d] += 1

    def route(self, ppo_dc: int, pes: float = 1.0) -> int:
        need = max(1.0, pes)
        if self.free[ppo_dc] >= need:
            self._commit(int(ppo_dc), need)
            return int(ppo_dc)
        cand = np.where(self.free >= need)[0]
        if cand.size == 0:
            self.queue[int(ppo_dc)] += 1        # rule 4: nobody fits -> PPO
            return int(ppo_dc)
        dp = need * W_PER_PE_DYN
        brown_inc = np.maximum(0.0, dp - np.maximum(0.0, self.head[cand]))
        j = min(range(cand.size),
                key=lambda i: (self.brown[cand[i]], brown_inc[i],
                               self.queue[cand[i]]))
        d = int(cand[j])
        self._commit(d, need)
        self.spills += 1
        return d


def load_frozen_gate(repo: pathlib.Path):
    """(q_star, comparator) frozen by the calibration scripts - never a CLI.

    Prefers the carbon/SLA freeze (comparator_v2 / q_star_carbon, Codex
    P0-2) when sqt2_blind_freeze.py has run; falls back to the offline
    accuracy freeze otherwise. comparator_v2 == null (no candidate met the
    dual SLA) is an escalation state: refuse to run a formal verdict."""
    art = json.loads((repo / "calib/sqt2_hazard_freeze.json").read_text())
    if "comparator_v2" in art:
        comp = art["comparator_v2"]
        if comp is None:
            raise RuntimeError("comparator_v2 is null (no blind candidate met "
                               "the dual SLA) - Codex ruling needed before "
                               "any formal prescreen run")
        q = float(art.get("q_star_carbon", art["q_star"]))
        return q, ("hazard" if comp.startswith("hazard") else "naive")
    return float(art["q_star"]), str(art["comparator"])


def gate_flags(mode: str, g, batch: int, ttd_scale: float, t: int,
               in_trough: bool, age: float, residual: float,
               hazard_q: float, backlog_scale: float) -> np.ndarray:
    """Temporal hold decisions; identical budget math for every arm."""
    flags = np.zeros(batch, dtype=bool)
    _wide = os.environ.get("GWO1_WIDE_DOMAIN", "").strip() == "1"
    if mode == "nowait" or t >= HORIZON_S or (not in_trough and not _wide):
        return flags
    backlog = int(_arr(g, "global_deferred_count", 1)[0] * backlog_scale)
    if backlog >= BACKLOG_CAP:
        return flags
    mi = _arr(g, "batch_cloudlet_mi", batch)
    pes = _arr(g, "batch_cloudlet_pes", batch)
    ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
    present = _arr(g, "batch_cloudlet_deadline_present", batch)
    for i in range(batch):
        if mi[i] <= 0 or present[i] <= 0.5:
            continue
        runtime = mi[i] / (max(1.0, pes[i]) * MIPS)
        budget = effective_budget(ttd[i], runtime, MARGIN_S, HORIZON_S - t)
        if budget <= RELEASE_EPS_S:
            continue
        if mode == "naive":
            flags[i] = True
        elif mode == "hazard":
            flags[i] = hazard_p_end_within(age, budget) >= hazard_q
        elif mode == "clairvoyant":
            flags[i] = residual <= budget
    return flags


def run_episode(env, cfg, base, mode, tindex, episode_index, head,
                hazard_q, brown):
    obs, info = env.reset(seed=1)
    if head is not None:
        head.reset()
    off_range = int(cfg.get("green_episode_offset_range", 0) or 0)
    expected = episode_offset(episode_index, off_range)
    actual = int(getattr(env, "_green_episode_offset_rows", 0))
    if actual != expected:
        raise RuntimeError(f"offset misalign ep{episode_index}: {actual}!={expected}")
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec", 3600.0)))
    backlog_scale = float(cfg.get("obs_v31_global_deferred_count_scale", 2000.0))
    done, t, defers, backlog_max, spills = False, 0, 0, 0, 0
    compl_7200 = carbon_7200 = forced_7200 = ontime_7200 = None
    while not done:
        g = obs["global"]
        row = WARMUP_ROWS + actual + t
        in_trough, age, residual = tindex.query(row)
        mi = _arr(g, "batch_cloudlet_mi", batch)
        pes = _arr(g, "batch_cloudlet_pes", batch)
        hold = gate_flags(mode, g, batch, ttd_scale, t, in_trough, age,
                          residual, hazard_q, backlog_scale)
        backlog_max = max(backlog_max, int(
            _arr(g, "global_deferred_count", 1)[0] * backlog_scale))
        if base == "ppo":
            route, _ = head.step(g)
            shield = SpillShield(_arr(g, "dc_available_pes", num_dc),
                                 _arr(g, "dc_current_green_power_w", num_dc),
                                 _arr(g, "dc_current_power_w", num_dc),
                                 _arr(g, "dc_queue_sizes", num_dc), brown)
            actions = [int(num_dc) if (hold[i] and mi[i] > 0)
                       else (shield.route(int(route[i]), pes[i])
                             if mi[i] > 0 else int(route[i]))
                       for i in range(batch)]
            spills += shield.spills
        else:
            alloc = PowerAwareAllocator(
                _arr(g, "dc_current_green_power_w", num_dc),
                _arr(g, "dc_current_power_w", num_dc),
                _arr(g, "dc_available_pes", num_dc),
                _arr(g, "dc_queue_sizes", num_dc), brown)
            actions = [int(num_dc) if (hold[i] and mi[i] > 0)
                       else alloc.take(pes[i]) for i in range(batch)]
        defers += int(sum(1 for i in range(batch) if hold[i] and mi[i] > 0))
        local = {dc: drain_action(env.get_local_action_masks(dc))
                 for dc in range(num_dc)}
        obs, _, term, trunc, info = env.step({"global": actions, "local": local})
        done = term or trunc
        t += 1
        if t == int(HORIZON_S):
            ges = info.get("global_energy_stats") or {}
            compl_7200 = float(ges.get("completion_rate_mi", 0.0) or 0.0)
            carbon_7200 = float(ges.get("total_carbon_emission_kg", 0.0) or 0.0)
            forced_7200 = int(ges.get("deadline_forced_count", 0) or 0)
            ontime_7200 = float(ges.get("ontime_mi_share", 1.0) or 1.0)
    m = collect_metrics(info, num_dc)
    if compl_7200 is None:      # episode drained before the horizon
        compl_7200 = float(m.get("completion_rate_mi", 0.0) or 0.0)
        carbon_7200 = float(m.get("total_carbon_kg", 0.0) or 0.0)
        forced_7200 = int(m.get("deadline_forced_count", 0) or 0)
        ontime_7200 = float(m.get("ontime_mi_share", 1.0) or 1.0)
    forced = int(m.get("deadline_forced_count", 0) or 0)
    return {"base": base, "mode": mode, "episode_index": episode_index,
            "green_offset": actual, "steps": t, "defer_slots": defers,
            "spill_slots": spills,
            "backlog_max": backlog_max,
            "deadline_forced_count": forced,
            "forced_at_7200": forced_7200,
            "ontime_mi_share": float(m.get("ontime_mi_share", 1.0) or 1.0),
            "ontime_at_7200": ontime_7200,
            "total_carbon_kg": float(m.get("total_carbon_kg", 0.0) or 0.0),
            "completion_rate_mi": float(m.get("completion_rate_mi", 0.0) or 0.0),
            "completion_at_7200": compl_7200, "carbon_at_7200": carbon_7200}


def pair_verdict_dual(gate_rec: dict, base_rec: dict,
                      contract: float = CONTRACT) -> dict:
    """Dual-horizon validity (Codex P0-1, 2026-08-18 afternoon).

    A pair is valid ONLY if BOTH arms meet the completion contract at ALL
    THREE accounts (SQT2.3 triple contract, Codex 2026-08-19):
    completion@7200 (the training-horizon SLA), terminal completion (the
    drain account - tail energy cannot vanish), and ontime_mi_share (per-job
    punctuality - blind maximal deferral must pay for lateness instead of
    hiding it). Carbon primary stays terminal; carbon@7200 is reported. A
    win produced by the reference arm failing ontime while carbon ties is a
    FEASIBILITY advantage and must be reported as such, never as carbon."""
    v = pair_verdict(gate_rec, base_rec, contract)
    v["valid_terminal"] = v["valid"]
    v["valid_7200"] = (gate_rec["completion_at_7200"] >= contract
                       and base_rec["completion_at_7200"] >= contract)
    v["valid_ontime"] = (gate_rec.get("ontime_mi_share", 1.0) >= contract
                         and base_rec.get("ontime_mi_share", 1.0) >= contract)
    v["valid"] = bool(v["valid_terminal"] and v["valid_7200"]
                      and v["valid_ontime"])
    b7, g7 = base_rec["carbon_at_7200"], gate_rec["carbon_at_7200"]
    v["rel_delta_c7200"] = (g7 - b7) / max(1e-9, b7)
    return v


def paired_stats(gate_recs: List[dict], base_recs: Dict[int, dict]):
    """Validity-gated pairs + sign counts vs an arbitrary reference arm."""
    pv = [dict(pair_verdict_dual(r, base_recs[r["episode_index"]]),
               episode_index=r["episode_index"]) for r in gate_recs]
    valid = [v for v in pv if v["valid"]]
    neg = sum(1 for v in valid if v["rel_delta_raw"] < 0)
    return {"valid_pairs": len(valid), "invalid_pairs": len(pv) - len(valid),
            "invalid_7200": sum(1 for v in pv if not v["valid_7200"]),
            "invalid_terminal": sum(1 for v in pv if not v["valid_terminal"]),
            "invalid_ontime": sum(1 for v in pv if not v["valid_ontime"]),
            "neg_signs": neg,
            "median_rel_delta": (float(np.median([v["rel_delta_raw"]
                                                  for v in valid]))
                                 if valid else None),
            "median_rel_delta_c7200": (float(np.median([v["rel_delta_c7200"]
                                                        for v in valid]))
                                       if valid else None)}


def final_verdict(vs_nowait: dict, vs_comp: dict, comparator: str) -> dict:
    """Codex layer 4: all five clauses, thresholds locked before the run."""
    ok_pairs = vs_comp["valid_pairs"] >= 8 and vs_nowait["valid_pairs"] >= 8
    m_now = vs_nowait["median_rel_delta"]
    m_cmp = vs_comp["median_rel_delta"]
    ok_now = m_now is not None and m_now <= -0.08
    ok_cmp = m_cmp is not None and m_cmp <= -0.05
    ok_signs = (vs_nowait["neg_signs"] >= 8 and vs_comp["neg_signs"] >= 8)
    return {"comparator": comparator,
            "clairvoyant_vs_nowait_median": m_now,
            "clairvoyant_vs_comparator_median": m_cmp,
            "pass_pairs": ok_pairs, "pass_vs_nowait_8pct": ok_now,
            "pass_vs_comparator_5pct": ok_cmp, "pass_signs_8of10": ok_signs,
            "PASS": bool(ok_pairs and ok_now and ok_cmp and ok_signs)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=("cal", "ho"), default="cal")
    ap.add_argument("--bases", default="ppo,greedy")
    ap.add_argument("--sentinel-arms", default="nowait,clairvoyant")
    ap.add_argument("--drain-horizon", type=int, default=10000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    repo = pathlib.Path(__file__).resolve().parent
    experiment, sched_art = SCHEDULES[args.schedule]
    art = json.loads((repo / sched_art).read_text())
    tindex = TroughIndex(art["troughs"])
    hazard_q, comparator = load_frozen_gate(repo)
    print(f"[SQT2PS] schedule={args.schedule} experiment={experiment} "
          f"frozen hazard q*={hazard_q} comparator={comparator}", flush=True)
    cfg = load_config(experiment)
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/sqt2_gateway")
    cfg.setdefault("output_dir", "/tmp/sqt2_gateway")
    pathlib.Path("/tmp/sqt2_gateway").mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = int(args.drain_horizon)
    brown = [float(d.get("brown_carbon_factor", 0.5)) for d in cfg["datacenters"]]

    all_arms = ("nowait", "naive", "hazard", "clairvoyant")
    plans = []
    for b in [x.strip() for x in args.bases.split(",") if x.strip()]:
        arms = all_arms if b == "ppo" else tuple(
            a.strip() for a in args.sentinel_arms.split(",") if a.strip())
        plans.append((b, arms))

    records = []
    aborted = False
    for base, arms in plans:
        head = ModuleHead(repo / BLIND_CK) if base == "ppo" else None
        for arm in arms:
            env = HierarchicalMultiDCEnv(dict(cfg))
            try:
                next_k = 0
                for k in ANCHORS:
                    while next_k < k:        # fast-forward the offset counter
                        env.reset(seed=1)
                        next_k += 1
                    rec = run_episode(env, cfg, base, arm, tindex, k, head,
                                      hazard_q, brown)
                    next_k = k + 1
                    records.append(rec)
                    print(f"[SQT2PS {base}:{arm:11s} ep{k:>3} "
                          f"off={rec['green_offset']:>6}] "
                          f"carbon={rec['total_carbon_kg']:.4f} "
                          f"compl={rec['completion_rate_mi']:.4f} "
                          f"c@7200={rec['completion_at_7200']:.4f} "
                          f"ontime={rec['ontime_mi_share']:.4f} "
                          f"defer={rec['defer_slots']} "
                          f"spill={rec['spill_slots']} "
                          f"forced={rec['deadline_forced_count']} "
                          f"blmax={rec['backlog_max']}", flush=True)
                    if (base == "ppo" and arm == "nowait"
                            and rec["completion_at_7200"] < CONTRACT):
                        print(f"[SQT2PS ABORT] ppo control below contract at "
                              f"ep{k} (c@7200={rec['completion_at_7200']:.4f})"
                              f" - capacity or spatial base FAILS honestly, "
                              f"no fallback", flush=True)
                        aborted = True
                        break
            finally:
                try:
                    env.close()
                except Exception:
                    pass
            if aborted:
                break
        if aborted:
            break

    out = {"anchors": list(ANCHORS), "schedule": args.schedule,
           "experiment": experiment, "hazard_q": hazard_q,
           "comparator": comparator, "blind_ck": BLIND_CK,
           "records": records}
    if aborted:
        out["verdict"] = "CONTROL_CEILING_FAIL"
    else:
        ppo = [r for r in records if r["base"] == "ppo"]
        nowait = {r["episode_index"]: r for r in ppo if r["mode"] == "nowait"}
        comp_mode = "hazard" if comparator == "hazard" else "naive"
        comp = {r["episode_index"]: r for r in ppo if r["mode"] == comp_mode}
        stats = {}
        for arm in ("naive", "hazard", "clairvoyant"):
            recs = [r for r in ppo if r["mode"] == arm]
            if recs:
                stats[f"{arm}_vs_nowait"] = paired_stats(recs, nowait)
        clair = [r for r in ppo if r["mode"] == "clairvoyant"]
        if clair and comp:
            stats["clairvoyant_vs_comparator"] = paired_stats(clair, comp)
            out["final"] = final_verdict(stats["clairvoyant_vs_nowait"],
                                         stats["clairvoyant_vs_comparator"],
                                         comparator)
        out["stats"] = stats
        for k_, v in stats.items():
            med = v["median_rel_delta"]
            print(f"[SQT2PS VERDICT {k_}] valid {v['valid_pairs']} "
                  f"median {med if med is not None else float('nan'):+.4f} "
                  f"neg {v['neg_signs']}/{v['valid_pairs']}", flush=True)
        if "final" in out:
            f = out["final"]
            print(f"[SQT2PS FINAL] vs-nowait {f['clairvoyant_vs_nowait_median']:+.4f} "
                  f"(<=-8%: {f['pass_vs_nowait_8pct']}) | "
                  f"vs-{comparator} {f['clairvoyant_vs_comparator_median']:+.4f} "
                  f"(<=-5%: {f['pass_vs_comparator_5pct']}) | "
                  f"signs: {f['pass_signs_8of10']} | PASS={f['PASS']}",
                  flush=True)
    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
