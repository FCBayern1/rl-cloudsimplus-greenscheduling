#!/usr/bin/env python3
"""R0: teacher-reward paired audit (docs/V32_POST_GATE2_DECISION.md §2).

Question: does the reward the V3.2A global PPO actually optimized PREFER the
slack-aware teacher's behaviour (-21..-29% carbon @ 100% completion), or does
it prefer the no-defer control? Two mutually exclusive explanations of the
Gate 2 FAIL demand opposite treatments (distill vs fix the reward timescale),
and this paired comparison is the cheapest decisive evidence.

Protocol (locked before running, decision doc §2.1-2.5):
  - experiment_v3_2_oracle for BOTH arms (identical env + reward yardstick;
    the V3.2 spatial term is part of what PPO optimized, so it is included);
  - teacher = slack-aware theta=0.5, control = no-defer, same greedy
    greenest-now-with-capacity route rule, both on fixed drain locals;
  - per step records ONLY r_t = float(rewards["global"]) — local rewards are
    frozen drain scaffolding the PPO never received (mixing them is banned);
  - reports undiscounted sum AND gamma=0.999-discounted return per episode
    (0.999 read from the 300k run's params.json global_policy overrides);
  - offsets follow the production schedule (1009*k mod range); both envs stay
    alive and reset in lockstep; the planner's future-green rows add the SAME
    offset; fail-fast if the env's authoritative offset ever disagrees;
  - S0 sentinel = 1 episode; S1 decision = 3 offsets; S2 arbitration = 6.

Usage:
    .venv/bin/python teacher_reward_audit.py --episodes 1   # S0 sentinel
    .venv/bin/python teacher_reward_audit.py --episodes 3   # S1 decision
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from oracle_slack_planner import (  # noqa: E402
    WARMUP_ROWS, VM_MIPS, _arr, drain_action, load_green_series,
)
from src.baselines.evaluate import load_config, collect_metrics  # noqa: E402

GAMMA_DEFAULT = 0.999   # authoritative: v32_g2_s1 params.json global_policy override

# global_energy_stats keys required for the reward decomposition (§2.3).
# Missing keys are a hard error from S1 on ("三-offset 定案前必须补齐").
DECOMP_KEYS = (
    "ep_carbon_level_term_sum", "ep_completion_term_sum", "ep_spatial_term_sum",
    "defer_urgency_cost_sum", "deadline_forced_count",
    "ep_carbon_raw_kg_sum", "ep_carbon_norm_sum", "ep_carbon_norm_clip_count",
)


def episode_offset(episode_index: int, offset_range: int) -> int:
    """Production closed-book schedule: (1009*k) mod range."""
    return (1009 * episode_index) % offset_range if offset_range > 0 else 0


def verify_offset(env, episode_index: int, offset_range: int) -> int:
    """Fail-fast offset alignment (§2.4): planner schedule vs env authority."""
    expected = episode_offset(episode_index, offset_range)
    actual = int(getattr(env, "_green_episode_offset_rows", 0))
    if actual != expected:
        raise RuntimeError(
            f"offset misalignment at episode {episode_index}: env reports "
            f"{actual}, schedule says {expected} — envs out of lockstep; "
            f"results would compare different green windows. ABORT.")
    return actual


class ReturnAccumulator:
    """Global-only reward accounting. Local rewards NEVER enter (§2.2)."""

    def __init__(self, gamma: float):
        self.gamma = float(gamma)
        self.reward_sum = 0.0
        self.discounted_return = 0.0
        self._discount = 1.0
        self.steps = 0

    def add(self, rewards: Dict[str, Any]) -> float:
        r_t = float(rewards["global"])
        self.reward_sum += r_t
        self.discounted_return += self._discount * r_t
        self._discount *= self.gamma
        self.steps += 1
        return r_t


def run_episode(env, cfg, green: np.ndarray, *, defer_enabled: bool,
                theta: float, margin: float, backlog_cap: int, seed: int,
                episode_index: int, gamma: float) -> Dict[str, Any]:
    """One episode of the slack-aware (or no-defer) policy WITH reward capture.

    Same decision rules as oracle_slack_planner.run(); the future-green lookup
    adds the episode's green offset so multi-offset episodes read the window
    the simulator actually plays.
    """
    obs, info = env.reset(seed=seed)
    off_range = int(cfg.get("green_episode_offset_range", 0) or 0)
    offset = verify_offset(env, episode_index, off_range)

    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    defer_idx = num_dc
    ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec",
                                       cfg.get("defer_urgency_window_sec", 3600.0))))
    acc = ReturnAccumulator(gamma)
    done, t, defers, routes = False, 0, 0, 0
    while not done:
        g = obs["global"]
        row = WARMUP_ROWS + offset + t
        green_now = green[min(row, len(green) - 1)]
        mi = _arr(g, "batch_cloudlet_mi", batch)
        ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
        present = _arr(g, "batch_cloudlet_deadline_present", batch)
        backlog = int(_arr(g, "global_deferred_count", 1)[0]
                      * float(cfg.get("obs_v31_global_deferred_count_scale", 2000.0)))
        green_dc = _arr(g, "dc_current_green_power_w", num_dc)
        pes_free = _arr(g, "dc_available_pes", num_dc)
        order = np.argsort(-green_dc)
        target = int(order[0])
        for d in order:
            if pes_free[int(d)] >= 1:
                target = int(d)
                break
        actions: List[int] = []
        for i in range(batch):
            if mi[i] <= 0:
                actions.append(target)
                continue
            if not defer_enabled:
                actions.append(target); routes += 1
                continue
            runtime = mi[i] / VM_MIPS
            budget = (ttd[i] - runtime - margin) if present[i] > 0.5 else 0.0
            if budget <= 0 or backlog >= backlog_cap:
                actions.append(target); routes += 1
                continue
            horizon = int(min(budget, 3600))
            best_future = green[row:min(row + horizon, len(green))].max(initial=green_now)
            if green_now < theta * best_future:
                actions.append(defer_idx); defers += 1
            else:
                actions.append(target); routes += 1
        local_actions = {dc: drain_action(env.get_local_action_masks(dc))
                         for dc in range(num_dc)}
        obs, rewards, term, trunc, info = env.step(
            {"global": actions, "local": local_actions})
        acc.add(rewards)
        done = term or trunc
        t += 1

    m = collect_metrics(info, num_dc)
    ges = info.get("global_energy_stats") or {}
    decomp = {k: (float(ges[k]) if k in ges else None) for k in DECOMP_KEYS}
    rec: Dict[str, Any] = {
        "arm": "teacher" if defer_enabled else "control",
        "episode_index": episode_index,
        "green_offset": offset,
        "seed": seed,
        "steps": acc.steps,
        "gamma": gamma,
        "global_reward_sum": acc.reward_sum,
        "global_discounted_return": acc.discounted_return,
        "defer_slots": defers,
        "route_slots": routes,
        "total_carbon_kg": float(m.get("total_carbon_kg", 0.0) or 0.0),
        # Contract metric: MI-weighted completion (the iso-completion yardstick).
        # collect_metrics["completion_rate"] is a routed_rate alias and proves
        # only that jobs were dispatched, NOT that they finished — using it
        # here overstated the sentinel (2026-08-15 review catch).
        "completion_rate_mi": float(m.get("completion_rate_mi", 0.0) or 0.0),
        "finished_rate": m.get("finished_rate"),
        "routed_rate": m.get("routed_rate"),
        "total_finished_cloudlets": m.get("total_finished_cloudlets"),
        "total_received_cloudlets": m.get("total_received_cloudlets"),
        "carbon_per_completion_mi": m.get("carbon_per_completion_mi"),
        "green_ratio": float(m.get("green_ratio", 0.0) or 0.0),
        "decomposition": decomp,
    }
    return rec


def paired_delta(teacher: Dict[str, Any], control: Dict[str, Any]) -> Dict[str, Any]:
    keys = ("global_reward_sum", "global_discounted_return", "total_carbon_kg")
    d = {f"d_{k}": teacher[k] - control[k] for k in keys}
    d["episode_index"] = teacher["episode_index"]
    d["green_offset"] = teacher["green_offset"]
    d["discounted_sign"] = ("teacher_higher" if d["d_global_discounted_return"] > 0
                            else "teacher_lower")
    return d


COMPLETION_CONTRACT = 0.995   # iso-completion gate on completion_rate_mi


def branch_verdict(records: List[Dict[str, Any]],
                   deltas: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Decision-doc §3 branch table on MI-completion-valid episodes only.

    Validity gate PER EPISODE: both arms completion_rate_mi >= 0.995 and the
    teacher's carbon lower. Verdict needs every episode valid (an invalid
    episode means the comparison cannot answer alignment -> STOP per §3 row 5);
    branch by discounted-sign unanimity/majority.
    """
    by_ep: Dict[int, Dict[str, Dict[str, Any]]] = {}
    for r in records:
        by_ep.setdefault(r["episode_index"], {})[r["arm"]] = r
    invalid = []
    for k, pair in sorted(by_ep.items()):
        t, c = pair.get("teacher"), pair.get("control")
        if not t or not c:
            invalid.append((k, "missing arm"))
            continue
        if t["completion_rate_mi"] < COMPLETION_CONTRACT:
            invalid.append((k, f"teacher completion_rate_mi {t['completion_rate_mi']:.4f}"))
        if c["completion_rate_mi"] < COMPLETION_CONTRACT:
            invalid.append((k, f"control completion_rate_mi {c['completion_rate_mi']:.4f}"))
        if t["total_carbon_kg"] >= c["total_carbon_kg"]:
            invalid.append((k, "teacher carbon not lower"))
    if invalid:
        return {"branch": "STOP", "reason": invalid}
    n = len(deltas)
    higher = sum(1 for d in deltas if d["discounted_sign"] == "teacher_higher")
    undisc_higher = sum(1 for d in deltas if d["d_global_reward_sum"] > 0)
    if higher == n:
        return {"branch": "L", "action": "V3.2B distillation",
                "reason": f"discounted return teacher_higher {higher}/{n}"}
    if higher == 0:
        sub = ("gamma_timescale" if undisc_higher == n else
               "objective_composition" if undisc_higher == 0 else "mixed")
        return {"branch": "R", "action": f"V3.2C reward fix ({sub})",
                "reason": f"discounted return teacher_lower {n}/{n}, "
                          f"undiscounted teacher_higher {undisc_higher}/{n}"}
    return {"branch": "WAIT", "action": "S2 six-offset arbitration",
            "reason": f"split signs {higher}/{n}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--experiment", default="experiment_v3_2_oracle")
    ap.add_argument("--episodes", type=int, default=1,
                    help="1=S0 sentinel, 3=S1 decision, 6=S2 arbitration")
    ap.add_argument("--theta", type=float, default=0.5)
    ap.add_argument("--margin", type=float, default=120.0)
    ap.add_argument("--backlog-cap", type=int, default=400)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--gamma", type=float, default=GAMMA_DEFAULT)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv

    cfg = load_config(args.experiment)
    cfg.pop("py4j_port", None)   # standalone boot: fresh gateway on a free port
    cfg.setdefault("gateway_log_dir", "/tmp/audit_gateway")
    cfg.setdefault("output_dir", "/tmp/audit_gateway")
    Path("/tmp/audit_gateway").mkdir(parents=True, exist_ok=True)
    if not cfg.get("obs_v31_features"):
        sys.exit("needs obs_v31_features=true (per-slot deadline features)")
    green = load_green_series(cfg)

    print(f"=== R0 teacher-reward paired audit ({args.experiment}, "
          f"theta={args.theta}, gamma={args.gamma}, "
          f"episodes={args.episodes}) ===")

    # Two envs, kept alive; reset in lockstep per episode index (§2.4).
    envs = {"control": HierarchicalMultiDCEnv(dict(cfg)),
            "teacher": HierarchicalMultiDCEnv(dict(cfg))}
    records: List[Dict[str, Any]] = []
    deltas: List[Dict[str, Any]] = []
    try:
        for k in range(args.episodes):
            pair = {}
            for arm, env in envs.items():
                rec = run_episode(
                    env, cfg, green,
                    defer_enabled=(arm == "teacher"), theta=args.theta,
                    margin=args.margin, backlog_cap=args.backlog_cap,
                    seed=args.seed, episode_index=k, gamma=args.gamma)
                records.append(rec)
                pair[arm] = rec
                print(f"[ep{k} off={rec['green_offset']:>4} {arm:7s}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"compl_mi={rec['completion_rate_mi']:.4f} "
                      f"finished={rec['total_finished_cloudlets']}/"
                      f"{rec['total_received_cloudlets']} "
                      f"R={rec['global_reward_sum']:.1f} "
                      f"Rdisc={rec['global_discounted_return']:.2f} "
                      f"defer={rec['defer_slots']}")
            d = paired_delta(pair["teacher"], pair["control"])
            deltas.append(d)
            print(f"[ep{k} PAIRED] dCarbon={d['d_total_carbon_kg']:+.4f} "
                  f"dR={d['d_global_reward_sum']:+.1f} "
                  f"dRdisc={d['d_global_discounted_return']:+.2f} "
                  f"-> {d['discounted_sign']}")
    finally:
        for env in envs.values():
            try:
                env.close()
            except Exception:
                pass

    out = {"protocol": {"experiment": args.experiment, "theta": args.theta,
                        "margin": args.margin, "backlog_cap": args.backlog_cap,
                        "seed": args.seed, "gamma": args.gamma},
           "records": records, "paired_deltas": deltas}
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(out, indent=1))
        print(f"json -> {args.json_out}")

    # Decomposition completeness (hard requirement from S1 on).
    missing = [k for r in records for k, v in r["decomposition"].items() if v is None]
    if missing and args.episodes > 1:
        sys.exit(f"decomposition keys missing from global_energy_stats: "
                 f"{sorted(set(missing))} — jar too old? S1 requires them.")

    signs = [d["d_global_discounted_return"] > 0 for d in deltas]
    print(f"\ndiscounted-return sign count: teacher_higher {sum(signs)}/{len(signs)}")
    verdict = branch_verdict(records, deltas)
    print(f"branch verdict: {verdict}")
    if args.json_out:
        out["verdict"] = verdict
        Path(args.json_out).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
