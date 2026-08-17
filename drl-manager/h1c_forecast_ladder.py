#!/usr/bin/env python3
"""H1c: the matched forecast-value ladder (Codex merge decision, 2026-08-17).

Six temporal-information arms over ONE frozen blind spatial router. The gate
FORMULA and thresholds are identical everywhere and locked before the run;
only the information source feeding the observation features changes:

    immediate    no gate                        strong no-forecast floor
    persistence  forecast_mode none             degenerates to immediate BY
                                                CONSTRUCTION (blind fill sets
                                                gain=0) - wiring null, must
                                                match immediate bit-for-bit
    clean        TimeCAP deployment forecast    can a real forecast cash in?
    oracle       godeye true future             physical headroom bound
    shuffle      godeye + DC-reversed bins      does the value depend on
    anti         godeye + A-prime mirrored bins the forecast being RIGHT?

Pre-registered gate (locked 2026-08-17, before any run):
    defer iff relative predicted saving (best_now - best_future)/best_now
              >= REL_GAIN_MIN (0.10 = the campaign noise floor)
          and wait budget > 0 under the 7200 s decision horizon (margin 120 s)
          and deferred backlog < 200
Drain window 10000 steps finishes scheduled work only. A pair is valid only
if both arms' terminal completion meets the 99.5% contract (pair_verdict).

Perturbed arms transform the WATT bins at the source (perturb_future_bins),
so every derived job feature (gain, time-to-best, best-future) is computed
from the corrupted trajectory - no clean leakage (test-locked).

Usage:
    .venv/bin/python h1c_forecast_ladder.py --blind-ck <ck> [--episodes 10]
"""
import argparse
import json
import os
import pathlib
import sys
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS, VM_MIPS, _arr, drain_action  # noqa: E402
from teacher_reward_audit import verify_offset  # noqa: E402
from h1_matched_headroom import ModuleHead, blindify, pair_verdict  # noqa: E402
from src.baselines.evaluate import load_config, collect_metrics  # noqa: E402

REL_GAIN_MIN = 0.10        # = pre-registered noise floor; locked before run
DECISION_HORIZON_S = 7200.0
MARGIN_S = 120.0
BACKLOG_CAP = 200
CONTRACT = 0.995

ARMS = ("immediate", "persistence", "clean", "oracle", "shuffle", "anti")
PERTURB_OF = {"shuffle": "shuffle", "anti": "anti"}


def feature_gate_flags(g: Dict[str, np.ndarray], batch: int, ttd_scale: float,
                       t: int, rel_gain_min: float = REL_GAIN_MIN) -> np.ndarray:
    """Pre-registered per-job gate on OBSERVATION features only.

    Reads whatever information set this arm's env computed into the features -
    the gate itself never touches the green series, so swapping the source is
    the only difference between arms.
    """
    mi = _arr(g, "batch_cloudlet_mi", batch)
    pes = _arr(g, "batch_cloudlet_pes", batch)
    ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
    present = _arr(g, "batch_cloudlet_deadline_present", batch)
    best_now = _arr(g, "batch_cloudlet_best_now_carbon", batch)
    best_fut = _arr(g, "batch_cloudlet_best_future_carbon", batch)
    backlog = int(_arr(g, "global_deferred_count", 1)[0] * 2000.0)
    flags = np.zeros(batch, dtype=bool)
    if backlog >= BACKLOG_CAP:
        return flags
    horizon_left = DECISION_HORIZON_S - t
    for i in range(batch):
        if mi[i] <= 0 or present[i] <= 0.5:
            continue
        runtime = mi[i] / (max(1.0, pes[i]) * VM_MIPS)   # Java ledger units
        budget = min(ttd[i], horizon_left) - runtime - MARGIN_S
        if budget <= 0:
            continue
        if best_now[i] <= 1e-9:
            continue
        rel_gain = (best_now[i] - best_fut[i]) / best_now[i]
        flags[i] = rel_gain >= rel_gain_min
    return flags


def run_episode(env, cfg, arm, blind_head, episode_index):
    mode = PERTURB_OF.get(arm)
    if mode:
        os.environ["FORECAST_PERTURB_MODE"] = mode
        os.environ["FORECAST_PERTURB_EPS"] = "1.0"
    else:
        os.environ.pop("FORECAST_PERTURB_MODE", None)
    try:
        obs, info = env.reset(seed=1)
        off_range = int(cfg.get("green_episode_offset_range", 0) or 0)
        offset = verify_offset(env, episode_index, off_range)
        blind_head.reset()
        num_dc = env.num_datacenters
        batch = env.global_routing_batch_size
        green_high = float(cfg.get("obs_green_power_high", 3000.0))
        ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec",
                                           cfg.get("defer_urgency_window_sec", 3600.0))))
        done, t, defers = False, 0, 0
        compl_7200, carbon_7200 = None, None
        gain_checksum = 0.0
        while not done:
            g = obs["global"]
            mi = _arr(g, "batch_cloudlet_mi", batch)
            gain_checksum += float(np.sum(
                _arr(g, "batch_cloudlet_forecast_gain", batch)))
            route, _ = blind_head.step(blindify(g, green_high))
            hold = (np.zeros(batch, dtype=bool) if arm == "immediate"
                    else feature_gate_flags(g, batch, ttd_scale, t))
            actions = [int(num_dc) if (hold[i] and mi[i] > 0) else int(route[i])
                       for i in range(batch)]
            defers += int(sum(1 for i in range(batch)
                              if hold[i] and mi[i] > 0))
            local_actions = {dc: drain_action(env.get_local_action_masks(dc))
                             for dc in range(num_dc)}
            obs, _, term, trunc, info = env.step(
                {"global": actions, "local": local_actions})
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
        return {"mode": arm, "episode_index": episode_index,
                "green_offset": offset, "steps": t, "defer_slots": defers,
                "gain_checksum": round(gain_checksum, 6),
                "total_carbon_kg": float(m.get("total_carbon_kg", 0.0) or 0.0),
                "completion_rate_mi": float(m.get("completion_rate_mi", 0.0) or 0.0),
                "completion_at_7200": compl_7200, "carbon_at_7200": carbon_7200}
    finally:
        os.environ.pop("FORECAST_PERTURB_MODE", None)


def arm_config(arm: str, oracle_cfg: dict, nofc_cfg: dict,
               capacity_w: List[float]) -> dict:
    cfg = dict(nofc_cfg if arm == "persistence" else oracle_cfg)
    if arm == "clean":
        cfg["green_oracle_mode"] = "timecap"
        tc = dict(cfg.get("timecap") or {})
        tc["device"] = "cpu"               # pitfall ledger: MUST be cpu
        cfg["timecap"] = tc
    if arm == "anti":
        cfg["v32_perturb_capacity_w"] = list(capacity_w)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-ck", required=True)
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--drain-horizon", type=int, default=10000)
    ap.add_argument("--demand-model", default="job_counterfactual_v1",
                    help="obs_v32_demand_model for ALL arms (legacy|job_counterfactual_v1)")
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    repo = pathlib.Path(__file__).resolve().parent
    capacity = json.loads((repo / "calib/v3_anti_capacity.json").read_text())[
        "capacity_w_per_dc"]

    def prep(name):
        c = load_config(name)
        c.pop("py4j_port", None)
        c.setdefault("gateway_log_dir", "/tmp/h1c_gateway")
        c.setdefault("output_dir", "/tmp/h1c_gateway")
        c["max_episode_length"] = int(args.drain_horizon)
        c["obs_v32_demand_model"] = args.demand_model
        return c

    pathlib.Path("/tmp/h1c_gateway").mkdir(parents=True, exist_ok=True)
    oracle_cfg = prep("experiment_v3_2_oracle")
    nofc_cfg = prep("experiment_v3_2_noforecast")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    blind_head = ModuleHead(args.blind_ck)

    envs = {a: HierarchicalMultiDCEnv(
        arm_config(a, oracle_cfg, nofc_cfg, capacity)) for a in arms}
    records = []
    try:
        for k in range(args.episodes):
            for a in arms:
                cfg_a = arm_config(a, oracle_cfg, nofc_cfg, capacity)
                rec = run_episode(envs[a], cfg_a, a, blind_head, k)
                records.append(rec)
                print(f"[H1C ep{k} off={rec['green_offset']:>4} {a:11s}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"compl={rec['completion_rate_mi']:.4f} "
                      f"c@7200={rec['carbon_at_7200']:.4f} "
                      f"compl@7200={rec['completion_at_7200']:.4f} "
                      f"defer={rec['defer_slots']}", flush=True)
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    base = {r["episode_index"]: r for r in records if r["mode"] == "immediate"}
    verdicts = {}
    for a in arms:
        if a == "immediate":
            continue
        pv = [dict(pair_verdict(r, base[r["episode_index"]]),
                   episode_index=r["episode_index"])
              for r in records if r["mode"] == a and r["episode_index"] in base]
        valid = [v for v in pv if v["valid"]]
        neg = sum(1 for v in valid if v["rel_delta_raw"] < 0)
        verdicts[a] = {"valid_pairs": len(valid),
                       "invalid_pairs": len(pv) - len(valid),
                       "improve_signs_valid": f"{neg}/{len(valid)}",
                       "median_rel_delta_valid": (float(np.median(
                           [v["rel_delta_raw"] for v in valid])) if valid else None),
                       "pairs": pv}
        v = verdicts[a]
        med = v["median_rel_delta_valid"]
        print(f"[H1C VERDICT {a}] valid {v['valid_pairs']} invalid "
              f"{v['invalid_pairs']} | median "
              f"{med if med is not None else float('nan'):+.4f} "
              f"improves {v['improve_signs_valid']}", flush=True)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(json.dumps(
            {"prereg": {"rel_gain_min": REL_GAIN_MIN,
                        "decision_horizon_s": DECISION_HORIZON_S,
                        "margin_s": MARGIN_S, "backlog_cap": BACKLOG_CAP,
                        "contract": CONTRACT,
                        "capacity_w_per_dc": capacity},
             "records": records, "verdicts": verdicts}, indent=1))


if __name__ == "__main__":
    main()
