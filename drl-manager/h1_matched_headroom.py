#!/usr/bin/env python3
"""H1: matched-temporal headroom audit (Codex decomposition plan, 2026-08-17).

Question: on top of the STRONGEST no-forecast spatial router (the blind PPO),
how much carbon does a forecast-driven temporal controller actually buy?

Design: ONE spatial router for every arm - the blind PPO's route logits,
always fed a BLINDIFIED view of the observation (its native training
distribution, byte-matching the env's forecast_mode=none fills). Only the
temporal controller varies:

    immediate   never defer                       (strong no-forecast floor)
    blindgate   blind module's own p_hold > 0.5   (current blind policy)
    oraclegate  slack-aware teacher rule, CSV godeye (explicit forecast bound)
    ftgate      FT module's p_hold > 0.5 on the full oracle view

Protocol: env = experiment_v3_2_oracle, horizon 10000 (drain-to-completion
diagnosis; completion@7200 recorded for the contract reference), 10 paired
offsets (production schedule), drain locals. Per-offset paired deltas vs the
immediate arm, sign counts, medians - no pooled-mean-only reporting.

Usage:
    .venv/bin/python h1_matched_headroom.py --blind-ck <ck> --ft-ck <ck> \
        [--episodes 10] [--arms immediate,blindgate,oraclegate,ftgate]
"""
import argparse
import json
import pathlib
import sys
from typing import Dict, List

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS, VM_MIPS, _arr, drain_action, load_green_series  # noqa: E402
from teacher_reward_audit import effective_budget, episode_offset, verify_offset  # noqa: E402
from probe_forecast_sensitivity import load_module  # noqa: E402
from src.baselines.evaluate import load_config, collect_metrics  # noqa: E402

TEACHER = {"theta": 0.6, "margin": 120.0, "backlog_cap": 200}


def blindify(obs: Dict[str, np.ndarray], green_high: float) -> Dict[str, np.ndarray]:
    """Rewrite the oracle view into the blind arm's information set.

    Byte-matches hierarchical_multidc_env._v32_apply_blind_persistence plus
    the pre-registered blind job tuple (gain=0, time=1, best_future=best_now).
    """
    o = {k: (np.asarray(v).copy() if isinstance(v, (np.ndarray, list)) else v)
         for k, v in obs.items()}
    cur = np.asarray(o.get("dc_current_green_power_w", np.zeros(1)),
                     dtype=np.float64).reshape(-1)
    cn = np.clip(cur / max(1e-9, green_high), 0.0, 1.0).astype(np.float32)
    o["dc_future_short_mean"] = cn.copy()
    o["dc_future_short_trend"] = np.zeros_like(cn)
    o["dc_future_long_mean"] = cn.copy()
    o["dc_future_long_peak_timing"] = np.full_like(cn, 0.5)
    if "batch_cloudlet_forecast_gain" in o:
        o["batch_cloudlet_forecast_gain"] = np.zeros_like(
            np.asarray(o["batch_cloudlet_forecast_gain"], dtype=np.float32))
        o["batch_cloudlet_time_to_best_green"] = np.ones_like(
            np.asarray(o["batch_cloudlet_time_to_best_green"], dtype=np.float32))
        o["batch_cloudlet_best_future_carbon"] = np.asarray(
            o["batch_cloudlet_best_now_carbon"], dtype=np.float32).copy()
    return o


class ModuleHead:
    """Recurrent-state-carrying wrapper: route argmax + p_hold per slot."""

    def __init__(self, ckpt: pathlib.Path):
        self.module = load_module(pathlib.Path(ckpt))
        self.n_slots = int(getattr(self.module, "num_batch_slots", 128))
        self.state = None

    def reset(self):
        self.state = None

    def step(self, obs: Dict[str, np.ndarray]):
        from ray.rllib.core.columns import Columns
        batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                               for k, v in obs.items()
                               if np.asarray(v).dtype.kind in "ifub"}}
        state = self.state
        if state is None:
            init = self.module.get_initial_state()
            if init:
                state = {k: torch.as_tensor(np.asarray(v))[None, ...]
                         for k, v in init.items()}
        if state:
            batch[Columns.STATE_IN] = state
        with torch.no_grad():
            out = self.module.forward_inference(batch)
        self.state = out.get(Columns.STATE_OUT) or None
        logits = out[Columns.ACTION_DIST_INPUTS].detach().reshape(-1)
        n_opt = logits.numel() // self.n_slots
        z = logits.reshape(self.n_slots, n_opt)
        route = z[:, :-1].argmax(-1).numpy()
        p_hold = z[:, -1].exp().numpy()
        return route, p_hold


def assemble_actions(mode: str, route: np.ndarray, mi: np.ndarray,
                     p_blind: np.ndarray, p_ft: np.ndarray,
                     teacher_defer: np.ndarray, defer_idx: int) -> List[int]:
    """Pure action assembly: shared route choice, mode-specific temporal."""
    acts = []
    for i in range(len(route)):
        if mi[i] <= 0:
            acts.append(int(route[i]))
            continue
        if mode == "immediate":
            hold = False
        elif mode == "blindgate":
            hold = p_blind[i] > 0.5
        elif mode == "ftgate":
            hold = p_ft[i] > 0.5
        elif mode == "oraclegate":
            hold = bool(teacher_defer[i])
        else:
            raise ValueError(mode)
        acts.append(defer_idx if hold else int(route[i]))
    return acts


def teacher_defer_flags(g, green, row, batch, num_dc, ttd_scale, cfg,
                        horizon_sec, t):
    """Slack-aware teacher rule per slot (same maths as the R0 audit)."""
    mi = _arr(g, "batch_cloudlet_mi", batch)
    ttd = _arr(g, "batch_cloudlet_time_to_deadline", batch) * ttd_scale
    present = _arr(g, "batch_cloudlet_deadline_present", batch)
    backlog = int(_arr(g, "global_deferred_count", 1)[0]
                  * float(cfg.get("obs_v31_global_deferred_count_scale", 2000.0)))
    green_now = green[min(row, len(green) - 1)]
    flags = np.zeros(batch, dtype=bool)
    if backlog >= TEACHER["backlog_cap"]:
        return flags
    for i in range(batch):
        if mi[i] <= 0 or present[i] <= 0.5:
            continue
        runtime = mi[i] / VM_MIPS
        budget = effective_budget(ttd[i], runtime, TEACHER["margin"],
                                  horizon_sec - t)
        if budget <= 0:
            continue
        horizon = int(min(budget, 3600))
        best_future = green[row:min(row + horizon, len(green))].max(initial=green_now)
        flags[i] = green_now < TEACHER["theta"] * best_future
    return flags


def run_episode(env, cfg, green, mode, blind_head, ft_head, episode_index):
    obs, info = env.reset(seed=1)
    off_range = int(cfg.get("green_episode_offset_range", 0) or 0)
    offset = verify_offset(env, episode_index, off_range)
    blind_head.reset()
    if ft_head is not None:
        ft_head.reset()
    num_dc = env.num_datacenters
    batch = env.global_routing_batch_size
    green_high = float(cfg.get("obs_green_power_high", 3000.0))
    ttd_scale = max(1.0, float(cfg.get("obs_v31_deadline_scale_sec",
                                       cfg.get("defer_urgency_window_sec", 3600.0))))
    horizon_sec = (float(cfg.get("max_episode_length", 7200))
                   * float(cfg.get("simulation_timestep", 1.0)))
    done, t, defers, compl_7200 = False, 0, 0, None
    while not done:
        g = obs["global"]
        row = WARMUP_ROWS + offset + t
        mi = _arr(g, "batch_cloudlet_mi", batch)
        route, p_blind = blind_head.step(blindify(g, green_high))
        p_ft = np.zeros(batch)
        if mode == "ftgate":
            _, p_ft = ft_head.step(g)
        td = (teacher_defer_flags(g, green, row, batch, num_dc, ttd_scale,
                                  cfg, horizon_sec, t)
              if mode == "oraclegate" else np.zeros(batch, dtype=bool))
        actions = assemble_actions(mode, route, mi, p_blind, p_ft, td, num_dc)
        defers += sum(1 for i, a in enumerate(actions)
                      if a == num_dc and mi[i] > 0)
        local_actions = {dc: drain_action(env.get_local_action_masks(dc))
                         for dc in range(num_dc)}
        obs, _, term, trunc, info = env.step(
            {"global": actions, "local": local_actions})
        done = term or trunc
        t += 1
        if t == 7200:
            ges = info.get("global_energy_stats") or {}
            compl_7200 = float(ges.get("completion_rate_mi", 0.0) or 0.0)
    m = collect_metrics(info, num_dc)
    if compl_7200 is None:            # episode finished before 7200
        compl_7200 = float(m.get("completion_rate_mi", 0.0) or 0.0)
    return {"mode": mode, "episode_index": episode_index, "green_offset": offset,
            "steps": t, "defer_slots": defers,
            "total_carbon_kg": float(m.get("total_carbon_kg", 0.0) or 0.0),
            "completion_rate_mi": float(m.get("completion_rate_mi", 0.0) or 0.0),
            "completion_at_7200": compl_7200}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--blind-ck", required=True)
    ap.add_argument("--ft-ck", default=None)
    ap.add_argument("--experiment", default="experiment_v3_2_oracle")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--arms", default="immediate,blindgate,oraclegate,ftgate")
    ap.add_argument("--max-episode-length", type=int, default=10000)
    ap.add_argument("--json-out", default=None)
    args = ap.parse_args()

    from gym_cloudsimplus.envs.hierarchical_multidc_env import HierarchicalMultiDCEnv
    cfg = load_config(args.experiment)
    cfg.pop("py4j_port", None)
    cfg.setdefault("gateway_log_dir", "/tmp/h1_gateway")
    cfg.setdefault("output_dir", "/tmp/h1_gateway")
    pathlib.Path("/tmp/h1_gateway").mkdir(parents=True, exist_ok=True)
    cfg["max_episode_length"] = int(args.max_episode_length)
    green = load_green_series(cfg)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    if "ftgate" in arms and not args.ft_ck:
        sys.exit("ftgate arm needs --ft-ck")

    blind_head = ModuleHead(args.blind_ck)
    ft_head = ModuleHead(args.ft_ck) if args.ft_ck else None
    records = []
    # one env per arm, reset in lockstep episode order (offset parity)
    envs = {a: HierarchicalMultiDCEnv(dict(cfg)) for a in arms}
    try:
        for k in range(args.episodes):
            for a in arms:
                rec = run_episode(envs[a], cfg, green, a, blind_head, ft_head, k)
                records.append(rec)
                print(f"[H1 ep{k} off={rec['green_offset']:>4} {a:10s}] "
                      f"carbon={rec['total_carbon_kg']:.4f} "
                      f"compl={rec['completion_rate_mi']:.4f} "
                      f"compl@7200={rec['completion_at_7200']:.4f} "
                      f"defer={rec['defer_slots']}", flush=True)
    finally:
        for e in envs.values():
            try:
                e.close()
            except Exception:
                pass

    # paired analysis vs immediate
    base = {r["episode_index"]: r for r in records if r["mode"] == "immediate"}
    verdicts = {}
    for a in arms:
        if a == "immediate":
            continue
        ds = [(r["total_carbon_kg"] - base[r["episode_index"]]["total_carbon_kg"])
              / max(1e-9, base[r["episode_index"]]["total_carbon_kg"])
              for r in records if r["mode"] == a and r["episode_index"] in base]
        neg = sum(1 for d in ds if d < 0)
        verdicts[a] = {"per_offset_rel_delta": ds, "improve_signs": f"{neg}/{len(ds)}",
                       "median_rel_delta": float(np.median(ds)) if ds else None}
        print(f"[H1 VERDICT {a}] median {verdicts[a]['median_rel_delta']:+.4f} "
              f"improves {neg}/{len(ds)}", flush=True)
    if args.json_out:
        pathlib.Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        pathlib.Path(args.json_out).write_text(
            json.dumps({"records": records, "verdicts": verdicts}, indent=1))


if __name__ == "__main__":
    main()
