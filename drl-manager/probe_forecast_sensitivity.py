#!/usr/bin/env python
"""Does the trained global router's action distribution depend on the forecast?

Background. On v3 the oracle arm (forecast in the observation) and the blind arm
place jobs identically -- green-DC share 92.1 / 90.5 / 91.5 %, per-DC counts
inside noise -- and realise the same carbon. Either the forecast never reaches
the action, or it reaches it and changes nothing. This probe separates the two
without the simulator: load the trained global RLModule, hold every other input
fixed, and rewrite only the forecast channel.

Design. Each trial builds one observation in which exactly one green DC is the
"good" one, then asks the policy for its routing distribution over the batch
slots. The quantity of interest is how much that distribution moves when the
identity of the good DC is moved from DC a to DC b.

Three channels are swept the same way so the forecast number has a scale:

  forecast  the four dc_future_* features -- what the oracle has and the blind
            arm does not; the channel under test
  control   dc_current_green_power_w -- a channel the policy demonstrably uses
            (both arms route 91 % of jobs to green DCs), so it calibrates what
            "the policy responds to this" looks like
  null      dc_cumulative_wasted_green_wh -- a channel with no routing meaning,
            giving the noise floor of the measurement itself

Reading the result. forecast ~= null means the forecast channel is inert: the
information is in the observation but never reaches the action, which points at
representation or credit assignment rather than the scenario or the reward.
forecast ~= control means the policy does read the forecast, and the failure is
in what it does with it. Anything between is a partial-use result.

Runs on CPU in minutes and touches no Java gateway, so it is safe to run beside
an evaluation.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

REPO = pathlib.Path(__file__).resolve().parent
N_DC = 8
GREEN_DCS = [0, 1, 2, 5]          # green_energy_enabled in experiment_v3_*
BATCH_SLOTS = 128                 # global_routing_batch_size
MI_HIGH = 50_000_000              # obs_cloudlet_mi_high for v3

FORECAST_KEYS = ("dc_future_short_mean", "dc_future_short_trend",
                 "dc_future_long_mean", "dc_future_long_peak_timing")


def checkpoint_env_config(checkpoint: pathlib.Path) -> dict:
    """Read the nearest Tune trial env_config without loading Ray."""
    for parent in (checkpoint, *checkpoint.parents):
        params = parent / "params.json"
        if not params.is_file():
            continue
        try:
            payload = json.loads(params.read_text())
            env_cfg = payload.get("env_config", payload.get("config", {}).get("env_config", {}))
            return env_cfg if isinstance(env_cfg, dict) else {}
        except (OSError, ValueError, TypeError):
            continue
    return {}


def checkpoint_forecast_baseline(checkpoint: pathlib.Path) -> str:
    """Infer the arm's pre-registered forecast baseline from checkpoint config."""
    mode = str(checkpoint_env_config(checkpoint).get(
        "forecast_mode", "full")).strip().lower()
    return "persistence" if mode == "none" else "forecast"


def apply_forecast_baseline(
    obs: dict, baseline_type: str, *, green_power_high: float = 3000.0
) -> dict:
    """Return an obs whose forecast block matches the arm's own null semantics.

    V3.2 blind checkpoints are trained on persistence, not an all-zero sentinel.
    A response when we deliberately move them away from persistence is therefore
    a sensitivity measurement, not evidence that future information leaked.
    """
    o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    if baseline_type == "forecast":
        return o
    if baseline_type != "persistence":
        raise ValueError(f"unknown forecast baseline {baseline_type!r}")
    current_norm = np.clip(
        np.asarray(o["dc_current_green_power_w"], dtype=np.float32)
        / max(1e-9, float(green_power_high)),
        0.0,
        1.0,
    )
    o["dc_future_short_mean"] = current_norm.copy()
    o["dc_future_short_trend"] = np.zeros(N_DC, dtype=np.float32)
    o["dc_future_long_mean"] = current_norm.copy()
    o["dc_future_long_peak_timing"] = np.full(N_DC, 0.5, dtype=np.float32)
    if "batch_cloudlet_forecast_gain" in o:
        o["batch_cloudlet_forecast_gain"] = np.zeros(BATCH_SLOTS, dtype=np.float32)
        o["batch_cloudlet_time_to_best_green"] = np.ones(BATCH_SLOTS, dtype=np.float32)
        o["batch_cloudlet_best_future_carbon"] = np.asarray(
            o["batch_cloudlet_best_now_carbon"], dtype=np.float32).copy()
    return o


def base_observation(rng: np.random.Generator) -> dict:
    """One plausible mid-episode global observation.

    Values follow the v3 testbed: green power in the tens-to-hundreds of Watts
    (the compressed-mode scale that obs_green_power_high=3000 was set for),
    queues partly filled, jobs of 6e6-40e6 MI.
    """
    return {
        "dc_current_green_power_w": rng.uniform(0, 300, N_DC).astype(np.float32),
        "dc_current_power_w": rng.uniform(100, 1200, N_DC).astype(np.float32),
        "dc_green_ratio": rng.uniform(0, 1, N_DC).astype(np.float32),
        "dc_cumulative_wasted_green_wh": rng.uniform(0, 500, N_DC).astype(np.float32),
        "dc_future_short_mean": rng.uniform(0, 1, N_DC).astype(np.float32),
        "dc_future_short_trend": rng.uniform(-1, 1, N_DC).astype(np.float32),
        "dc_future_long_mean": rng.uniform(0, 1, N_DC).astype(np.float32),
        "dc_future_long_peak_timing": rng.uniform(0, 1, N_DC).astype(np.float32),
        "dc_queue_sizes": rng.integers(0, 60, N_DC).astype(np.int32),
        "dc_utilizations": rng.uniform(0, 1, N_DC).astype(np.float32),
        "dc_available_pes": rng.integers(0, 200, N_DC).astype(np.int32),
        "dc_ram_utilizations": rng.uniform(0, 1, N_DC).astype(np.float32),
        "upcoming_cloudlets_count": np.array([rng.integers(0, 400)], dtype=np.int32),
        "batch_cloudlet_pes": np.ones(BATCH_SLOTS, dtype=np.int32),
        "batch_cloudlet_mi": rng.integers(6_000_000, 40_000_000, BATCH_SLOTS).astype(np.int64),
        "upcoming_pes_distribution": rng.integers(0, 100, 3).astype(np.int32),
        "load_imbalance": np.array([rng.uniform(0, 3)], dtype=np.float32),
        "recent_completed": np.array([rng.integers(0, 500)], dtype=np.int32),
    }


def maybe_add_v31_features(obs: dict, module, rng: np.random.Generator) -> dict:
    """Add the V3.1 defer-state observation keys when (and only when) the
    loaded module was trained with obs_v31_features=true.

    Detection is by the module's own observation space, so old checkpoints keep
    the exact observation dict they were probed with before (regression-safe).
    Values are plausible mid-episode states in the normalized/clipped units the
    env declares (wait_age [0,1], time_to_deadline [-1,4], flags/counts [0,1]).
    They are HELD CONSTANT across the two observations a trial compares, so the
    channel sweeps remain single-variable.
    """
    # Key detection must UNION every source: the declared observation_space on
    # this module class is the wrapped Dict ['action_mask','observation'] (not
    # the flat feature keys - that non-empty decoy defeated an if-empty
    # fallback on 08-14), while the flat feature names live in the module's
    # own key lists.
    keys = set()
    space = getattr(module, "observation_space", None) or getattr(
        getattr(module, "config", None), "observation_space", None)
    if hasattr(space, "spaces"):
        keys |= set(space.spaces.keys())
        inner = space.spaces.get("observation")
        if hasattr(inner, "spaces"):
            keys |= set(inner.spaces.keys())
    for attr in ("cloudlet_keys", "per_dc_keys", "context_keys"):
        keys |= set(getattr(module, attr, []) or [])
    if "batch_cloudlet_wait_age" not in keys and "batch_cloudlet_deadline_present" not in keys:
        return obs
    obs["batch_cloudlet_wait_age"] = rng.uniform(0.0, 0.3, BATCH_SLOTS).astype(np.float32)
    obs["batch_cloudlet_time_to_deadline"] = rng.uniform(0.2, 1.5, BATCH_SLOTS).astype(np.float32)
    obs["batch_cloudlet_deadline_present"] = np.ones(BATCH_SLOTS, dtype=np.float32)
    obs["batch_cloudlet_is_deferred"] = (rng.uniform(0, 1, BATCH_SLOTS) < 0.1).astype(np.float32)
    obs["batch_cloudlet_defer_count"] = (obs["batch_cloudlet_is_deferred"]
                                         * rng.uniform(0.0, 0.2, BATCH_SLOTS)).astype(np.float32)
    obs["global_deferred_count"] = np.array([0.05], dtype=np.float32)
    obs["global_deferred_mi"] = np.array([0.05], dtype=np.float32)
    return obs


def maybe_add_v32_features(obs: dict, module, rng: np.random.Generator) -> dict:
    """Match a V3.2 module's schema without changing single-channel sweeps."""
    keys = set(getattr(module, "cloudlet_keys", []) or [])
    if "batch_cloudlet_forecast_gain" not in keys:
        return obs
    best_now = rng.uniform(0.05, 0.8, BATCH_SLOTS).astype(np.float32)
    gain = rng.uniform(0.0, 0.25, BATCH_SLOTS).astype(np.float32)
    obs["batch_cloudlet_forecast_gain"] = gain
    obs["batch_cloudlet_time_to_best_green"] = rng.uniform(
        0.05, 1.0, BATCH_SLOTS).astype(np.float32)
    obs["batch_cloudlet_best_now_carbon"] = best_now
    obs["batch_cloudlet_best_future_carbon"] = np.maximum(
        0.0, best_now - gain).astype(np.float32)
    return obs


def prepared_observation(
    rng: np.random.Generator,
    module,
    baseline_type: str,
    *,
    green_power_high: float = 3000.0,
) -> dict:
    obs = maybe_add_v31_features(base_observation(rng), module, rng)
    obs = maybe_add_v32_features(obs, module, rng)
    return apply_forecast_baseline(
        obs, baseline_type, green_power_high=green_power_high)


def set_channel(obs: dict, channel: str, good_dc: int) -> dict:
    """Rewrite one channel so that `good_dc` is unambiguously the best green DC.

    Only the named channel is touched; every other input is byte-identical
    between the two observations a trial compares, so any change in the action
    distribution is attributable to this channel alone.
    """
    o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    lo, hi = 0.05, 0.95
    if channel == "forecast":
        for dc in range(N_DC):
            good = dc == good_dc
            o["dc_future_short_mean"][dc] = hi if good else lo
            o["dc_future_long_mean"][dc] = hi if good else lo
            o["dc_future_short_trend"][dc] = 0.8 if good else -0.8
            o["dc_future_long_peak_timing"][dc] = 0.05 if good else 0.9
    elif channel == "control":
        for dc in range(N_DC):
            o["dc_current_green_power_w"][dc] = 2800.0 if dc == good_dc else 20.0
    elif channel == "null":
        for dc in range(N_DC):
            o["dc_cumulative_wasted_green_wh"][dc] = 5.0 if dc == good_dc else 900.0
    else:
        raise ValueError(channel)
    return o


def set_temporal(obs: dict, regime: str) -> dict:
    """Rewrite the forecast so it describes WHEN green arrives, not WHERE.

    v3's preflight forces the green DCs to be synchronised, so "which DC is
    greener" carries almost nothing and the spatial sweep above measures a lever
    that is empty by construction. The lever the scenario was built around is
    temporal: hold a job when green is about to arrive, run it when green is
    about to leave. Both regimes below keep the DCs identical to each other and
    keep the present (dc_current_green_power_w) fixed, so only the future moves.

      "arriving"  green is low now and rises within the window -> waiting pays
      "leaving"   green is high now and falls within the window -> waiting costs

    If the defer option's probability does not move between these, the policy
    reads the forecast spatially only, and the temporal lever the scenario was
    designed around is never actuated.
    """
    o = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in obs.items()}
    o["dc_current_green_power_w"][:] = 120.0        # identical present in both
    for dc in range(N_DC):
        green = dc in GREEN_DCS
        if not green:
            o["dc_future_short_mean"][dc] = 0.0
            o["dc_future_long_mean"][dc] = 0.0
            o["dc_future_short_trend"][dc] = 0.0
            o["dc_future_long_peak_timing"][dc] = 0.5
        elif regime == "arriving":
            o["dc_future_short_mean"][dc] = 0.15
            o["dc_future_long_mean"][dc] = 0.85
            o["dc_future_short_trend"][dc] = 0.9
            o["dc_future_long_peak_timing"][dc] = 0.1
        else:                                        # leaving
            o["dc_future_short_mean"][dc] = 0.85
            o["dc_future_long_mean"][dc] = 0.15
            o["dc_future_short_trend"][dc] = -0.9
            o["dc_future_long_peak_timing"][dc] = 0.9
    return o


def load_module(ckpt: pathlib.Path):
    from ray.rllib.core.rl_module.rl_module import RLModule
    path = ckpt / "learner_group" / "learner" / "rl_module" / "global_policy"
    if not path.exists():
        sys.exit(f"no global_policy module under {ckpt}")
    return RLModule.from_checkpoint(path)


def action_logits_raw(module, obs: dict) -> np.ndarray:
    """Raw per-slot action logits (BATCH_SLOTS, n_options), no softmax."""
    from ray.rllib.core.columns import Columns
    batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                           for k, v in obs.items()}}
    state = module.get_initial_state()
    if state:
        batch[Columns.STATE_IN] = {
            k: torch.as_tensor(np.asarray(v))[None, ...] for k, v in state.items()
        }
    with torch.no_grad():
        out = module.forward_inference(batch)
    logits = out[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy().reshape(-1)
    return logits.reshape(BATCH_SLOTS, logits.size // BATCH_SLOTS)


def action_probs(module, obs: dict) -> np.ndarray:
    """Per-slot categorical distribution, shape (BATCH_SLOTS, n_options).

    The global action is factored: one categorical per routing slot, so the
    logits come back flattened and are reshaped per slot.
    """
    from ray.rllib.core.columns import Columns

    batch = {Columns.OBS: {k: torch.as_tensor(np.asarray(v)[None, ...])
                           for k, v in obs.items()}}
    state = module.get_initial_state()
    if state:
        batch[Columns.STATE_IN] = {
            k: torch.as_tensor(np.asarray(v))[None, ...] for k, v in state.items()
        }
    with torch.no_grad():
        out = module.forward_inference(batch)
    logits = out[Columns.ACTION_DIST_INPUTS].detach().cpu().numpy().reshape(-1)
    n_opt = logits.size // BATCH_SLOTS
    if n_opt * BATCH_SLOTS != logits.size:
        sys.exit(f"logits size {logits.size} not divisible by {BATCH_SLOTS} slots")
    z = logits.reshape(BATCH_SLOTS, n_opt)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--trials", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--raw-logits", action="store_true",
                    help="also report max|delta| of RAW defer vs route logits "
                         "between the two temporal regimes (direct-edge check, "
                         "Gate 1: legacy modules must show defer ~invariant)")
    ap.add_argument(
        "--forecast-baseline",
        choices=("auto", "forecast", "persistence"),
        default="auto",
        help="null semantics for this arm; auto reads checkpoint params.json",
    )
    args = ap.parse_args()

    checkpoint = pathlib.Path(args.checkpoint)
    module = load_module(checkpoint)
    checkpoint_config = checkpoint_env_config(checkpoint)
    baseline_type = (
        checkpoint_forecast_baseline(checkpoint)
        if args.forecast_baseline == "auto"
        else args.forecast_baseline
    )
    green_power_high = float(checkpoint_config.get(
        "obs_green_power_high", 3000.0))
    rng = np.random.default_rng(args.seed)

    results = {c: {"tv": [], "flip": [], "mass": []} for c in ("forecast", "control", "null")}
    for _ in range(args.trials):
        obs = prepared_observation(
            rng, module, baseline_type, green_power_high=green_power_high)
        a, b = rng.choice(GREEN_DCS, size=2, replace=False)
        for channel in results:
            pa = action_probs(module, set_channel(obs, channel, int(a)))
            pb = action_probs(module, set_channel(obs, channel, int(b)))
            # total variation per slot, averaged over slots
            results[channel]["tv"].append(0.5 * np.abs(pa - pb).sum(axis=1).mean())
            results[channel]["flip"].append((pa.argmax(1) != pb.argmax(1)).mean())
            # does probability mass follow the DC the channel just marked as good?
            results[channel]["mass"].append(
                (pa[:, int(a)].mean() - pb[:, int(a)].mean())
            )

    print(f"\ncheckpoint : {args.checkpoint}")
    print(f"baseline   : {baseline_type}"
          + (" (blind null; perturbation means departure from persistence, not leakage)"
             if baseline_type == "persistence" else " (forecast arm)"))
    print(f"green high : {green_power_high:g} W (checkpoint observation contract)")
    print(f"trials     : {args.trials}  (each moves the 'good' DC between two green DCs)\n")
    print(f"{'channel':>10}{'TV distance':>14}{'argmax flips':>15}{'mass follows':>15}")
    print("-" * 54)
    summary = {}
    for c in ("control", "forecast", "null"):
        tv = float(np.mean(results[c]["tv"]))
        fl = float(np.mean(results[c]["flip"]))
        ms = float(np.mean(results[c]["mass"]))
        summary[c] = {"tv": tv, "flip": fl, "mass": ms}
        print(f"{c:>10}{tv:>14.4f}{fl*100:>14.1f}%{ms:>+15.4f}")
    print("-" * 54)
    ctrl, fc, nul = summary["control"]["tv"], summary["forecast"]["tv"], summary["null"]["tv"]
    denom = ctrl - nul
    frac = (fc - nul) / denom if abs(denom) > 1e-9 else float("nan")
    print(f"forecast sensitivity as a fraction of the control channel: {frac:.3f}")
    print("  ~0 -> the forecast channel is inert (information never reaches the action)")
    print("  ~1 -> the policy reads the forecast as strongly as current green")

    # --- temporal lever: does the forecast move the DEFER option? --------------
    rng = np.random.default_rng(args.seed + 1000)
    n_opt = action_probs(module, prepared_observation(
        rng, module, baseline_type,
        green_power_high=green_power_high,
    )).shape[1]
    defer_idx = n_opt - 1                     # defer is appended after the N DCs
    arriving, leaving, tvs = [], [], []
    for _ in range(args.trials):
        obs = prepared_observation(
            rng, module, baseline_type, green_power_high=green_power_high)
        pa = action_probs(module, set_temporal(obs, "arriving"))
        pl = action_probs(module, set_temporal(obs, "leaving"))
        arriving.append(pa[:, defer_idx].mean())
        leaving.append(pl[:, defer_idx].mean())
        tvs.append(0.5 * np.abs(pa - pl).sum(axis=1).mean())
    d_arr, d_lev, tv_t = float(np.mean(arriving)), float(np.mean(leaving)), float(np.mean(tvs))
    print(f"\ntemporal lever ({n_opt} options, defer = index {defer_idx})")
    print(f"{'P(defer) | green arriving':>34}{d_arr:>10.4f}")
    print(f"{'P(defer) | green leaving':>34}{d_lev:>10.4f}")
    print(f"{'difference (want >> 0)':>34}{d_arr - d_lev:>+10.4f}")
    print(f"{'TV of the whole distribution':>34}{tv_t:>10.4f}\n")

    raw = None
    if args.raw_logits:
        # Direct-edge check (docs/V32_FORECAST_REVIVAL_PLAN.md §6.1): only the
        # forecast keys differ between the two observations, so a defer column
        # that moves orders of magnitude less than the route columns proves the
        # temporal head has no direct forecast edge (and vice versa for V3.2).
        rng2 = np.random.default_rng(args.seed + 2000)
        obs0 = maybe_add_v31_features(base_observation(rng2), module, rng2)
        za = action_logits_raw(module, set_temporal(dict(obs0), "arriving"))
        zl = action_logits_raw(module, set_temporal(dict(obs0), "leaving"))
        d_defer = float(np.abs(za[:, -1] - zl[:, -1]).max())
        d_route = float(np.abs(za[:, :-1] - zl[:, :-1]).max())
        ratio = d_route / max(d_defer, 1e-12)
        raw = {"max_abs_delta_raw_defer_logit": d_defer,
               "max_abs_delta_raw_route_logits": d_route,
               "route_over_defer_ratio": ratio}
        print(f"\nraw-logit direct-edge check (temporal perturbation only):")
        print(f"{'max|d raw defer logit|':>34}{d_defer:>12.3e}")
        print(f"{'max|d raw route logits|':>34}{d_route:>12.3e}")
        print(f"{'route/defer response ratio':>34}{ratio:>12.1f}")

    if args.json_out:
        pathlib.Path(args.json_out).write_text(json.dumps(
            {"checkpoint": args.checkpoint, "trials": args.trials,
             "forecast_baseline": baseline_type,
             "forecast_green_power_high": green_power_high,
             "perturbation_interpretation": (
                 "departure_from_persistence_not_leakage"
                 if baseline_type == "persistence" else "forecast_channel_sensitivity"
             ),
             "summary": summary, "forecast_over_control": frac,
             "raw_logits": raw,
             "temporal": {"n_options": int(n_opt), "defer_index": int(defer_idx),
                          "p_defer_arriving": d_arr, "p_defer_leaving": d_lev,
                          "delta": d_arr - d_lev, "tv": tv_t}}, indent=2))


if __name__ == "__main__":
    main()
