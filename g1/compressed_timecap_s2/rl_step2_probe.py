#!/usr/bin/env python3
"""Two post-hoc probes on the step-2 pilot's trained policies. DESIGN_PILOT, read-only.

The learning curves alone cannot separate the three things that all look like "the two arms
came out the same":

    (a) the contaminated policy was not fooled -- the lie did not matter to what it learned
    (b) it was fooled and then learned to ignore the channel, netting out
    (c) neither arm learned to depend on the forecast yet, so corrupting it changes nothing

At twelve PPO updates (c) is the prior, and it is not a statement about mechanism. These two
probes separate them without further training:

  action divergence   feed both trained policies the SAME observations and compare argmax.
                      Near-zero divergence means the corrupted channel never reached the
                      decision; a real divergence means it did.

  forecast sensitivity  perturb ONLY the four forecast keys and measure how far each
                      policy's action distribution moves. Compared BETWEEN arms this is the
                      discriminator: a contaminated policy markedly less sensitive than the
                      clean one is evidence of escape (b); both similar and both non-trivial
                      is (c); both near zero is (a) with the channel unused.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(REPO, "drl-manager"))

FORECAST_KEYS = ("dc_future_short_mean", "dc_future_short_trend",
                 "dc_future_long_mean", "dc_future_long_peak_timing")
# Control channel: present-tense per-DC signals of the same shape and comparable range.
# Without it, "the forecast sensitivity is 0.035" is a number with no scale. With it, the
# question becomes the answerable one: does the policy react to the FUTURE as much as it
# reacts to the PRESENT, when both are nudged by the same amount?
CONTROL_KEYS = ("dc_green_ratio", "dc_utilizations")
ARMS = ("godeye", "shrink50")


def latest_checkpoint(arm):
    pat = os.path.join(REPO, "logs",
                       f"rlp2_s2_r48_w72_c3_n35_{arm}_GTrXL", "*", "multidc_gtrxl_training",
                       "PPO_multidc_env_*", "checkpoint_*")
    cks = sorted(glob.glob(pat))
    if not cks:
        raise SystemExit(f"no checkpoint for {arm}")
    return cks[-1]


def load_module(ckpt):
    """The global policy's RLModule, on CPU, in eval mode."""
    from ray.rllib.core.rl_module.rl_module import RLModule
    p = os.path.join(ckpt, "learner_group", "learner", "rl_module")
    if not os.path.isdir(p):
        p = os.path.join(ckpt, "learner", "rl_module")
    mods = RLModule.from_checkpoint(p)
    key = "global_policy" if "global_policy" in mods else sorted(mods)[0]
    m = mods[key]
    m.eval()
    return m, key


def sample_batch(space, n, seed):
    """A fixed batch of observations from the declared space.

    Synthetic rather than replayed: the probe asks what the POLICY does with a forecast, so
    the observations only have to be legal and identical across arms. Replaying real states
    would tie the answer to one arm's own trajectory, which is exactly the confound.

    The space is nested -- {action_mask, observation:{...}} -- so the stack is recursive.
    The mask matters: with an all-zero mask every action is illegal and the comparison
    would be between two degenerate distributions, so masks are forced to all-valid.
    """
    space.seed(seed)
    rows = [space.sample() for _ in range(n)]

    def stack(key_rows):
        if isinstance(key_rows[0], dict):
            return {k: stack([r[k] for r in key_rows]) for k in key_rows[0]}
        return np.stack(key_rows).astype(np.float32)

    batch = stack(rows)
    if "action_mask" in batch:
        batch["action_mask"] = np.ones_like(batch["action_mask"])
    return batch


def get_forecast(batch):
    return batch["observation"] if "observation" in batch else batch


def to_torch(x):
    import torch
    if isinstance(x, dict):
        return {k: to_torch(v) for k, v in x.items()}
    return torch.as_tensor(x)


def probs(module, batch, nvec):
    """Per-slot action probabilities, shape (B, n_slots, n_choices).

    The action space is MultiDiscrete([6] * 128): 128 independent 6-way choices, one per
    routing slot, NOT one 768-way choice. Softmaxing the flat 768-vector -- which an earlier
    version of this probe did -- mixes 128 separate distributions into one and reports an
    entropy near log(768) for a policy that may be perfectly decisive within each slot. Every
    statistic here is therefore computed per slot and then averaged over slots.
    """
    import torch
    with torch.no_grad():
        out = module.forward_inference({"obs": to_torch(batch)})
    logits = out.get("action_dist_inputs")
    if logits is None:
        raise SystemExit(f"no action_dist_inputs in module output: {sorted(out)}")
    logits = logits.detach().cpu().numpy().astype(np.float64)
    n_slots, n_choice = len(nvec), int(nvec[0])
    logits = logits.reshape(logits.shape[0], n_slots, n_choice)
    e = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--eps", type=float, default=0.25,
                    help="size of the forecast-only perturbation, in feature units")
    ap.add_argument("--out", default=os.path.join(HERE, "rl_step2_probe.json"))
    a = ap.parse_args()

    import gymnasium as gym  # noqa: F401
    mods = {}
    for arm in ARMS:
        ck = latest_checkpoint(arm)
        m, key = load_module(ck)
        mods[arm] = m
        print(f"{arm}: {os.path.relpath(ck, REPO)}  module={key}")

    nvec = mods["godeye"].action_space.nvec
    space = getattr(mods["godeye"], "observation_space", None)
    if space is None:
        raise SystemExit("the module does not carry its observation space")
    batch = sample_batch(space, a.n, a.seed)
    fspace = space["observation"] if "observation" in space.spaces else space
    missing = [k for k in FORECAST_KEYS if k not in get_forecast(batch)]
    if missing:
        raise SystemExit(f"forecast keys absent from the observation: {missing}")

    # Perturb the forecast channel and nothing else, clipped to each key's own bounds.
    import copy as _copy
    pert = _copy.deepcopy(batch)
    rng = np.random.default_rng(a.seed + 1)
    src, dst = get_forecast(batch), get_forecast(pert)
    for k in FORECAST_KEYS:
        lo, hi = float(fspace[k].low.min()), float(fspace[k].high.max())
        dst[k] = np.clip(src[k] + rng.normal(0.0, a.eps, src[k].shape), lo, hi
                         ).astype(np.float32)
    moved = float(np.mean([np.abs(dst[k] - src[k]).mean() for k in FORECAST_KEYS]))

    # Same-magnitude nudge on the present-tense control channel.
    ctrl = _copy.deepcopy(batch)
    cdst = get_forecast(ctrl)
    rng2 = np.random.default_rng(a.seed + 2)
    ctrl_keys = [k for k in CONTROL_KEYS if k in src]
    for k in ctrl_keys:
        lo, hi = float(fspace[k].low.min()), float(fspace[k].high.max())
        cdst[k] = np.clip(src[k] + rng2.normal(0.0, a.eps, src[k].shape), lo, hi
                          ).astype(np.float32)
    cmoved = float(np.mean([np.abs(cdst[k] - src[k]).mean() for k in ctrl_keys]))

    res = {"n": a.n, "seed": a.seed, "eps": a.eps, "forecast_keys": list(FORECAST_KEYS),
           "control_keys": ctrl_keys,
           "mean_abs_forecast_shift_applied": moved,
           "mean_abs_control_shift_applied": cmoved, "per_arm": {}}
    P = {}
    for arm, m in mods.items():
        p0, p1, p2 = (probs(m, b, nvec) for b in (batch, pert, ctrl))
        P[arm] = p0
        # L1 per slot, then averaged over slots and rows; max possible is 2.0 per slot.
        l1 = np.abs(p1 - p0).sum(axis=-1).mean(axis=-1)
        l1c = np.abs(p2 - p0).sum(axis=-1).mean(axis=-1)
        n_act = p0.shape[-1]
        res["action_space"] = f"MultiDiscrete([{n_act}] * {p0.shape[1]})"
        res["max_entropy_nats"] = float(np.log(n_act))
        res["per_arm"][arm] = {
            "forecast_sensitivity_l1_mean": float(l1.mean()),
            "forecast_sensitivity_l1_p90": float(np.percentile(l1, 90)),
            "control_sensitivity_l1_mean": float(l1c.mean()),
            "forecast_over_control_ratio": float(l1.mean() / max(l1c.mean(), 1e-12)),
            "argmax_flip_rate_forecast": float((p0.argmax(-1) != p1.argmax(-1)).mean()),
            "argmax_flip_rate_control": float((p0.argmax(-1) != p2.argmax(-1)).mean()),
            "per_slot_entropy_mean": float((-p0 * np.log(p0 + 1e-12)).sum(-1).mean()),
            "entropy_fraction_of_uniform": float(
                (-p0 * np.log(p0 + 1e-12)).sum(-1).mean() / np.log(n_act)),
            "top1_prob_mean": float(p0.max(-1).mean()),
            "defer_prob_mean": float(p0[..., -1].mean()),
        }

    a0, a1 = P["godeye"].argmax(-1), P["shrink50"].argmax(-1)
    res["between_arms"] = {
        "argmax_divergence": float((a0 != a1).mean()),
        "mean_l1_between_policies": float(
            np.abs(P["godeye"] - P["shrink50"]).sum(-1).mean()),
        "sensitivity_ratio_shrink50_over_godeye": (
            res["per_arm"]["shrink50"]["forecast_sensitivity_l1_mean"]
            / max(res["per_arm"]["godeye"]["forecast_sensitivity_l1_mean"], 1e-12)),
    }
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
