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
    """
    space.seed(seed)
    rows = [space.sample() for _ in range(n)]
    return {k: np.stack([r[k] for r in rows]).astype(np.float32) for k in rows[0]}


def probs(module, batch):
    import torch
    with torch.no_grad():
        out = module.forward_inference(
            {"obs": {k: torch.as_tensor(v) for k, v in batch.items()}})
    logits = out.get("action_dist_inputs")
    if logits is None:
        raise SystemExit(f"no action_dist_inputs in module output: {sorted(out)}")
    logits = logits.detach().cpu().numpy().astype(np.float64)
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

    space = getattr(mods["godeye"], "observation_space", None)
    if space is None:
        raise SystemExit("the module does not carry its observation space")
    batch = sample_batch(space, a.n, a.seed)
    missing = [k for k in FORECAST_KEYS if k not in batch]
    if missing:
        raise SystemExit(f"forecast keys absent from the observation: {missing}")

    # Perturb the forecast channel and nothing else, clipped to each key's own bounds.
    pert = {k: v.copy() for k, v in batch.items()}
    rng = np.random.default_rng(a.seed + 1)
    for k in FORECAST_KEYS:
        lo, hi = float(space[k].low.min()), float(space[k].high.max())
        pert[k] = np.clip(batch[k] + rng.normal(0.0, a.eps, batch[k].shape), lo, hi
                          ).astype(np.float32)

    res = {"n": a.n, "seed": a.seed, "eps": a.eps, "forecast_keys": list(FORECAST_KEYS),
           "per_arm": {}}
    P = {}
    for arm, m in mods.items():
        p0, p1 = probs(m, batch), probs(m, pert)
        P[arm] = p0
        l1 = np.abs(p1 - p0).sum(axis=-1)
        res["per_arm"][arm] = {
            "forecast_sensitivity_l1_mean": float(l1.mean()),
            "forecast_sensitivity_l1_p90": float(np.percentile(l1, 90)),
            "argmax_flip_rate_under_perturbation": float(
                (p0.argmax(-1) != p1.argmax(-1)).mean()),
            "action_entropy_mean": float((-p0 * np.log(p0 + 1e-12)).sum(-1).mean()),
        }

    a0, a1 = P["godeye"].argmax(-1), P["shrink50"].argmax(-1)
    res["between_arms"] = {
        "argmax_divergence": float((a0 != a1).mean()),
        "mean_l1_between_policies": float(np.abs(P["godeye"] - P["shrink50"]).sum(-1).mean()),
        "sensitivity_ratio_shrink50_over_godeye": (
            res["per_arm"]["shrink50"]["forecast_sensitivity_l1_mean"]
            / max(res["per_arm"]["godeye"]["forecast_sensitivity_l1_mean"], 1e-12)),
    }
    json.dump(res, open(a.out, "w"), indent=2, sort_keys=True)
    print(json.dumps(res, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
