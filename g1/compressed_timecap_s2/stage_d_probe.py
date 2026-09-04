"""Stage D forecast-sensitivity probe for one trained line (reuses rl_step2_probe's math).

For a checkpoint's global RLModule: the per-slot L1 shift of action probabilities when only
the four forecast keys are perturbed (forecast sensitivity), and when only the present-tense
control keys are perturbed by the same magnitude (control sensitivity), on a fixed synthetic
batch. Output: results/stage_d/probe_<line>.json in the F2 probe's per_arm layout, which
stage_d_health_verdict.py reads.

Usage: python stage_d_probe.py <line> <checkpoint_dir> [--n 512] [--eps 0.25]
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import rl_step2_probe as rp  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("line")
    ap.add_argument("checkpoint")
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--seed", type=int, default=20260903)
    ap.add_argument("--eps", type=float, default=0.25)
    ap.add_argument("--out-dir", default=os.path.join(rp.REPO, "results", "stage_d"))
    a = ap.parse_args()

    m, key = rp.load_module(a.checkpoint)
    nvec = m.action_space.nvec
    space = getattr(m, "observation_space", None)
    if space is None:
        raise SystemExit("the module does not carry its observation space")
    batch = rp.sample_batch(space, a.n, a.seed)
    fspace = space["observation"] if "observation" in space.spaces else space
    src = rp.get_forecast(batch)
    missing = [k for k in rp.FORECAST_KEYS if k not in src]
    if missing:
        raise SystemExit(f"forecast keys absent from the observation: {missing}")

    pert = copy.deepcopy(batch)
    rng = np.random.default_rng(a.seed + 1)
    dst = rp.get_forecast(pert)
    for k in rp.FORECAST_KEYS:
        lo, hi = float(fspace[k].low.min()), float(fspace[k].high.max())
        dst[k] = np.clip(src[k] + rng.normal(0.0, a.eps, src[k].shape), lo, hi).astype(np.float32)
    ctrl = copy.deepcopy(batch)
    cdst = rp.get_forecast(ctrl)
    rng2 = np.random.default_rng(a.seed + 2)
    ctrl_keys = [k for k in rp.CONTROL_KEYS if k in src]
    for k in ctrl_keys:
        lo, hi = float(fspace[k].low.min()), float(fspace[k].high.max())
        cdst[k] = np.clip(src[k] + rng2.normal(0.0, a.eps, src[k].shape), lo, hi).astype(np.float32)

    p0, p1, p2 = (rp.probs(m, b, nvec) for b in (batch, pert, ctrl))
    l1 = np.abs(p1 - p0).sum(axis=-1).mean(axis=-1)
    l1c = np.abs(p2 - p0).sum(axis=-1).mean(axis=-1)
    n_act = p0.shape[-1]
    per_arm = {
        "forecast_sensitivity_l1_mean": float(l1.mean()),
        "forecast_sensitivity_l1_p90": float(np.percentile(l1, 90)),
        "control_sensitivity_l1_mean": float(l1c.mean()),
        "forecast_over_control_ratio": float(l1.mean() / max(l1c.mean(), 1e-12)),
        "argmax_flip_rate_forecast": float((p0.argmax(-1) != p1.argmax(-1)).mean()),
        "argmax_flip_rate_control": float((p0.argmax(-1) != p2.argmax(-1)).mean()),
        "per_slot_entropy_mean": float((-p0 * np.log(p0 + 1e-12)).sum(-1).mean()),
        "entropy_fraction_of_uniform": float((-p0 * np.log(p0 + 1e-12)).sum(-1).mean() / np.log(n_act)),
        "top1_prob_mean": float(p0.max(-1).mean()),
        "defer_prob_mean": float(p0[..., -1].mean()),
    }
    res = {"line": a.line, "checkpoint": os.path.relpath(a.checkpoint, rp.REPO), "module": key,
           "n": a.n, "seed": a.seed, "eps": a.eps, "forecast_keys": list(rp.FORECAST_KEYS),
           "control_keys": ctrl_keys, "action_space": f"MultiDiscrete([{n_act}] * {p0.shape[1]})",
           "per_arm": {a.line: per_arm}}
    os.makedirs(a.out_dir, exist_ok=True)
    out = os.path.join(a.out_dir, f"probe_{a.line}.json")
    json.dump(res, open(out, "w"), indent=2, sort_keys=True)
    print(json.dumps(res["per_arm"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
