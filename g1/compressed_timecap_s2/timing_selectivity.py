"""Stage D' timing-selectivity gate (STAGE_D_PRIME_DESIGN §2 Q4).

Corpus: the truth-informed planner ST replayed on the frozen corpus windows with
EVAL_DECISION_DUMP (per-slot decisions) and EVAL_DECISION_DUMP_OBS=1 (the global
observation of every step). Labels: ST-defer (1) vs ST-route (0) per (step, slot).
Score: the trained V line's probability of DEFER on the same observations, per slot.

    lift = mean P_V(defer | ST-defer) - mean P_V(defer | ST-route)     >= 0.10
    balanced AUC (rank statistic, each class weighted equally)          >= 0.60

The module is scored per observation with a fresh recurrent state (no memory across
steps), which is the stated caveat of this diagnostic. Pure metric in `lift_and_auc`.

Usage: python timing_selectivity.py <decisions.csv> <obs.npz> <checkpoint_dir> [--out json]
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LIFT_MIN, AUC_MIN = 0.10, 0.60


def lift_and_auc(p, labels):
    """p: DEFER probabilities; labels: 1 = ST deferred, 0 = ST routed. Pure."""
    p = np.asarray(p, float); y = np.asarray(labels, int)
    pos, neg = p[y == 1], p[y == 0]
    out = {"n_pos": int(pos.size), "n_neg": int(neg.size)}
    if pos.size == 0 or neg.size == 0:
        out.update({"lift": None, "auc": None, "pass": False, "reason": "one class empty"})
        return out
    lift = float(pos.mean() - neg.mean())
    # balanced AUC = P(score_pos > score_neg) + 0.5 P(tie), equal class weighting by construction
    order = np.argsort(np.concatenate([pos, neg]), kind="mergesort")
    ranks = np.empty(order.size, float)
    allv = np.concatenate([pos, neg])[order]
    i = 0
    while i < order.size:                       # average ranks for ties
        j = i
        while j + 1 < order.size and allv[j + 1] == allv[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    auc = float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2.0) / (pos.size * neg.size))
    out.update({"lift": lift, "auc": auc,
                "pass": lift >= LIFT_MIN and auc >= AUC_MIN,
                "mean_p_defer_given_st_defer": float(pos.mean()),
                "mean_p_defer_given_st_route": float(neg.mean())})
    return out


def score_checkpoint(decisions_csv, obs_npz, checkpoint):
    """P_V(defer) per (step, slot) of the corpus, from the checkpoint's global module."""
    import torch
    import rl_step2_probe as rp
    m, _key = rp.load_module(checkpoint)
    z = np.load(obs_npz)
    keys = list(z.keys())
    n_steps = int(z[keys[0]].shape[0])
    nvec = m.action_space.nvec
    n_slots, n_choices = len(nvec), int(nvec[0])
    defer_idx = n_choices - 1
    rows = list(csv.DictReader(open(decisions_csv)))
    by_step = {}
    for r in rows:
        by_step.setdefault(int(r["step"]), []).append((int(r["slot"]), int(r["is_defer"])))
    probs, labels, meta = [], [], []
    with torch.no_grad():
        for t in range(n_steps):
            if t not in by_step:
                continue
            obs = {k: torch.as_tensor(np.asarray(z[k][t])[None, ...]) for k in keys}
            batch = {"obs": {"observation": obs, "action_mask": torch.ones(1, n_slots)}}
            try:
                out = m.forward_inference(batch)
            except Exception:
                out = m.forward_exploration(batch)
            logits = out["action_dist_inputs"]
            logits = logits.reshape(-1, n_slots, n_choices)[-1]        # (slots, choices)
            p_defer = torch.softmax(logits, dim=-1)[:, defer_idx].cpu().numpy()
            for slot, lab in by_step[t]:
                if slot < n_slots:
                    probs.append(float(p_defer[slot])); labels.append(lab); meta.append((t, slot))
    return np.asarray(probs), np.asarray(labels), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("decisions_csv", help="one decisions CSV, or a corpus DIRECTORY holding *_decisions.csv + *_obs.npz pairs")
    ap.add_argument("obs_npz", nargs="?", default=None); ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import glob
    pairs = []
    if os.path.isdir(a.decisions_csv):
        for dec in sorted(glob.glob(os.path.join(a.decisions_csv, "*_decisions.csv"))):
            obs = dec.replace("_decisions.csv", "_decisions_obs.npz")
            if os.path.exists(obs):
                pairs.append((dec, obs))
    else:
        pairs.append((a.decisions_csv, a.obs_npz))
    ps, ys, per_window = [], [], {}
    for dec, obs in pairs:
        p, y, _meta = score_checkpoint(dec, obs, os.path.abspath(a.checkpoint))
        ps.append(p); ys.append(y)
        per_window[os.path.basename(dec)] = lift_and_auc(p, y)
    p = np.concatenate(ps) if ps else np.array([]); y = np.concatenate(ys) if ys else np.array([])
    res = lift_and_auc(p, y)
    res.update({"checkpoint": a.checkpoint, "corpus": a.decisions_csv, "n_scored": int(p.size),
                "overall_p_defer": float(p.mean()) if p.size else None, "per_window": per_window})
    print(json.dumps(res, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
