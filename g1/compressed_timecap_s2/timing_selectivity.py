"""Stage D' timing-selectivity gate (STAGE_D_PRIME_DESIGN §2 Q4, tightened by §16 Q2).

Corpus: the truth-informed planner ST replayed on the frozen development windows with
EVAL_DECISION_DUMP (per-slot decisions incl. cloudlet_id and defer_allowed) and
EVAL_DECISION_DUMP_OBS=1 (the global observation of every step).

Main gate = job-paired, recurrent, PRE-mask:
  * per job at most one deterministic ST-defer sample (its first sighting where DEFER was
    legal) and one ST-route sample (its routing sighting, only if DEFER was legal there,
    i.e. the route was not forced by the deadline mask); jobs lacking either are dropped;
    metrics are job-equal-weighted (one pair per job);
  * the V module is run over each window IN TIME ORDER carrying its GTrXL memory;
  * two probabilities per scored slot: RAW = the module's DEFER preference with the mask
    key removed from the observation (learning-selectivity gate), DEPLOYED = with the key
    (safety diagnostic). Passing only after the mask means the safety layer works, not that
    the policy learned timing.
Appendix = the full decision-point corpus (every sighting, ~41:1), same recurrent pass.

    lift = mean P(defer | ST-defer) - mean P(defer | ST-route)   >= 0.10
    balanced AUC                                                >= 0.60

Usage: python timing_selectivity.py <corpus_dir> <checkpoint_dir> [--out json]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
LIFT_MIN, AUC_MIN = 0.10, 0.60
MASK_KEY = "batch_cloudlet_defer_allowed"


def lift_and_auc(p, labels):
    """p: DEFER probabilities; labels: 1 = ST deferred, 0 = ST routed. Pure."""
    p = np.asarray(p, float); y = np.asarray(labels, int)
    pos, neg = p[y == 1], p[y == 0]
    out = {"n_pos": int(pos.size), "n_neg": int(neg.size)}
    if pos.size == 0 or neg.size == 0:
        out.update({"lift": None, "auc": None, "pass": False, "reason": "one class empty"})
        return out
    lift = float(pos.mean() - neg.mean())
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
    out.update({"lift": lift, "auc": auc, "pass": lift >= LIFT_MIN and auc >= AUC_MIN,
                "mean_p_defer_given_st_defer": float(pos.mean()),
                "mean_p_defer_given_st_route": float(neg.mean())})
    return out


def pair_corpus(rows):
    """Pure. rows: dicts with step, slot, cloudlet_id, is_defer, defer_allowed (str/num).
    Returns {cloudlet_id: {"defer": (step, slot), "route": (step, slot)}} for jobs that have
    both a legal DEFER sample and a not-mask-forced ROUTE sample; plus counts."""
    by_job = {}
    for r in rows:
        cid = int(float(r.get("cloudlet_id", -1) or -1))
        if cid < 0:
            continue
        step, slot = int(float(r["step"])), int(float(r["slot"]))
        is_defer = int(float(r.get("is_defer", 0) or 0))
        da = r.get("defer_allowed")
        legal = True if da in (None, "", "None") else float(da) >= 0.5
        j = by_job.setdefault(cid, {"defer": None, "route": None, "route_forced": 0})
        if is_defer and legal and j["defer"] is None:
            j["defer"] = (step, slot)                         # first legal DEFER sighting
        elif not is_defer:
            if legal:
                if j["route"] is None or (step, slot) < j["route"]:
                    j["route"] = (step, slot)
            else:
                j["route_forced"] += 1                        # route forced by the mask: excluded
    pairs = {c: {"defer": j["defer"], "route": j["route"]} for c, j in by_job.items()
             if j["defer"] is not None and j["route"] is not None}
    return {"pairs": pairs, "n_jobs": len(by_job), "n_paired": len(pairs),
            "n_route_forced_excluded": sum(1 for j in by_job.values() if j["route"] is None and j["route_forced"] > 0),
            "n_never_deferred": sum(1 for j in by_job.values() if j["defer"] is None)}


def _to_state_in(state):
    import torch
    if isinstance(state, dict):
        return {k: (torch.as_tensor(v) if not hasattr(v, "shape") or not isinstance(v, torch.Tensor) else v)
                for k, v in state.items()}
    return {"gtrxl_mem": torch.as_tensor(state)}


def _next_state(out, prev):
    import torch
    so = out.get("state_out")
    if so is None:
        return prev
    if isinstance(so, dict):
        return {k: (v[0] if isinstance(v, torch.Tensor) and v.dim() == 4 else v) for k, v in so.items()}
    return {"gtrxl_mem": so[0] if isinstance(so, torch.Tensor) and so.dim() == 4 else so}


def score_window(decisions_csv, obs_npz, module, nvec):
    """Recurrent pass over one window. Returns per-(step, slot) RAW and DEPLOYED DEFER
    probabilities for every sighting, and the rows."""
    import torch
    rows = list(csv.DictReader(open(decisions_csv)))
    z = np.load(obs_npz)
    keys = list(z.keys())
    n_steps = int(z[keys[0]].shape[0])
    n_slots, n_choices = len(nvec), int(nvec[0])
    defer_idx = n_choices - 1
    wanted = {}
    for r in rows:
        wanted.setdefault(int(float(r["step"])), []).append(int(float(r["slot"])))
    raw_p, dep_p = {}, {}
    state = module.get_initial_state()
    with torch.no_grad():
        for t in range(n_steps):
            obs_dep = {k: torch.as_tensor(np.asarray(z[k][t])[None, ...]) for k in keys}
            si = _to_state_in(state)
            b_dep = {"obs": {"observation": obs_dep, "action_mask": torch.ones(1, n_slots)}, "state_in": si}
            module._audit_skip_defer_mask = False
            out_dep = module.forward_inference(b_dep)
            if t in wanted:
                # RAW = same observation (the trunk still sees defer_allowed as a feature),
                # same memory, only the -1e9 on the DEFER column skipped
                module._audit_skip_defer_mask = True
                out_raw = module.forward_inference(b_dep)
                module._audit_skip_defer_mask = False
                ld = out_dep["action_dist_inputs"].reshape(-1, n_slots, n_choices)[-1]
                lr = out_raw["action_dist_inputs"].reshape(-1, n_slots, n_choices)[-1]
                pd_ = torch.softmax(ld, dim=-1)[:, defer_idx].cpu().numpy()
                pr_ = torch.softmax(lr, dim=-1)[:, defer_idx].cpu().numpy()
                for s in wanted[t]:
                    if s < n_slots:
                        raw_p[(t, s)] = float(pr_[s]); dep_p[(t, s)] = float(pd_[s])
            state = _next_state(out_dep, state)          # the deployed trajectory's memory
    return raw_p, dep_p, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus_dir"); ap.add_argument("checkpoint")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    import rl_step2_probe as rp
    module, _key = rp.load_module(os.path.abspath(a.checkpoint))
    nvec = module.action_space.nvec
    P = {"raw": [], "dep": []}; Y = []
    A = {"raw": [], "dep": []}; AY = []
    per_window, totals = {}, {"n_jobs": 0, "n_paired": 0, "n_route_forced_excluded": 0, "n_never_deferred": 0}
    for dec in sorted(glob.glob(os.path.join(a.corpus_dir, "*_decisions.csv"))):
        obs = dec.replace("_decisions.csv", "_decisions_obs.npz")
        if not os.path.exists(obs):
            continue
        raw_p, dep_p, rows = score_window(dec, obs, module, nvec)
        pc = pair_corpus(rows)
        for k in totals:
            totals[k] += pc[k]
        wp, wy = {"raw": [], "dep": []}, []
        for cid, pr in pc["pairs"].items():
            for lab, key in ((1, pr["defer"]), (0, pr["route"])):
                if key in raw_p:
                    wp["raw"].append(raw_p[key]); wp["dep"].append(dep_p[key]); wy.append(lab)
        for r in rows:                                          # appendix: every sighting
            key = (int(float(r["step"])), int(float(r["slot"])))
            if key in raw_p:
                A["raw"].append(raw_p[key]); A["dep"].append(dep_p[key]); AY.append(int(float(r["is_defer"])))
        per_window[os.path.basename(dec)] = {"paired_raw": lift_and_auc(wp["raw"], wy),
                                             "paired_deployed": lift_and_auc(wp["dep"], wy),
                                             "n_paired": pc["n_paired"]}
        P["raw"] += wp["raw"]; P["dep"] += wp["dep"]; Y += wy
    res = {"checkpoint": a.checkpoint, "corpus": a.corpus_dir, "pairing": totals,
           "main_gate_raw_paired": lift_and_auc(P["raw"], Y),
           "diagnostic_deployed_paired": lift_and_auc(P["dep"], Y),
           "appendix_all_sightings_raw": lift_and_auc(A["raw"], AY),
           "appendix_all_sightings_deployed": lift_and_auc(A["dep"], AY),
           "per_window": per_window}
    res["pass"] = bool(res["main_gate_raw_paired"].get("pass"))
    print(json.dumps({k: res[k] for k in ("pairing", "main_gate_raw_paired", "diagnostic_deployed_paired", "pass")}, indent=1))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    main()
