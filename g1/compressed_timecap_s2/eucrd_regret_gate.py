"""Instrument gates for the forecast responsibility signal defined as local decision regret
(reports/EUCRD_REGRET_SIGNAL_PREREG.md §3). Zero training, no carbon claim, no tuning.

  H1 truth in, zero out      : godeye gives max |signal| <= 1e-12 on every step.
  H2 alive under error       : shrink75 gives a mean signal > 0 on every window.
  H3 non-negative            : min signal >= 0 on every step of every run.
  H4 fixed-state pairing     : replaying the GODEYE arm's own schedule (same reservation grid,
                               same job batches) while swapping in each tier's forecast, every
                               degraded tier's pooled mean must exceed godeye's zero. The
                               ordering across degraded tiers is REPORTED, not gated.
  H5 no observation leak     : the policy observation is identical, key by key, to a control
                               run whose only difference is the responsibility source, and no
                               crd_* key appears in the policy observation.

Any failure is STOP_REGRET_SIGNAL and the source is not trained on.
Usage: python eucrd_regret_gate.py [run|replay|judge|all]
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(HERE)), "drl-manager"))
import ladder_run as lr  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out", "eucrd_regret")
EVAL_CFG_SRC = os.path.join(HERE, "config_rl_v2_eval.yml")
CFG = os.path.join(HERE, "config_eucrd_regret.yml")
TIERS = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0")
WINDOWS_N = 3
ZERO_TOL = 1e-12
SIGNAL = "candidate_carbon_regret"
START_LAG = 1                                  # the executor's start lag, ScheduleReplay.START_LAG


def build_config():
    """Blocks identical to the RL_V2 evaluation twin plus the responsibility source. The
    control block differs from its sibling ONLY in `crd.forecast.source`."""
    cfg = yaml.safe_load(open(EVAL_CFG_SRC))
    out = {"common": cfg["common"]}
    for tier in TIERS:
        src = cfg[f"rl2e_full_{tier}"]
        b = yaml.safe_load(yaml.safe_dump(src))
        b["experiment_name"] = f"rg_{tier}"; b["simulation_name"] = f"RG_{tier}"
        crd = dict(b.get("crd", {})); fc = dict(crd.get("forecast", {}))
        fc["source"] = SIGNAL; crd["forecast"] = fc; crd["enabled"] = True
        b["crd"] = crd
        out[f"rg_{tier}"] = b
        c = yaml.safe_load(yaml.safe_dump(b))
        c["experiment_name"] = f"rc_{tier}"; c["simulation_name"] = f"RC_{tier}"
        ccrd = dict(c["crd"]); cfc = dict(ccrd["forecast"])
        cfc["source"] = "instantaneous_carbon_cf"; ccrd["forecast"] = cfc; c["crd"] = ccrd
        out[f"rc_{tier}"] = c
        diff = sorted(k for k in set(b) | set(c) if b.get(k) != c.get(k))
        assert diff == ["crd", "experiment_name", "simulation_name"], diff
    with open(CFG, "w") as f:
        yaml.safe_dump(out, f, sort_keys=True)
    return CFG


def windows():
    man = json.load(open(os.path.join(HERE, "stage_a_out", "rl_v2", "manifest.json")))
    return man["windows"]["read"][:WINDOWS_N]


def _env(tag, tier, i, extra=None):
    e = {"OFFSET_GRID_DENSE": "1", "COVER_TIE": "index",
         "EVAL_DECISION_DUMP": os.path.join(OUT, f"{tag}_{tier}_k{i}_decisions.csv"),
         "EVAL_DECISION_DUMP_OBS": "1",
         "CRD_INFO_DUMP": os.path.join(OUT, f"{tag}_{tier}_k{i}_crd.csv"),
         "CRD_EMPTY_GRID_DIAG": "1"}
    e.update(extra or {})
    return e


def _clean(tag, tier, i):
    base = os.path.join(OUT, f"{tag}_{tier}_k{i}")
    for p in (base + ".csv", base + "_decisions.csv", base + "_decisions_obs.npz",
              base + "_crd.csv"):
        if os.path.exists(p):
            os.remove(p)


def run():
    """H1-H3, H5: each tier's arm on its own forecast, with a source-only control."""
    build_config()
    os.makedirs(OUT, exist_ok=True)
    for tier in TIERS:
        for i, off in enumerate(windows()):
            for tag in ("rg", "rc"):
                _clean(tag, tier, i)
                ok = lr._evaluate(CFG, f"{tag}_{tier}", i, off, "cover_argmax",
                                  os.path.join(OUT, f"{tag}_{tier}_k{i}.csv"), _env(tag, tier, i))
                print(f"{tag} {tier} k{i} ({off}): {'ok' if ok else 'FAILED'}", flush=True)


def schedule_from_decisions(path):
    """{cloudlet id: [site, start step]} from a decision dump: the executor starts a job at
    step + kappa + lag, so replaying this plan reproduces the same reservation grid."""
    plan, seen = {}, {}
    for r in csv.DictReader(open(path)):
        cid = int(r["cloudlet_id"])
        if cid < 0:
            continue
        step, kappa, site = int(r["step"]), int(r["kappa"]), int(r["site"])
        seen[cid] = seen.get(cid, 0) + 1
        plan[cid] = [site, step + kappa + START_LAG]        # last sighting wins
    return plan, {"jobs": len(plan), "multi_sighting": sum(1 for v in seen.values() if v > 1)}


def replay():
    """H4: the godeye arm's own trajectory, each tier's forecast swapped in for the credit."""
    os.makedirs(OUT, exist_ok=True)
    meta = {}
    for i, off in enumerate(windows()):
        plan, stats = schedule_from_decisions(os.path.join(OUT, f"rg_godeye_k{i}_decisions.csv"))
        p = os.path.join(OUT, f"plan_godeye_k{i}.json")
        json.dump({"schedule": {str(k): v for k, v in plan.items()},
                   "grid": list(range(73))}, open(p, "w"))
        meta[f"k{i}"] = stats
        for tier in TIERS:
            _clean("rp", tier, i)
            ok = lr._evaluate(CFG, f"rg_{tier}", i, off, "schedule_replay",
                              os.path.join(OUT, f"rp_{tier}_k{i}.csv"),
                              _env("rp", tier, i, {"SCHEDULE_JSON": p}))
            print(f"rp {tier} k{i} ({off}): {'ok' if ok else 'FAILED'} [{stats}]", flush=True)
    json.dump(meta, open(os.path.join(OUT, "replay_plans.json"), "w"), indent=1)


def _series(tag, tier, i, col=SIGNAL):
    p = os.path.join(OUT, f"{tag}_{tier}_k{i}_crd.csv")
    if not os.path.exists(p):
        return None, None
    rows = list(csv.DictReader(open(p)))
    vals = {(int(r["episode"]), int(r["step"])): float(r[col])
            for r in rows if r.get(col) not in (None, "")}
    d = os.path.join(OUT, f"{tag}_{tier}_k{i}_decisions.csv")
    dec = set()
    if os.path.exists(d):
        dec = {(int(r["episode"]), int(r["step"])) for r in csv.DictReader(open(d))
               if int(r["cloudlet_id"]) >= 0}
    return vals, dec


def _stats(tag, tier, col=SIGNAL):
    per_window, all_v, dec_v = [], [], []
    for i in range(WINDOWS_N):
        vals, dec = _series(tag, tier, i, col)
        if not vals:
            continue
        v = np.array(list(vals.values()), dtype=float)
        dv = np.array([x for k, x in vals.items() if k in dec], dtype=float)
        per_window.append({"window": i, "n": int(v.size), "mean": float(v.mean()),
                           "max": float(v.max()), "min": float(v.min()),
                           "n_decision": int(dv.size),
                           "decision_mean": (float(dv.mean()) if dv.size else None)})
        all_v.append(v); dec_v.append(dv)
    if not per_window:
        return {"windows": [], "mean": None, "max": None, "min": None, "decision_mean": None}
    a = np.concatenate(all_v); d = np.concatenate(dec_v) if any(x.size for x in dec_v) else np.array([])
    return {"windows": per_window, "mean": float(a.mean()), "max": float(a.max()),
            "min": float(a.min()),
            "decision_mean": (float(d.mean()) if d.size else None)}


def judge():
    res = {"windows": windows(), "signal": SIGNAL, "online": {}, "replay": {}, "empty_grid": {}}
    for tier in TIERS:
        res["online"][tier] = _stats("rg", tier)
        res["replay"][tier] = _stats("rp", tier)
        res["empty_grid"][tier] = _stats("rg", tier, "candidate_cover_mae_empty_grid")
    g, notes = {}, []
    z = res["online"]["godeye"]
    g["H1_zero_under_godeye"] = bool(z["max"] is not None and abs(z["max"]) <= ZERO_TOL)
    s75 = res["online"]["shrink75"]
    g["H2_alive_under_shrink75"] = bool(s75["windows"] and all(w["mean"] > 0 for w in s75["windows"]))
    mins = [res[k][t]["min"] for k in ("online", "replay") for t in TIERS
            if res[k][t]["min"] is not None]
    g["H3_non_negative"] = bool(mins and min(mins) >= 0.0)
    rep = {t: res["replay"][t]["decision_mean"] for t in TIERS}
    g["H4_fixed_state_pairing"] = bool(
        rep["godeye"] is not None and abs(rep["godeye"]) <= ZERO_TOL
        and all(rep[t] is not None and rep[t] > 0 for t in TIERS[1:]))
    notes.append({"replay_decision_means": rep,
                  "replay_ordering_reported_not_gated":
                      [t for t in TIERS[1:] if rep[t] is not None]})
    leaks = []
    for tier in TIERS:
        for i in range(WINDOWS_N):
            a = os.path.join(OUT, f"rg_{tier}_k{i}_decisions_obs.npz")
            b = os.path.join(OUT, f"rc_{tier}_k{i}_decisions_obs.npz")
            if not (os.path.exists(a) and os.path.exists(b)):
                leaks.append(f"{tier}_k{i}: missing dump"); continue
            za, zb = np.load(a), np.load(b)
            for k in sorted(set(za.files) | set(zb.files)):
                if k.startswith("_sentinel"):
                    continue
                if k.startswith("crd_"):
                    leaks.append(f"{tier}_k{i}: {k} present in the policy observation"); continue
                if k not in za.files or k not in zb.files:
                    leaks.append(f"{tier}_k{i}: key {k} only in one run"); continue
                if za[k].shape != zb[k].shape or not np.allclose(
                        np.nan_to_num(za[k]), np.nan_to_num(zb[k]), atol=0, rtol=0):
                    leaks.append(f"{tier}_k{i}: {k} differs")
    g["H5_no_observation_leak"] = not leaks
    res["leak"] = {"violations": leaks[:10], "n": len(leaks)}
    res["gates"] = g
    res["notes"] = notes
    keys = ("H1_zero_under_godeye", "H2_alive_under_shrink75", "H3_non_negative",
            "H4_fixed_state_pairing", "H5_no_observation_leak")
    res["verdict"] = ("REGRET_SIGNAL_PASS" if all(g[k] for k in keys)
                      else "STOP_REGRET_SIGNAL:" + ",".join(k for k in keys if not g[k]))
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "regret_gate.json"), "w"), indent=1)
    print(json.dumps({
        "gates": g, "verdict": res["verdict"],
        "online_decision_mean": {t: res["online"][t]["decision_mean"] for t in TIERS},
        "replay_decision_mean": rep,
        "empty_grid_mae_decision_mean": {t: res["empty_grid"][t]["decision_mean"] for t in TIERS},
    }, indent=1))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("run", "all"):
        run()
    if what in ("replay", "all"):
        replay()
    if what in ("judge", "all"):
        judge()
