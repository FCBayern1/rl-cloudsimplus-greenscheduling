"""EU-CRD forecast-signal hard gate, zero training (EUCRD_REPAIR_PLAN_2026_09_07 step 1).

The repaired responsibility source `crd.forecast.source: candidate_cover_mae` measures, per
decision state, the mean absolute difference between the candidate coverage the policy sees
(built from the arm's forecast) and the same quantity on the simulator's hidden future. Before
any training is spent on it, the signal must behave:

  G1 zero under a correct forecast   : the godeye tier gives max |signal| <= 1e-9 on every step.
  G2 alive under a degraded forecast : shrink75 gives a mean signal > 0 on every window.
  G3 monotone in the error           : the mean signal is non-decreasing along
                                       godeye < shrink75 < shrink50 < shrink25 < shrink0,
                                       with at most no inversion (exact rule below).
  G4 no observation leak             : the policy observation of a run with the repaired source
                                       is identical, key by key, to the same run without it.

Any failure is a STOP: the source is not used for training. No carbon is read and no policy is
trained by this file. Usage: python eucrd_signal_gate.py [run|judge]
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

OUT = os.path.join(HERE, "stage_a_out", "eucrd_signal")
EVAL_CFG_SRC = os.path.join(HERE, "config_rl_v2_eval.yml")
CFG = os.path.join(HERE, "config_eucrd_signal.yml")
TIERS = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0")
WINDOWS_N = 3                      # the first three reading windows; no window is trained on
ZERO_TOL = 1e-9


def build_config():
    """Blocks identical to the RL_V2 evaluation twin plus the repaired forecast source."""
    cfg = yaml.safe_load(open(EVAL_CFG_SRC))
    out = {"common": cfg["common"]}
    for tier in TIERS:
        src = cfg[f"rl2e_full_{tier}"]
        b = yaml.safe_load(yaml.safe_dump(src))
        b["experiment_name"] = f"sig_{tier}"; b["simulation_name"] = f"SIG_{tier}"
        crd = dict(b.get("crd", {})); fc = dict(crd.get("forecast", {}))
        fc["source"] = "candidate_cover_mae"; crd["forecast"] = fc; crd["enabled"] = True
        b["crd"] = crd
        out[f"sig_{tier}"] = b
        # control block: identical in every way except the forecast source, so the leak check
        # isolates the repaired source rather than the crd.enabled flag
        c = yaml.safe_load(yaml.safe_dump(b))
        c["experiment_name"] = f"ctl_{tier}"; c["simulation_name"] = f"CTL_{tier}"
        ccrd = dict(c["crd"]); cfc = dict(ccrd["forecast"]); cfc["source"] = "instantaneous_carbon_cf"
        ccrd["forecast"] = cfc; c["crd"] = ccrd
        out[f"ctl_{tier}"] = c
    with open(CFG, "w") as f:
        yaml.safe_dump(out, f, sort_keys=True)
    return CFG


def windows():
    man = json.load(open(os.path.join(HERE, "stage_a_out", "rl_v2", "manifest.json")))
    return man["windows"]["read"][:WINDOWS_N]


def run():
    build_config()
    os.makedirs(OUT, exist_ok=True)
    for tier in TIERS:
        for i, off in enumerate(windows()):
            for cell, tag in ((f"sig_{tier}", "sig"), (f"ctl_{tier}", "ctl")):
                out_csv = os.path.join(OUT, f"{tag}_{tier}_k{i}.csv")
                dump = out_csv.replace(".csv", "_decisions.csv")
                for p in (out_csv, dump, dump.replace(".csv", "_obs.npz")):
                    if os.path.exists(p):
                        os.remove(p)
                env = {"OFFSET_GRID_DENSE": "1", "COVER_TIE": "index", "EVAL_DECISION_DUMP": dump,
                       "EVAL_DECISION_DUMP_OBS": "1", "CRD_INFO_DUMP": os.path.join(OUT, f"{tag}_{tier}_k{i}_crd.csv")}
                ok = lr._evaluate(CFG, cell, i, off, "cover_argmax", out_csv, env)
                print(f"{tag} {tier} k{i} ({off}): {'ok' if ok else 'FAILED'}", flush=True)


def judge():
    res = {"tiers": {}, "leak": {}, "windows": windows()}
    for tier in TIERS:
        vals = []
        for i in range(WINDOWS_N):
            p = os.path.join(OUT, f"sig_{tier}_k{i}_crd.csv")
            if not os.path.exists(p):
                continue
            v = [float(r["candidate_cover_mae"]) for r in csv.DictReader(open(p)) if r.get("candidate_cover_mae") not in (None, "")]
            if v:
                vals.append({"window": i, "n": len(v), "mean": float(np.mean(v)), "max": float(np.max(v))})
        res["tiers"][tier] = {"windows": vals, "mean": (float(np.mean([w["mean"] for w in vals])) if vals else None),
                              "max": (float(np.max([w["max"] for w in vals])) if vals else None)}
    g = {}
    z = res["tiers"].get("godeye", {})
    g["G1_zero_under_godeye"] = bool(z.get("max") is not None and z["max"] <= ZERO_TOL)
    s75 = res["tiers"].get("shrink75", {})
    g["G2_alive_under_shrink75"] = bool(s75.get("windows") and all(w["mean"] > 0 for w in s75["windows"]))
    means = [res["tiers"][t]["mean"] for t in TIERS]
    g["G3_monotone_in_error"] = bool(all(m is not None for m in means) and all(means[i] <= means[i + 1] + 1e-12 for i in range(len(means) - 1)))
    g["means_by_tier"] = {t: res["tiers"][t]["mean"] for t in TIERS}
    # G4: the policy observation must be identical with and without the repaired source
    leaks = []
    for tier in TIERS:
        for i in range(WINDOWS_N):
            a = os.path.join(OUT, f"sig_{tier}_k{i}_decisions_obs.npz")
            b = os.path.join(OUT, f"ctl_{tier}_k{i}_decisions_obs.npz")
            if not (os.path.exists(a) and os.path.exists(b)):
                leaks.append(f"{tier}_k{i}: missing dump"); continue
            za, zb = np.load(a), np.load(b)
            keys = sorted(set(za.files) | set(zb.files))
            for k in keys:
                if k.startswith("_sentinel"):
                    continue
                if k not in za.files or k not in zb.files:
                    leaks.append(f"{tier}_k{i}: key {k} only in one run"); continue
                if za[k].shape != zb[k].shape or not np.allclose(np.nan_to_num(za[k]), np.nan_to_num(zb[k]), atol=0, rtol=0):
                    leaks.append(f"{tier}_k{i}: {k} differs")
    g["G4_no_observation_leak"] = not leaks
    res["leak"] = {"violations": leaks[:10], "n": len(leaks)}
    res["gates"] = g
    res["verdict"] = "SIGNAL_GATE_PASS" if all(g[k] for k in ("G1_zero_under_godeye", "G2_alive_under_shrink75", "G3_monotone_in_error", "G4_no_observation_leak")) else \
        "STOP_SIGNAL_GATE:" + ",".join(k for k in ("G1_zero_under_godeye", "G2_alive_under_shrink75", "G3_monotone_in_error", "G4_no_observation_leak") if not g[k])
    json.dump(res, open(os.path.join(OUT, "signal_gate.json"), "w"), indent=1)
    print(json.dumps({"gates": g, "verdict": res["verdict"], "tiers": {t: res["tiers"][t]["mean"] for t in TIERS}}, indent=1))
    return res


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "run"
    if what in ("run", "all"):
        run()
    if what in ("judge", "all"):
        judge()
