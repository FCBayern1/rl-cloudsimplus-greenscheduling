"""Small-step trial reading (reports/EUCRD_SMALL_STEP_PREREG.md §3). Pools the deterministic
deployments of the four arms over the six development windows, per tier, and applies the
frozen direction screen. Screens direction only; no verdict on the matched pair.

Usage: python small_step_judge.py
"""
from __future__ import annotations

import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage_a_out", "small_step")
ARMS = ("ss_base", "ss_on", "ss_off", "ss_norw")
TIERS = ("godeye", "shrink75")
MARGIN = 0.05
KEYS = ("total_carbon_kg", "completion_rate", "completion_rate_mi", "ontime_mi_share",
        "deadline_forced_count")


def _rows(arm, tier):
    out = []
    for p in sorted(glob.glob(os.path.join(OUT, "eval", f"{arm}_{tier}_k*.csv"))):
        rs = list(csv.DictReader(open(p)))
        if rs:
            out.append({k: float(rs[0][k]) for k in KEYS if rs[0].get(k) not in (None, "")})
    return out


def _pool(rows):
    if not rows:
        return None
    return {k: float(np.mean([r[k] for r in rows if k in r])) for k in KEYS if any(k in r for r in rows)} | {"n_windows": len(rows)}


def judge():
    res = {"arms": {}, "screen": {}}
    for arm in ARMS:
        res["arms"][arm] = {t: _pool(_rows(arm, t)) for t in TIERS}

    def rel(a, b, key="total_carbon_kg"):
        if not a or not b or key not in a or key not in b or b[key] == 0:
            return None
        return (a[key] - b[key]) / b[key]

    for tier in TIERS:
        on, off, base = (res["arms"][a][tier] for a in ("ss_on", "ss_off", "ss_base"))
        r = rel(on, off); dc = (on or {}).get("completion_rate_mi"); dcf = (off or {}).get("completion_rate_mi")
        if r is None:
            verdict = "UNREADABLE"
        elif r > MARGIN and (dc is None or dcf is None or dc <= dcf):
            verdict = "CLEARLY_WORSE"
        elif r < -MARGIN and (dc is None or dcf is None or dc >= dcf):
            verdict = "CLEARLY_BETTER"
        else:
            verdict = "LITTLE_CHANGE"
        res["screen"][tier] = {
            "on_vs_off_carbon_rel": r, "on_completion_mi": dc, "off_completion_mi": dcf,
            "on_vs_base_carbon_rel": rel(on, base), "off_vs_base_carbon_rel": rel(off, base),
            "norw_vs_base_carbon_rel": rel(res["arms"]["ss_norw"][tier], base),
            "verdict": verdict,
        }
    res["verdict"] = "SMALL_STEP:" + res["screen"]["shrink75"]["verdict"] + \
        ";clean:" + res["screen"]["godeye"]["verdict"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "small_step.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    judge()
