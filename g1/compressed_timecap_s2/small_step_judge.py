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


def batch_identity():
    """Addendum A2: the first iteration's sampling statistics must agree across the trained
    arms (same restored policy, same seed, sampled before any update). Reported as a caveat."""
    DRL = os.path.join(os.path.dirname(os.path.dirname(HERE)), "drl-manager")
    out = {}
    for arm in ("ss_on", "ss_off", "ss_norw"):
        ps = sorted(glob.glob(os.path.join(DRL, "logs", "small_step", arm, "*", "PPO_*", "result.json")))
        if not ps:
            out[arm] = None; continue
        first = None
        for line in open(ps[0]):
            if line.strip():
                first = json.loads(line); break
        if not first:
            out[arm] = None; continue
        er = first.get("env_runners", {}) or {}
        out[arm] = {"episode_return_mean": er.get("episode_return_mean"),
                    "num_episodes": er.get("num_episodes"),
                    "num_env_steps_sampled": first.get("num_env_steps_sampled_lifetime")}
    vals = [json.dumps(v, sort_keys=True) for v in out.values() if v is not None]
    return {"per_arm": out, "identical": bool(vals and len(set(vals)) == 1)}


def judge():
    res = {"arms": {}, "screen": {}}
    for arm in ARMS:
        res["arms"][arm] = {t: _pool(_rows(arm, t)) for t in TIERS}

    def rel(a, b, key="total_carbon_kg"):
        if not a or not b or key not in a or key not in b or b[key] == 0:
            return None
        return (a[key] - b[key]) / b[key]

    def screen(a, b):
        """The frozen direction screen (prereg §3) on arm a against arm b."""
        r = rel(a, b); ca = (a or {}).get("completion_rate_mi"); cb = (b or {}).get("completion_rate_mi")
        if r is None:
            return r, "UNREADABLE"
        if r > MARGIN and (ca is None or cb is None or ca <= cb):
            return r, "CLEARLY_WORSE"
        if r < -MARGIN and (ca is None or cb is None or ca >= cb):
            return r, "CLEARLY_BETTER"
        return r, "LITTLE_CHANGE"

    for tier in TIERS:
        on, off, norw, base = (res["arms"][a][tier] for a in ("ss_on", "ss_off", "ss_norw", "ss_base"))
        # Addendum A: the primary pair isolates the reweighting switch; on-vs-off is the broader
        # forecast-channel control; every trained arm is also read against the pre-update state
        r_pri, v_pri = screen(on, norw)
        r_off, v_off = screen(on, off)
        res["screen"][tier] = {
            "primary_on_vs_norw_carbon_rel": r_pri, "primary_verdict": v_pri,
            "control_on_vs_off_carbon_rel": r_off, "control_verdict": v_off,
            "on_completion_mi": (on or {}).get("completion_rate_mi"),
            "norw_completion_mi": (norw or {}).get("completion_rate_mi"),
            "off_completion_mi": (off or {}).get("completion_rate_mi"),
            "base_completion_mi": (base or {}).get("completion_rate_mi"),
            "on_vs_base_carbon_rel": rel(on, base), "off_vs_base_carbon_rel": rel(off, base),
            "norw_vs_base_carbon_rel": rel(norw, base),
            "verdict": v_pri,
        }
        # contracts first, per window: forced deadlines and on-time share per arm
        res["screen"][tier]["contracts"] = {
            a: {k: (res["arms"][a][tier] or {}).get(k) for k in ("completion_rate_mi", "ontime_mi_share", "deadline_forced_count")}
            for a in ARMS}
    # per-window absolute carbon, so no single window can carry the reading
    res["per_window_carbon"] = {t: {a: [r["total_carbon_kg"] for r in _rows(a, t)] for a in ARMS} for t in TIERS}
    res["batch_identity"] = batch_identity()
    res["verdict"] = "SMALL_STEP:" + res["screen"]["shrink75"]["verdict"] + \
        ";clean:" + res["screen"]["godeye"]["verdict"]
    os.makedirs(OUT, exist_ok=True)
    json.dump(res, open(os.path.join(OUT, "small_step.json"), "w"), indent=1)
    print(json.dumps(res, indent=1))
    return res


if __name__ == "__main__":
    judge()
