"""Reading of the matched pair (reports/EUCRD_MATCHED_PAIR_PREREG.md §3-§5 and Addendum D).

Pools the deterministic deployments of V_err and E_err over the six development windows, per
seed and per tier, applies the frozen primary criterion and reports every quantity Addendum D's
table needs. No threshold is computed from the data; nothing is dropped.

Usage: python matched_pair_judge.py [--out DIR]
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "stage_a_out", "matched_pair")
LINES = ("V", "E")
SEEDS = (20260911, 20260912, 20260913)
TIERS = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0", "shuffle", "anti",
         "calibrated_shrink_v1")
PRIMARY_TIER = "shrink75"
CLEAN_TIER = "godeye"
BENEFIT_MIN = 0.03            # pooled improvement required at the primary tier
CLEAN_TOLERANCE = 0.03        # E may not exceed V's clean carbon by more than this
N_WINDOWS = 6
METRICS = ("total_carbon_kg", "completion_rate", "completion_rate_mi", "ontime_mi_share",
           "deadline_forced_count", "global_decision_us_mean", "global_decision_us_p95",
           "global_decision_us_p99")


def _row(path):
    if not os.path.exists(path):
        return None
    rs = list(csv.DictReader(open(path)))
    if not rs:
        return None
    out = {}
    for k in METRICS:
        v = rs[0].get(k)
        if v not in (None, ""):
            try:
                out[k] = float(v)
            except ValueError:
                pass
    return out


def cell(arm, tier, window, out_dir):
    """One deployment: arm is 'V_s<seed>', 'E_s<seed>' or 'cover_argmax'."""
    return _row(os.path.join(out_dir, "eval", f"{arm}_{tier}_k{window}.csv"))


def per_window(arm, tier, out_dir):
    return [cell(arm, tier, w, out_dir) for w in range(N_WINDOWS)]


def pooled(cells):
    """Mean per window over the windows that produced a row; missing cells are counted, never
    silently dropped from the record."""
    got = [c for c in cells if c]
    if not got:
        return None
    out = {k: float(np.mean([c[k] for c in got if k in c]))
           for k in METRICS if any(k in c for c in got)}
    out["n_windows"] = len(got)
    out["n_missing"] = len(cells) - len(got)
    return out


def decisions(arm, tier, window, out_dir):
    """Chosen (site, kappa) per real job at this deployment, keyed by (episode, step, slot,
    cloudlet id). Padding slots (cloudlet id < 0) are excluded: action 0 is a legal choice
    ("dispatch now to site 0"), so a zero action cannot stand in for "no job"."""
    p = os.path.join(out_dir, "eval", f"{arm}_{tier}_k{window}_decisions.csv")
    if not os.path.exists(p):
        return {}
    out = {}
    for r in csv.DictReader(open(p)):
        try:
            cid = int(r["cloudlet_id"])
        except (KeyError, ValueError):
            continue
        if cid < 0:
            continue
        out[(r.get("episode"), r.get("step"), r.get("slot"), cid)] = (int(r["site"]), int(r["kappa"]))
    return out


def action_divergence(arms, out_dir):
    """Reported, not gated (Addendum E): how much the actions on real jobs differ between arms
    and against the rule, plus each arm's own site and kappa distribution."""
    out = {}
    for tier in TIERS:
        per_arm = {a: {} for a in arms}
        for a in arms:
            for w in range(N_WINDOWS):
                per_arm[a].update({(w,) + k: v for k, v in decisions(a, tier, w, out_dir).items()})
        dist = {}
        for a, d in per_arm.items():
            if not d:
                continue
            sites, kaps = {}, {}
            for site, kap in d.values():
                sites[site] = sites.get(site, 0) + 1
                kaps[kap] = kaps.get(kap, 0) + 1
            dist[a] = {"n_decisions": len(d), "sites": dict(sorted(sites.items())),
                       "kappa_mean": float(np.mean([k for _, k in d.values()])),
                       "kappa_top": sorted(kaps.items(), key=lambda x: -x[1])[:5]}
        pairs = {}
        names = [a for a in arms if per_arm.get(a)]
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                shared = set(per_arm[a]) & set(per_arm[b])
                if not shared:
                    continue
                same = sum(1 for k in shared if per_arm[a][k] == per_arm[b][k])
                same_site = sum(1 for k in shared if per_arm[a][k][0] == per_arm[b][k][0])
                dk = [abs(per_arm[a][k][1] - per_arm[b][k][1]) for k in shared]
                pairs[f"{a}|{b}"] = {"n_shared": len(shared),
                                     "identical_frac": same / len(shared),
                                     "same_site_frac": same_site / len(shared),
                                     "mean_abs_dkappa": float(np.mean(dk))}
        out[tier] = {"per_arm": dist, "pairs": pairs}
    return out


def judge(out_dir=OUT):
    res = {"out_dir": out_dir, "windows": N_WINDOWS, "seeds": list(SEEDS),
           "thresholds": {"benefit_min": BENEFIT_MIN, "clean_tolerance": CLEAN_TOLERANCE,
                          "primary_tier": PRIMARY_TIER, "clean_tier": CLEAN_TIER},
           "arms": {}, "per_window_carbon": {}, "paired": {}, "reference": {}}

    arms = [f"{L}_s{s}" for s in SEEDS for L in LINES] + ["cover_argmax"]
    for arm in arms:
        res["arms"][arm] = {t: pooled(per_window(arm, t, out_dir)) for t in TIERS}
    res["per_window_carbon"] = {
        t: {arm: [(c or {}).get("total_carbon_kg") for c in per_window(arm, t, out_dir)]
            for arm in arms} for t in TIERS}

    def carbon(arm, tier):
        a = res["arms"].get(arm, {}).get(tier)
        return None if not a else a.get("total_carbon_kg")

    # paired differences, one pair per seed, every tier (E - V, relative to V)
    for tier in TIERS:
        per_seed = {}
        for s in SEEDS:
            v, e = carbon(f"V_s{s}", tier), carbon(f"E_s{s}", tier)
            per_seed[str(s)] = {
                "V": v, "E": e,
                "abs_diff": (None if v is None or e is None else e - v),
                "rel_diff": (None if not v else (e - v) / v),
                "per_window_rel": [
                    (None if not (a and b and a.get("total_carbon_kg")) else
                     (b["total_carbon_kg"] - a["total_carbon_kg"]) / a["total_carbon_kg"])
                    for a, b in zip(per_window(f"V_s{s}", tier, out_dir),
                                    per_window(f"E_s{s}", tier, out_dir))],
            }
        rels = [d["rel_diff"] for d in per_seed.values() if d["rel_diff"] is not None]
        res["paired"][tier] = {
            "per_seed": per_seed,
            "rel_mean": (float(np.mean(rels)) if rels else None),
            "rel_min": (float(np.min(rels)) if rels else None),
            "rel_max": (float(np.max(rels)) if rels else None),
            "n_seeds_complete": len(rels),
            "all_negative": bool(rels and len(rels) == len(SEEDS) and all(r < 0 for r in rels)),
        }

    # the zero-parameter rule, reported at every tier and never used as a bar
    for tier in TIERS:
        r = carbon("cover_argmax", tier)
        res["reference"][tier] = {
            "cover_argmax_carbon": r,
            "V_vs_rule_rel": {str(s): (None if not r or carbon(f"V_s{s}", tier) is None
                                       else (carbon(f"V_s{s}", tier) - r) / r) for s in SEEDS},
            "E_vs_rule_rel": {str(s): (None if not r or carbon(f"E_s{s}", tier) is None
                                       else (carbon(f"E_s{s}", tier) - r) / r) for s in SEEDS},
        }

    # ── the frozen primary criterion (§4) ──────────────────────────────────────
    p = res["paired"][PRIMARY_TIER]
    clean = res["paired"][CLEAN_TIER]
    complete = p["n_seeds_complete"] == len(SEEDS) and clean["n_seeds_complete"] == len(SEEDS)
    benefit = bool(p["all_negative"] and p["rel_mean"] is not None and p["rel_mean"] <= -BENEFIT_MIN)
    clean_ok = bool(clean["rel_mean"] is not None and clean["rel_mean"] <= CLEAN_TOLERANCE)
    if not complete:
        verdict = "INCOMPLETE"
    elif not clean_ok:
        verdict = "FAIL_CLEAN_GUARD"
    elif benefit:
        verdict = "BENEFIT_ESTABLISHED"
    elif p["all_negative"] or (p["rel_mean"] is not None and p["rel_mean"] <= -BENEFIT_MIN):
        verdict = "SUGGESTIVE_NOT_ESTABLISHED"
    else:
        verdict = "NO_BENEFIT"
    res["primary"] = {"tier": PRIMARY_TIER, "rel_per_seed": {k: v["rel_diff"] for k, v in p["per_seed"].items()},
                      "rel_mean": p["rel_mean"], "all_three_negative": p["all_negative"],
                      "clean_rel_mean": clean["rel_mean"], "clean_guard_ok": clean_ok,
                      "verdict": verdict}

    # ── contracts, kept per cell (Addendum D: no cell is dropped) ──────────────
    res["contracts"] = {}
    for tier in TIERS:
        res["contracts"][tier] = {
            arm: {"completion_rate_mi": [(c or {}).get("completion_rate_mi") for c in per_window(arm, tier, out_dir)],
                  "ontime_mi_share": [(c or {}).get("ontime_mi_share") for c in per_window(arm, tier, out_dir)],
                  "deadline_forced_count": [(c or {}).get("deadline_forced_count") for c in per_window(arm, tier, out_dir)]}
            for arm in arms}
    # a carbon advantage with lower completion is flagged, not silently reported as an advantage
    flags = []
    for tier in TIERS:
        for s in SEEDS:
            v, e = res["arms"].get(f"V_s{s}", {}).get(tier), res["arms"].get(f"E_s{s}", {}).get(tier)
            if not v or not e:
                continue
            if e.get("total_carbon_kg", 0) < v.get("total_carbon_kg", 0) and \
               e.get("completion_rate_mi", 1) < v.get("completion_rate_mi", 1) - 1e-9:
                flags.append(f"{tier} s{s}: E lower carbon but lower completion")
            if e.get("n_missing") or v.get("n_missing"):
                flags.append(f"{tier} s{s}: missing cells V={v.get('n_missing')} E={e.get('n_missing')}")
    res["flags"] = flags

    # ── question 3 (Addendum E): what changed in the actions on REAL jobs ──────
    res["action_divergence"] = action_divergence(arms, out_dir)

    # cost, reported with any benefit
    res["cost"] = {arm: {"decision_us_mean": (res["arms"][arm].get(PRIMARY_TIER) or {}).get("global_decision_us_mean"),
                         "decision_us_p95": (res["arms"][arm].get(PRIMARY_TIER) or {}).get("global_decision_us_p95")}
                   for arm in arms}

    os.makedirs(out_dir, exist_ok=True)
    json.dump(res, open(os.path.join(out_dir, "matched_pair.json"), "w"), indent=1)
    ad = res["action_divergence"].get(PRIMARY_TIER, {})
    print(json.dumps({"primary": res["primary"],
                      "paired_shrink75": {k: v for k, v in res["paired"][PRIMARY_TIER].items() if k != "per_seed"},
                      "vs_rule_shrink75": res["reference"][PRIMARY_TIER],
                      "action_divergence_shrink75": {"per_arm": ad.get("per_arm"), "pairs": ad.get("pairs")},
                      "flags": res["flags"][:6]}, indent=1))
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default=OUT)
    judge(ap.parse_args().out)
