"""Stage A' verdict reader: the dose-response ladder, judged by the frozen criteria.

Addendum A section 3, mechanically:

    benefit(tier)   = pooled(blind) - pooled(tier), pooled over the frozen region's
                      cells x DISCOVERY windows, blind = the Stage-A frozen arm
    monotonicity    benefit non-increasing along godeye -> s05 -> s15 -> s30 -> s60,
                    ties allowed, nothing may exceed godeye
    negatives       shuffle and anti each retain at most 50% of godeye's benefit and
                    may not exceed it
    realistic tier  timecap_cal retaining >= 50% clears Stage D; in (0, 50%) is
                    marginal; <= 0 is STOP_REALISTIC_QUALITY
    contract        every tier, cell and window passes the same contract as Stage A
    zero point      the godeye tier must reproduce oracle144 (reported as a check)
"""
from __future__ import annotations

import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_stage_a as ra      # noqa: E402
import stage_a_verdict as sv  # noqa: E402

OUT = ra.OUT
NOISE_ORDER = ("godeye", "s05", "s15", "s30", "s60")
EPS = 1e-12


def _carbon(dirname, cell, k):
    path = os.path.join(OUT, dirname, f"{cell}_k{k}.csv")
    if not os.path.exists(path):
        return None, "missing"
    r = list(csv.DictReader(open(path)))[-1]
    if not sv._contract_ok(r):
        return None, "contract"
    return float(r["total_carbon_kg"]), None


def read_verdict():
    art = json.load(open(os.path.join(OUT, "blind_freeze.json")))
    blind_arm = art["frozen_blind"]
    cells = ra.stable_region_cells()
    wins = [k for k, _o in ra.windows()]

    pooled, problems = {}, {}
    for tier in ra.TIERS:
        vals, bad = [], []
        for cell in cells:
            for k in wins:
                c, why = _carbon(f"tier_{tier}", cell, k)
                if c is None:
                    bad.append((cell, k, why))
                else:
                    vals.append(c)
        pooled[tier] = sum(vals) / len(vals) if vals else None
        problems[tier] = bad
    blind_vals, blind_bad = [], []
    for cell in cells:
        for k in wins:
            c, why = _carbon(blind_arm, cell, k)
            (blind_vals.append(c) if c is not None else blind_bad.append((cell, k, why)))
    blind = sum(blind_vals) / len(blind_vals)
    o144_vals = [c for cell in cells for k in wins
                 for c in [_carbon("oracle144_planner", cell, k)[0]] if c is not None]
    o144 = sum(o144_vals) / len(o144_vals)

    complete = all(pooled[t] is not None and not problems[t] for t in ra.TIERS) \
        and not blind_bad
    benefit = {t: (blind - pooled[t]) if pooled[t] is not None else None
               for t in ra.TIERS}
    g_ben = benefit["godeye"]
    retention = {t: (benefit[t] / g_ben if g_ben and benefit[t] is not None else None)
                 for t in ra.TIERS}

    mono = all(benefit[NOISE_ORDER[i + 1]] <= benefit[NOISE_ORDER[i]] + EPS
               for i in range(len(NOISE_ORDER) - 1)) if complete else False
    none_exceed = all(benefit[t] <= g_ben + EPS for t in ra.TIERS) if complete else False
    negatives = all(retention[t] is not None and retention[t] <= 0.5 + EPS
                    for t in ("shuffle", "anti")) if complete else False
    cal = retention.get("timecap_cal")
    if not complete:
        realistic = "INCOMPLETE"
    elif cal is not None and cal >= 0.5:
        realistic = "PASS_STAGE_D_ALLOWED"
    elif cal is not None and cal > 0:
        realistic = "MARGINAL"
    else:
        realistic = "STOP_REALISTIC_QUALITY"

    gates = {"complete_and_contract_green": complete,
             "monotone_noise_axis": mono,
             "nothing_exceeds_godeye": none_exceed,
             "negatives_destroy_half": negatives}
    verdict = ("PASS_STAGE_APRIME" if all(gates.values())
               and realistic == "PASS_STAGE_D_ALLOWED"
               else "STOP_REALISTIC_QUALITY" if all(gates.values())
               and realistic == "STOP_REALISTIC_QUALITY"
               else "MARGINAL_REALISTIC_QUALITY" if all(gates.values())
               else "STOP_LADDER_GATE")
    return {"frozen_blind": blind_arm, "region_cells": len(cells),
            "pooled_blind": blind, "pooled_oracle144": o144,
            "pooled": pooled, "benefit": benefit, "retention": retention,
            "godeye_equals_oracle144_check": {
                "godeye": pooled["godeye"], "oracle144": o144,
                "rel_diff": (abs(pooled["godeye"] - o144) / o144) if pooled["godeye"]
                            is not None and o144 else None},
            "gates": gates, "realistic_tier": realistic, "verdict": verdict,
            "problems": {t: v[:10] for t, v in problems.items() if v}}


def main():
    out = read_verdict()
    with open(os.path.join(OUT, "stage_aprime_verdict.json"), "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "problems"},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
