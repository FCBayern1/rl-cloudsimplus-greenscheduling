"""Ladder-v2 verdict reader, frozen before the CONFIRMATION sweep is interpreted.

Implements LADDER_V2_PREREG sections 3 and 4 mechanically:

    per cell, per arm:   I = (sum over the three windows of carbon)
                             / (sum over windows of finished_cloudlets x mi_per_job)
    retention            R_i(q) = (I_blind,i - I_q,i) / (I_blind,i - I_godeye,i)
                         denominator must be positive; non-positive cells are excluded
                         and counted, more than 20% excluded -> INVALID
    statistic            median over the frozen 97-cell region

    gates                medians monotone non-increasing godeye -> s05 -> s15 -> s30 -> s60
                         shuffle and anti medians <= 50%
                         checkpoint_residual_surrogate_v2 median >= 50%
                         contract green on every arm, cell and window
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_stage_a as ra      # noqa: E402
import stage_a_verdict as sv  # noqa: E402

OUT = ra.OUT
NOISE_ORDER = ("godeye", "s05", "s15", "s30", "s60")
EPS = 1e-12
EXCLUDE_CAP = 0.20


def _mi_per_job():
    man = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                      "s2_manifest.json")))
    return {name: rep["mi_per_job"] for name, rep in man["reports"].items()}


def _cell_intensity(dirname, cell, wins, mi_job):
    carbon, mi, bad = 0.0, 0.0, []
    for k in wins:
        path = os.path.join(OUT, dirname, f"{cell}_k{k}.csv")
        if not os.path.exists(path):
            return None, [(k, "missing")]
        r = list(csv.DictReader(open(path)))[-1]
        if not sv._contract_ok(r):
            bad.append((k, "contract"))
        carbon += float(r["total_carbon_kg"])
        mi += float(r["total_finished_cloudlets"]) * mi_job
    if bad:
        return None, bad
    return (carbon / mi if mi > 0 else None), []


def read_verdict():
    art = json.load(open(os.path.join(OUT, "blind_freeze.json")))
    blind_arm = art["frozen_blind"]
    cells = ra.stable_region_cells()
    wins = [k for k, _o in ra.windows("confirmation")]
    mi_map = _mi_per_job()

    arms = {"blind": f"conf_{blind_arm}"}
    for t in ra.TIERS_V2:
        arms[t] = f"conf_tier_{t}"

    I, problems = {a: {} for a in arms}, []
    for cell in cells:
        for label, dirname in arms.items():
            v, bad = _cell_intensity(dirname, cell, wins, mi_map[cell])
            if bad or v is None:
                problems.append({"cell": cell, "arm": label, "why": bad or "no MI"})
            I[label][cell] = v

    complete = not problems
    excluded, R = [], {t: [] for t in ra.TIERS_V2}
    for cell in cells:
        b, gd = I["blind"].get(cell), I["godeye"].get(cell)
        if b is None or gd is None:
            continue
        denom = b - gd
        if denom <= 0:
            excluded.append(cell)
            continue
        for t in ra.TIERS_V2:
            v = I[t].get(cell)
            if v is not None:
                R[t].append((b - v) / denom)
    frac_excluded = len(excluded) / max(len(cells), 1)
    medians = {t: (float(np.median(R[t])) if R[t] else None) for t in ra.TIERS_V2}

    mono = all(medians[NOISE_ORDER[i + 1]] is not None
               and medians[NOISE_ORDER[i]] is not None
               and medians[NOISE_ORDER[i + 1]] <= medians[NOISE_ORDER[i]] + EPS
               for i in range(len(NOISE_ORDER) - 1))
    gates = {
        "complete_and_contract_green": complete,
        "exclusions_within_cap": frac_excluded <= EXCLUDE_CAP,
        "monotone_noise_axis": mono,
        "shuffle_destroys_half": medians["shuffle"] is not None
                                 and medians["shuffle"] <= 0.5 + EPS,
        "anti_destroys_half": medians["anti"] is not None
                              and medians["anti"] <= 0.5 + EPS,
        "surrogate_retains_half": medians["checkpoint_residual_surrogate_v2"] is not None
                                  and medians["checkpoint_residual_surrogate_v2"]
                                  >= 0.5 - EPS,
    }
    if not gates["complete_and_contract_green"] or not gates["exclusions_within_cap"]:
        verdict = "INVALID"
    elif all(gates.values()):
        verdict = "PASS_LADDER_V2"
    else:
        verdict = "STOP_LADDER_V2"
    return {"frozen_blind": blind_arm, "region_cells": len(cells),
            "windows": wins, "median_retention": medians,
            "cells_per_tier": {t: len(R[t]) for t in ra.TIERS_V2},
            "excluded_nonpositive_denominator": excluded,
            "excluded_fraction": frac_excluded,
            "gates": gates, "verdict": verdict,
            "problems": problems[:20]}


def main():
    out = read_verdict()
    with open(os.path.join(OUT, "ladder_v2_verdict.json"), "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "problems"},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
