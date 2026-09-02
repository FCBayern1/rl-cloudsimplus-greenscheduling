"""Scheme 2-E DISCOVERY verdict reader, frozen before e_main results are interpreted.

Implements SCHEME2_ERROR_REGRET_PREREG section 5, gates 1-5, mechanically. Metric: per
cell, carbon intensity aggregated over the three DISCOVERY windows (sum carbon over sum
completed MI); gate statistics are medians across contract-valid cells.

    gate 1   clean (godeye) cuts total carbon vs the frozen strongest blind by >= 5%
             (median of per-cell relative reduction)
    gate 2   the primary error (calibrated_shrink_v1) raises carbon vs clean by >= 5%,
             or gives back >= 50% of clean's benefit over the blind (median)
    gate 3   direction consistent in >= 2 of 3 windows (pooled per window)
    gate 4   the primary error changed behavior: not every cell bitwise-identical to
             clean on (carbon, mean completion time, per-DC finished counts)
    gate 5   every arm, cell and window passes the contract

PASS -> the untouched CONFIRMATION may be read once under the same gates (6/7).
FAIL -> STOP_NO_LOAD_BEARING_FORECAST_ERROR.
"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gen_s2 as g            # noqa: E402
import ladder_v2_verdict as lv  # noqa: E402
import run_stage_a as ra      # noqa: E402
import stage_a_verdict as sv  # noqa: E402

OUT = ra.OUT
GATE1 = 0.05
GATE2_RAISE = 0.05
GATE2_GIVEBACK = 0.50


def _split(part):
    return json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                       "e_data_split.json")))[part]


def _row(dirname, cell, k):
    path = os.path.join(OUT, dirname, f"{cell}_k{k}.csv")
    if not os.path.exists(path):
        return None
    rows = list(csv.DictReader(open(path)))
    return rows[-1] if rows else None


def read_verdict(part="discovery"):
    ks = _split(part)["windows_k"]
    mi_map = lv._mi_per_job()
    freeze = json.load(open(os.path.join(OUT, "e_blind_freeze.json")))
    if freeze.get("status") != "FROZEN":
        raise RuntimeError("no frozen blind; the verdict has no denominator")
    blind_dir = f"e_{part[:4]}_{freeze['frozen_blind']}"
    arms = {"blind": blind_dir,
            "clean": f"e_{part[:4]}_tier_godeye",
            "primary": f"e_{part[:4]}_tier_calibrated_shrink_v1"}
    aux = {t: f"e_{part[:4]}_tier_{t}" for t in ("s30", "shuffle", "anti")}

    cells_out, problems = [], []
    per_window = {k: {"blind": 0.0, "clean": 0.0, "primary": 0.0} for k in ks}
    identical = []
    for cell in g.cells():
        name = g.cell_name(cell)
        vals, ok = {}, True
        sig = {}
        for label, d in {**arms, **aux}.items():
            carbon = mi = 0.0
            rowsig = []
            for k in ks:
                r = _row(d, name, k)
                if r is None or not sv._contract_ok(r):
                    ok = False
                    problems.append((name, label, k,
                                     "missing" if r is None else "contract"))
                    continue
                carbon += float(r["total_carbon_kg"])
                mi += float(r["total_finished_cloudlets"]) * mi_map[name]
                rowsig.append((r["total_carbon_kg"], r["mean_completion_time"],
                               tuple(r.get(f"finished_dc_{i}") for i in range(5))))
                if label in per_window[k]:
                    per_window[k][label] += float(r["total_carbon_kg"])
            vals[label] = carbon / mi if mi > 0 else None
            sig[label] = tuple(rowsig)
        if not ok or any(v is None for v in vals.values()):
            continue
        identical.append(sig["clean"] == sig["primary"])
        cells_out.append({"cell": name, **{k: vals[k] for k in vals}})

    b = np.array([c["blind"] for c in cells_out])
    cl = np.array([c["clean"] for c in cells_out])
    pr = np.array([c["primary"] for c in cells_out])
    g1v = float(np.median((b - cl) / b))
    raise_med = float(np.median((pr - cl) / cl))
    give = (pr - cl) / np.maximum(b - cl, 1e-15)
    give_med = float(np.median(give[(b - cl) > 0]))
    win_dir = sum(1 for k in ks
                  if per_window[k]["primary"] > per_window[k]["clean"])
    gates = {
        "g1_clean_beats_blind_5pc": g1v >= GATE1,
        "g2_primary_hurts": raise_med >= GATE2_RAISE or give_med >= GATE2_GIVEBACK,
        "g3_direction_2_of_3": win_dir >= 2,
        "g4_actions_changed": not all(identical) if identical else False,
        "g5_contract_green": not problems,
    }
    aux_med = {t: float(np.median((b - np.array([c[t] for c in cells_out])) / b))
               for t in aux}
    verdict = "PASS_E_DISCOVERY" if all(gates.values()) \
        else "STOP_NO_LOAD_BEARING_FORECAST_ERROR"
    return {"part": part, "frozen_blind": freeze["frozen_blind"],
            "cells_valid": len(cells_out), "gates": gates,
            "clean_vs_blind_median": g1v, "primary_vs_clean_median_raise": raise_med,
            "primary_giveback_median": give_med, "windows_adverse": win_dir,
            "aux_vs_blind_median": aux_med,
            "identical_cells": int(sum(identical)),
            "problems": problems[:20], "verdict": verdict}


def main():
    part = sys.argv[1] if len(sys.argv) > 1 else "discovery"
    out = read_verdict(part)
    with open(os.path.join(OUT, f"e_verdict_{part}.json"), "w") as f:
        f.write(json.dumps(out, sort_keys=True, indent=2))
    print(json.dumps({k: v for k, v in out.items() if k != "problems"},
                     sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
