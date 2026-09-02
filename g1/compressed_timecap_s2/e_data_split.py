"""Scheme 2-E data split: fresh turbines, disjoint windows, one mechanical rule.

Work-order section 4: every S2 window (k = 1/9/17/25/33/41) is read, so Scheme 2-E needs
new DISCOVERY and CONFIRMATION sets with zero overlap between them, zero overlap with the
TimeCAP training turbines (1/15/30), and none of the five S2 scheduling turbines. TB13's
48 screening turbines are also excluded (their green values were read by that campaign's
offline screens), as are the sealed turbines 116/117 and every 2021 file that does not
carry the standard 52,559 rows. 2022 is banned outright.

From the surviving pool one seeded shuffle (seed 20260901, the campaign convention) deals
five turbines to DISCOVERY and the next five to CONFIRMATION, assigned to the C-regime
green sites in dealt order as DC0 = first two, DC1 = next two, DC2 = fifth. Windows stay
on the simulator's own offset schedule (1009 k mod 44950): DISCOVERY k = 2/10/18,
CONFIRMATION k = 26/34/42, pairwise spacing 8,072 rows >= 7,300 so no two windows can
touch even at the full episode cap, and none of the six was ever run by any arm.
"""
from __future__ import annotations

import csv
import glob
import hashlib
import json
import os
import re

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
SPL = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/windProduction/simplified")
SEED = 20260901
ROWS = 52559
OFFSET_RANGE = 44950
EPISODE_ROWS_MAX = 7200
SPACING = 7300
DISCOVERY_K = (2, 10, 18)
CONFIRMATION_K = (26, 34, 42)
S2_TURBINES = {12, 36, 91, 95, 96}
TRAIN_TURBINES = {1, 15, 30}
SEALED = {116, 117}


def tb13_turbines():
    txt = open(os.path.join(REPO, "g1/tb13/data_split.txt")).read()
    out = set()
    for part in ("DISCOVERY", "CONFIRMATION"):
        out.update(int(x) for x in txt.split(part + " [")[1].split("]")[0].split(","))
    return out


def fresh_pool():
    have = {}
    for p in glob.glob(os.path.join(SPL, "Turbine_*_2021.csv")):
        m = re.match(r"Turbine_(\d+)_2021\.csv", os.path.basename(p))
        if m:
            have[int(m.group(1))] = sum(1 for _ in open(p)) - 1
    banned = S2_TURBINES | TRAIN_TURBINES | SEALED | tb13_turbines()
    return sorted(t for t, n in have.items() if n == ROWS and t not in banned)


def deal():
    pool = fresh_pool()
    rng = np.random.default_rng(SEED)
    order = [pool[i] for i in rng.permutation(len(pool))]
    d, c = order[:5], order[5:10]
    return {"discovery": {"turbines": d, "dc_map": {"0": d[:2], "1": d[2:4], "2": [d[4]]},
                          "windows_k": list(DISCOVERY_K),
                          "offsets": [(1009 * k) % OFFSET_RANGE for k in DISCOVERY_K]},
            "confirmation": {"turbines": c, "dc_map": {"0": c[:2], "1": c[2:4], "2": [c[4]]},
                             "windows_k": list(CONFIRMATION_K),
                             "offsets": [(1009 * k) % OFFSET_RANGE for k in CONFIRMATION_K]},
            "pool_size": len(pool), "seed": SEED, "rows": ROWS,
            "excluded": {"s2": sorted(S2_TURBINES), "train": sorted(TRAIN_TURBINES),
                         "sealed": sorted(SEALED), "tb13": sorted(tb13_turbines())}}


def main():
    split = deal()
    offs = split["discovery"]["offsets"] + split["confirmation"]["offsets"]
    assert all(abs(a - b) >= SPACING for i, a in enumerate(offs)
               for b in offs[i + 1:]), "window spacing violated"
    assert all(o + EPISODE_ROWS_MAX <= ROWS for o in offs), "window exceeds the trace"
    assert not set(split["discovery"]["turbines"]) & set(split["confirmation"]["turbines"])
    blob = json.dumps(split, sort_keys=True, separators=(",", ":"))
    split["split_sha"] = hashlib.sha256(blob.encode()).hexdigest()[:16]
    path = os.path.join(HERE, "e_data_split.json")
    with open(path, "w") as f:
        json.dump(split, f, indent=2, sort_keys=True)
    print(json.dumps({k: split[k] for k in ("discovery", "confirmation", "pool_size",
                                            "split_sha")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
