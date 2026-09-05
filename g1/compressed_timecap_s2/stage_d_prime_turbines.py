"""Deterministic never-used turbine set for the D' formal scene (STAGE_D_PRIME_DESIGN §24–25).

Rule (reads no wind value):
  candidates = real turbine ids (1..146) that (i) appear in no tracked experiment config,
               audit or report (structured inventory), and (ii) have complete 2020 and 2021
               files (32,224 / 52,559 rows);
  order      = sha256("stage-d-prime-turbines-v1:" + id), ascending;
  choice     = the first five, mapped onto the HZ structure in hash order:
               DC0 <- ids 1,2   DC1 <- ids 3,4   DC2 <- id 5   (DC3, DC4 have no turbines).
The chosen set then has to pass the zero-training HZ scene gate, the TimeCAP error
calibration (calibrated_shrink_hz_v2) and the wiring gate afresh; nothing is inherited.

Usage: python stage_d_prime_turbines.py [--select]   (default prints candidates only)
"""
from __future__ import annotations

import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
SPLIT = os.path.join(REPO, "cloudsimplus-gateway", "src", "main", "resources", "windProduction", "split")
TAG = "stage-d-prime-turbines-v1"
ROWS = {2020: 32224, 2021: 52559}
STRUCTURE = ((0, 2), (1, 2), (2, 1))          # (dc index, number of turbines), HZ layout


def choose(candidates, tag=TAG, structure=STRUCTURE):
    """Pure. candidates: iterable of eligible ids. Returns the hash-ordered choice mapped to DCs."""
    order = sorted(set(int(c) for c in candidates), key=lambda i: hashlib.sha256(f"{tag}:{i}".encode()).hexdigest())
    need = sum(n for _dc, n in structure)
    if len(order) < need:
        return {"status": "STOP_NO_CANDIDATES", "n_candidates": len(order)}
    picked = order[:need]
    mapping, k = {}, 0
    for dc, n in structure:
        mapping[dc] = picked[k:k + n]
        k += n
    return {"status": "OK", "tag": tag, "n_candidates": len(order), "turbines": picked, "dc_turbines": mapping}


def eligible(inventory_path=None):
    inv = json.load(open(inventory_path or os.path.join(HERE, "stage_a_out", "turbine_usage_inventory.json")))
    never = inv["never_used_complete_2021"]
    out = []
    for i in never:
        ok = True
        for y, rows in ROWS.items():
            p = os.path.join(SPLIT, f"Turbine_{i}_{y}.csv")
            if not os.path.exists(p) or sum(1 for _ in open(p)) - 1 < rows:
                ok = False
        if ok:
            out.append(int(i))
    return out


def main():
    cands = eligible()
    res = choose(cands)
    res["candidates"] = cands
    print(json.dumps({k: res[k] for k in res if k != "candidates"}, indent=1))
    if "--select" in sys.argv:
        with open(os.path.join(HERE, "stage_a_out", "stage_d_prime_turbines.json"), "w") as f:
            json.dump(res, f, indent=2)
        print("written stage_a_out/stage_d_prime_turbines.json")


if __name__ == "__main__":
    main()
