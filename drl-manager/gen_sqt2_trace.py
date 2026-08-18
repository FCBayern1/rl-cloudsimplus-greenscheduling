#!/usr/bin/env python3
"""SQT2.2 bimodal-slack trace generator (Codex step 3, 2026-08-18).

Derives from v3b_n1200.csv into NEW files (never overwrites): arrival, MI,
PES, id, row order unchanged; ONLY deadline is redrawn for the tight class.
Tight candidates get slack ~ U[200,900]; the loose class keeps its original
deadline. Assignment is seeded and STRATIFIED by runtime so both classes
carry comparable MI/runtime profiles (per-stratum tight quota). Mixtures are
generated for {30,40,50}% and selection between them may use ONLY exposure
statistics and control completion - never carbon.
"""
import csv
import pathlib
import random
from pathlib import Path

SRC = Path("../cloudsimplus-gateway/src/main/resources/traces/v3b_n1200.csv")
OUT = Path("../cloudsimplus-gateway/src/main/resources/traces")
SEED = 20260818
TIGHT_LO, TIGHT_HI = 200, 900
MIPS = 40000.0


def generate(frac_tight: float, seed: int = SEED, prefix: str = "sqt2"):
    rows = list(csv.DictReader(open(SRC)))
    runtimes = [max(1.0, float(r["length"]) / (max(1, int(r["pes_required"])) * MIPS))
                for r in rows]
    order = sorted(range(len(rows)), key=lambda i: runtimes[i])
    rng = random.Random(seed)
    tight = set()
    stratum = 60                       # 20 strata over the runtime ordering
    for s in range(0, len(order), stratum):
        block = order[s:s + stratum]
        k = round(len(block) * frac_tight)
        tight.update(rng.sample(block, k))
    out_rows = []
    for i, r in enumerate(rows):
        r = dict(r)
        if i in tight:
            slack = rng.randint(TIGHT_LO, TIGHT_HI)
            r["deadline"] = str(int(float(r["arrival_time"]) + runtimes[i] + slack))
        out_rows.append(r)
    name = f"{prefix}_n1200_t{int(frac_tight*100)}.csv"
    with open(OUT / name, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(out_rows)
    # artifact: tight cloudlet IDs (the preflight identifies the class from
    # this file, never by re-deriving thresholds)
    import json
    art = {"trace": name, "seed": seed, "frac_tight": frac_tight,
           "tight_slack_range": [TIGHT_LO, TIGHT_HI],
           "tight_cloudlet_ids": sorted(rows[i]["cloudlet_id"] for i in tight)}
    pathlib.Path(f"calib/{prefix}_trace_t{int(frac_tight*100)}.json").write_text(
        json.dumps(art, indent=1))
    return name, tight


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--fracs", default="0.30,0.40,0.50,0.60")
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--prefix", default="sqt2")
    args = ap.parse_args()
    for frac in [float(x) for x in args.fracs.split(",")]:
        name, tight = generate(frac, seed=args.seed, prefix=args.prefix)
        print(f"{name}: {len(tight)}/1200 tight jobs (seed {args.seed})")
