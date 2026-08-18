#!/usr/bin/env python3
"""SQT2.1 series generator (docs/SQT2_SCENARIO_SPEC.md, Codex-approved revisions).

Synchronized square-wave green with pre-registered bimodal trough durations:
    ON  ~ U[1500, 2700] s                     (mean 2100)
    OFF ~ 80%: U[300, 1500]  (short, waitable)
          20%: U[2700, 4500] (long, not waitable)
Expected green ratio 2100/(2100+1440) = 59.3%. Short/long troughs carry equal
TIME weight (720 s each per cycle), so a job landing at a random time meets
either class with ~equal probability (kills the length-bias flaw of v1).

Turbine IDs 95xx (fresh range - the 9xxx overwrite incident stays unrepeatable).
Per-DC peak power reproduces the v3 measured capacities H_d so the demand side
of the testbed is comparable. 90000 rows (~25 mean cycles) so ten evaluation
windows over offset range 72000 cover >= 8 distinct trough instances.

Writes an artifact JSON (schedule stats, trough instances, seeds) next to the
CSVs for the preflight exposure checks.
"""
import argparse
import json
import random
from pathlib import Path

# v3 topology mapping: 95xx twin of each 9xxx turbine, same DC assignment.
# Per-DC capacity H_d (sim W) from calib/v3_anti_capacity.json green DCs.
DC_TURBINES = {0: [9512, 9536], 1: [9595, 9591], 2: [9596], 5: [9501, 9503]}
H_D_W = {0: 595.93, 1: 545.33, 2: 211.0, 5: 504.62}

ON_LO, ON_HI = 1500, 2700
OFF_SHORT = (300, 1500)
OFF_LONG = (2700, 4500)
P_SHORT = 0.8


def build_schedule(n_rows: int, seed: int):
    """Synchronized ON/OFF schedule; returns (on_flags, trough_instances)."""
    rng = random.Random(seed)
    on = [False] * n_rows
    troughs = []
    r, green = 0, True
    while r < n_rows:
        if green:
            dur = rng.randint(ON_LO, ON_HI)
        else:
            if rng.random() < P_SHORT:
                dur = rng.randint(*OFF_SHORT)
                kind = "short"
            else:
                dur = rng.randint(*OFF_LONG)
                kind = "long"
            troughs.append({"start": r, "dur": dur, "kind": kind})
        for j in range(r, min(r + dur, n_rows)):
            on[j] = green
        r += dur
        green = not green
    return on, troughs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=90000)
    ap.add_argument("--seed", type=int, default=20260818)   # pre-registered
    ap.add_argument("--divisor", type=float, required=True,
                    help="compressed_power_divisor of the target experiment")
    ap.add_argument("--out", default="../cloudsimplus-gateway/src/main/resources/windProduction/simplified")
    ap.add_argument("--year", type=int, default=2021)
    args = ap.parse_args()

    on, troughs = build_schedule(args.rows, args.seed)
    green_ratio = sum(on) / len(on)
    out = Path(args.out)
    for dc, tids in DC_TURBINES.items():
        peak_kw_each = H_D_W[dc] * args.divisor / 1000.0 / len(tids)
        body = "\n".join(
            f"2021-01-01 00:00:00,{peak_kw_each if flag else 0.0:.3f}"
            for flag in on)
        for tid in tids:
            (out / f"Turbine_{tid}_{args.year}.csv").write_text(
                "timestamp,power_kw\n" + body + "\n")
        print(f"DC{dc}: {tids} peak {peak_kw_each:.2f} kW each "
              f"(sum -> H_d {H_D_W[dc]:.1f} W sim-scale)")

    short = [t for t in troughs if t["kind"] == "short"]
    long_ = [t for t in troughs if t["kind"] == "long"]
    art = {"spec": "SQT2.1", "seed": args.seed, "rows": args.rows,
           "green_ratio": round(green_ratio, 4),
           "on_range": [ON_LO, ON_HI], "off_short": OFF_SHORT,
           "off_long": OFF_LONG, "p_short": P_SHORT,
           "trough_count": len(troughs),
           "short_count": len(short), "long_count": len(long_),
           "short_time_s": sum(t["dur"] for t in short),
           "long_time_s": sum(t["dur"] for t in long_),
           "troughs": troughs}
    art_path = Path("calib/sqt2_schedule.json")
    art_path.parent.mkdir(exist_ok=True)
    art_path.write_text(json.dumps(art, indent=1))
    print(f"green ratio {green_ratio:.3f} (target 0.55-0.65) | "
          f"troughs {len(troughs)} = {len(short)} short + {len(long_)} long | "
          f"time split {art['short_time_s']}s / {art['long_time_s']}s")
    assert 0.55 <= green_ratio <= 0.65, "green ratio out of band - do not ship"


if __name__ == "__main__":
    main()
