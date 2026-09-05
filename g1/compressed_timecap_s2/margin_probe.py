"""Stage D' deadline-safe DEFER mask margin, set mechanically (STAGE_D_PRIME_DESIGN §10).

Reads the saturated-dispatch probe rows (run_stage_a.py hz_margin_probe) and returns
    margin_steps = ceil(max route->exec-start delay / timestep) + 1
    margin_sec   = margin_steps * timestep
over all probe windows. Never derived from carbon or training results.

Usage: python margin_probe.py [timestep_sec]
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))


def margin_from_delays(max_delays_sec, timestep_sec):
    """Pure. max_delays_sec: per-window max route->start delays."""
    worst = max([float(x) for x in max_delays_sec] + [0.0])
    steps = int(math.ceil(worst / float(timestep_sec))) + 1
    return {"worst_delay_sec": worst, "margin_steps": steps, "margin_sec": steps * float(timestep_sec)}


def main():
    ts = float(sys.argv[1]) if len(sys.argv) > 1 else 1.0
    d = os.path.join(HERE, "stage_a_out", "dprime_margin_probe")
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        r = list(csv.DictReader(open(f)))[-1]
        rows.append({"file": os.path.basename(f), "max": float(r.get("ep_route_to_start_max_sec", 0) or 0),
                     "p95": float(r.get("ep_route_to_start_p95_sec", 0) or 0),
                     "n": int(float(r.get("ep_route_to_start_n", 0) or 0)),
                     "forced": float(r.get("deadline_forced_count", 0) or 0),
                     "ontime": float(r.get("ontime_mi_share", 1.0) or 1.0)})
    if not rows:
        print("no probe rows; run run_stage_a.py hz_margin_probe first")
        return
    out = margin_from_delays([x["max"] for x in rows], ts)
    out["rows"] = rows
    out["timestep_sec"] = ts
    with open(os.path.join(HERE, "stage_a_out", "dprime_margin.json"), "w") as fh:
        json.dump(out, fh, indent=2)
    for x in rows:
        print(f"{x['file']:36s} max {x['max']:8.1f}s  p95 {x['p95']:8.1f}s  n {x['n']:5d}  forced {x['forced']:.0f}  ontime {x['ontime']:.3f}")
    print(f"worst {out['worst_delay_sec']:.1f}s -> margin {out['margin_steps']} steps = {out['margin_sec']:.1f}s")


if __name__ == "__main__":
    main()
