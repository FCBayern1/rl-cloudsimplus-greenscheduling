"""Zero-training decomposition of the temporal (defer) lever on a wind window.

For every job the brown share of its own energy is computed under three policies on the
same green truth, solo (no competition for green), marginal accounting (host floor sunk):

  run-now   : start at arrival on the greenest DC
  myopic    : start at the first row where some DC covers the job fully, else at the wait
              cap (needs only the present, no forecast)
  oracle    : the start in [arrival, arrival + wait_cap] with the least brown energy
              (needs the forecast)

forecast-only lever = myopic - oracle. wasted waits = myopic waited the full cap and still
ran partly brown (a deadline miss under tight slack; the oracle runs those jobs at once).

Usage: python lever_decomp.py <runtime_rows> <wait_cap_rows> [config_yml] [cell]
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WIND = ROOT / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TRACES = ROOT / "cloudsimplus-gateway/src/main/resources/traces/s2"
P_JOB_W = 132.7  # one 32-PE job on a woken RS500A: 51.4 W floor + 81.3 W dynamic


def turbine_kw(tid: int, year: int = 2021) -> np.ndarray:
    with open(WIND / f"Turbine_{tid}_{year}.csv") as f:
        return np.array([float(r["power_kw"] or 0) for r in csv.DictReader(f)])


def green_matrix(datacenters, offset_rows: int, divisor: float, rows: int = 3000) -> np.ndarray:
    """(n_green_dcs, rows) green watts on the planner's own row mapping (offset + tz)."""
    out = []
    for d in datacenters:
        if not d.get("turbine_ids"):
            continue
        acc = sum(turbine_kw(t) for t in d["turbine_ids"])
        base = offset_rows + int(d.get("time_zone_offset_rows", 0))
        out.append(acc[base:base + rows] * 1000.0 / divisor)
    return np.array(out)


def decompose(G: np.ndarray, arrivals, runtime: int, wait_cap: int, p_job: float = P_JOB_W) -> dict:
    gf = np.clip(G / p_job, 0.0, 1.0)

    def brown(s):  # brown share of the job's energy if started at row s on the greenest DC
        return float((1.0 - gf[:, s:s + runtime].max(axis=0)).sum()) / runtime

    now = myo = best = 0.0
    wasted = full_now = 0
    for a in arrivals:
        b_now = brown(a)
        s_m = next((s for s in range(a, a + wait_cap + 1) if gf[:, s].max() >= 1.0), a + wait_cap)
        b_myo = brown(s_m)
        b_best = min(brown(s) for s in range(a, a + wait_cap + 1))
        now += b_now
        myo += b_myo
        best += b_best
        wasted += int(s_m == a + wait_cap and b_myo > 0)
        full_now += int(b_now == 0)
    n = len(arrivals)
    return {
        "n": n,
        "brown_now": now / n,
        "brown_myopic": myo / n,
        "brown_oracle": best / n,
        "lever_total": (now - best) / n,
        "lever_forecast_only": (myo - best) / n,
        "wasted_wait_rate": wasted / n,
        "full_green_now_rate": full_now / n,
    }


def main(argv):
    import yaml
    runtime, wait_cap = int(argv[1]), int(argv[2])
    cfg_path = argv[3] if len(argv) > 3 else str(ROOT / "g1/compressed_timecap_s2/config_s2h_m1.yml")
    cell = argv[4] if len(argv) > 4 else "s2_r48_w72_c3_n50"
    blk = yaml.safe_load(open(cfg_path))[cell]
    offset = int(argv[5]) if len(argv) > 5 else 2018
    G = green_matrix(blk["datacenters"], offset, float(blk["compressed_power_divisor"]))
    with open(TRACES / f"{cell}_pes32.csv") as f:
        arrivals = [int(float(r["arrival_time"])) for r in csv.DictReader(f)]
    r = decompose(G, arrivals, runtime, wait_cap)
    print(f"{cell} R={runtime} WC={wait_cap} divisor={blk['compressed_power_divisor']}: "
          f"brown now {100*r['brown_now']:.1f}% myopic {100*r['brown_myopic']:.1f}% oracle {100*r['brown_oracle']:.1f}% | "
          f"forecast-only lever {100*r['lever_forecast_only']:.1f}pp | wasted waits {100*r['wasted_wait_rate']:.0f}% | "
          f"fully green now {100*r['full_green_now_rate']:.0f}%")


if __name__ == "__main__":
    main(sys.argv)
