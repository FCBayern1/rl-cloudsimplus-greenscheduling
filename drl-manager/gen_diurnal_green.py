#!/usr/bin/env python3
"""Generate SYNTHETIC DIURNAL green-energy CSVs for the Version-B (temporal-deferral) sim.

WHY: the real wind CSVs (windProduction/simplified) are spiky noise — cv~1.3, often 0W,
rare gusts, and POSITIVELY correlated across DCs (DC0-DC1 corr +0.69). That structure makes
both routing and deferral physically worthless: 72% of green is wasted while brown burns,
and there is no smooth, forecastable, *capturable* signal (see the static feasibility analysis).

Version B needs green that is:
  - SMOOTH & PERIODIC  -> forecastable; the agent can anticipate the next green window;
  - DEEP TROUGH (~0W)  -> running during the trough is all-brown, so timing genuinely matters;
  - SYNCHRONIZED across DCs (same phase) -> during the system-wide trough there is NO green DC
    to route to, so the ONLY carbon lever is to DEFER work into the next peak (the temporal story);
  - PEAK sized vs compute headroom so a concentrated deferred backlog can actually run on green;
  - trough->peak gap < deadline slack, so a deferred job can wait for the peak.

This writes per-turbine CSVs matching the provider's expected format/location
(<base>/Turbine_<id>_2021.csv, columns: timestamp,power_kw). With compressed_power_divisor=1000
the provider computes green_W = power_kw * 1000 / 1000 = (the value in the column), i.e. the
column holds WATTS directly (kept under the 'power_kw' header the reader expects).

COMPRESSED time scaling => 1 CSV row = 1 simulation second.
"""
import argparse
import math
import os

# DC -> turbine ids (must match the config's turbine_ids for the green DCs)
DC_TURBINES = {0: [12, 36], 1: [95, 91], 2: [96]}

# Per-DC PEAK green watts. Chosen ~ proportional to per-DC max compute draw
# (DC0 2140W, DC1 1712W, DC2 2150W) but scaled so total green MEAN ~= steady demand
# (~700W measured) => green is BINDING (not over-supplied) yet day-windows can absorb a
# deferred backlog. Tune these from the oracle's measured green-capture frontier.
DEFAULT_DC_PEAK_W = {0: 800.0, 1: 620.0, 2: 800.0}


def diurnal_watts(t, period, peak, duty=0.5, phase=0.0):
    """Clipped raised-sine: peak*max(0, sin(...)) over a `duty` fraction 'day', else 0.

    duty=0.5 => green 'on' for half the period (day), exactly 0W the other half (night).
    Trough is a true 0 so running at night is unambiguously all-brown.
    """
    # position in cycle [0,1)
    x = ((t / period) - phase) % 1.0
    if x >= duty:
        return 0.0
    # half-sine across the day window: 0 at edges, peak at the middle
    return peak * math.sin(math.pi * (x / duty))


def build_turbine_series(peak_w, n_rows, period, duty, phase):
    return [round(diurnal_watts(t, period, peak_w, duty, phase), 3) for t in range(n_rows)]


def write_csv(path, series, start_epoch_min=0):
    # timestamps are cosmetic in COMPRESSED mode (row index drives time); keep valid format.
    lines = ["timestamp,power_kw"]
    for i, v in enumerate(series):
        total_min = start_epoch_min + i * 10  # 10-min cadence like the original SDWPF files
        hh = (total_min // 60) % 24
        dd = 1 + (total_min // 60) // 24
        mm = total_min % 60
        lines.append(f"2021-01-{dd:02d} {hh:02d}:{mm:02d}:00,{v}")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(__file__), "..",
        "cloudsimplus-gateway/src/main/resources/windProduction/synthetic_diurnal"))
    # COMPRESSED mode skips the first 12 CSV rows and indexes by row thereafter (1 row = 1 sim-sec).
    # Size to cover the full deadline horizon (~9600s) + the 12-row skip + forecast look-ahead,
    # so no cyclic wrap happens mid-episode. 12012 = 12 (skip) + 5*2400 (5 clean periods).
    ap.add_argument("--rows", type=int, default=12012, help="rows = sim-seconds (COMPRESSED); covers full horizon")
    ap.add_argument("--period", type=int, default=2400, help="diurnal period in sim-seconds")
    ap.add_argument("--duty", type=float, default=0.5, help="fraction of period that is 'day'")
    ap.add_argument("--phase", type=float, default=0.0, help="global phase (all DCs share it -> synchronized)")
    args = ap.parse_args()

    assert (args.rows - 12) % args.period == 0, "rows-12 (post-skip data) should be an integer number of periods"
    os.makedirs(args.out, exist_ok=True)

    for dc, turbs in DC_TURBINES.items():
        dc_peak = DEFAULT_DC_PEAK_W[dc]
        per_turb_peak = dc_peak / len(turbs)
        for tid in turbs:
            series = build_turbine_series(per_turb_peak, args.rows, args.period, args.duty, args.phase)
            write_csv(os.path.join(args.out, f"Turbine_{tid}_2021.csv"), series)
    print(f"wrote synthetic diurnal turbines to {os.path.abspath(args.out)}")
    print(f"  rows={args.rows} period={args.period}s duty={args.duty} phase={args.phase} (DCs SYNCHRONIZED)")
    print(f"  trough->peak gap = {args.period*args.duty/2:.0f}s ; per-DC peak W = {DEFAULT_DC_PEAK_W}")


if __name__ == "__main__":
    main()
