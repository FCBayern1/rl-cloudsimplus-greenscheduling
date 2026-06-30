#!/usr/bin/env python3
"""Generate SHARP ANTI-PHASE solar green CSVs for the 5-DC carbon_v2 regime, to make the
forecast LOAD-BEARING (current-green != future-green; greenest DC rotates over time).

COMPRESSED mode: 1 CSV row = 1 sim-second; the gateway skips the first 12 rows, so row 12 -> t=0.
Only DC0/DC1/DC2 have turbines: DC0=[12,36], DC1=[95,91], DC2=[96]. We give each DC a distinct
phase so the green peak rotates across DCs; peaks are sharp (sin^k) so windows are narrow.
power_W = power_kw*1000 / compressed_power_divisor(1500); calibrate peak_kw so green~=demand.
"""
import argparse, math, os

# turbine_id -> (dc_index, peak_kw)  [peak_kw set so sum over a DC's turbines ~= DC peak demand W]
#   DC0 demand 2140W /2 turb -> 1605 each ; DC1 1712W/2 -> 1284 ; DC2 2150W/1 -> 3225
TURBINES = {12: (0, 1605.0), 36: (0, 1605.0), 95: (1, 1284.0), 91: (1, 1284.0), 96: (2, 3225.0)}
N_GREEN_DC = 3  # DC0,1,2

def gen(out_dir, T, k, n_rows, skip=12, year=2021, amp_scale=1.0):
    os.makedirs(out_dir, exist_ok=True)
    for tid, (dc, peak_kw) in TURBINES.items():
        phase = dc * T / N_GREEN_DC            # DC0=0, DC1=T/3, DC2=2T/3  -> anti-phase
        peak = peak_kw * amp_scale
        rows = []
        for r in range(n_rows):
            if r < skip:
                p = 0.0
            else:
                t = r - skip                    # sim-second
                s = math.sin(2.0 * math.pi * (t - phase) / T)
                p = peak * (max(0.0, s) ** k)
            # cosmetic timestamp (ignored under COMPRESSED)
            rows.append(f"2021-01-01 00:00:00,{p:.3f}")
        path = os.path.join(out_dir, f"Turbine_{tid}_{year}.csv")
        with open(path, "w") as f:
            f.write("timestamp,power_kw\n")
            f.write("\n".join(rows) + "\n")
        print(f"  Turbine_{tid} (DC{dc}): peak={peak:.0f}kW phase={phase:.0f}s -> {path}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../cloudsimplus-gateway/src/main/resources/windProduction/simplified")
    ap.add_argument("--period", type=float, default=2400.0, help="solar period in sim-seconds (cycles/episode = 7200/T)")
    ap.add_argument("--sharp", type=float, default=4.0, help="sin^k sharpness (higher=narrower peaks)")
    ap.add_argument("--rows", type=int, default=7400)
    ap.add_argument("--amp-scale", type=float, default=1.0)
    a = ap.parse_args()
    print(f"Generating sharp anti-phase solar: T={a.period}s ({7200/a.period:.1f} cycles/ep), k={a.sharp}, rows={a.rows}")
    gen(a.out, a.period, a.sharp, a.rows, amp_scale=a.amp_scale)
    print("DONE. Rebuild the gateway jar so the sim picks these up.")
