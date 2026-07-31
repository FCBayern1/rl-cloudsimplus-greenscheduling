#!/usr/bin/env python3
"""Generate SQUARE-WAVE deep-trough green CSVs to make the forecast TRULY
load-bearing for the temporal (defer) decision.

Why real wind fails: smooth green ramps, so the current-green slope already
reveals an approaching trough -> forecast is redundant (godeye ~= noforecast).

This regime instead uses SYNCHRONIZED square troughs of RANDOM duration:
  - green is binary: FULL (surplus) during "on", EXACTLY 0 during a trough
  - all green DCs trough together -> no spatial escape, only the temporal lever
  - trough duration is random and straddles the forecast horizon, so:
      * short trough (ends within horizon)  -> future-mean > 0 -> "wait, green coming"
      * long  trough (outlasts the horizon) -> future-mean ~ 0 -> "don't wait, run brown"
  - current-green is 0 in BOTH cases and CANNOT distinguish them; ONLY the
    forecast can. => forecast carries information the current state does not.

COMPRESSED mode: 1 CSV row = 1 sim-second; gateway skips the first `skip` rows.
power_W = power_kw*1000 / compressed_power_divisor(1500).
"""
import argparse, os, random

# NEW turbine IDs (90xx) to AVOID overwriting the real-wind CSVs (12/36/95/91/96)
# that rwdefer and other regimes depend on. A squaretrough regime config must
# reference these 90xx ids in its datacenters' turbine_ids.
# id -> (dc_index, peak_kw) ; peak calibrated so ON-period green ~ amp x demand
TURBINES = {9012: (0, 1605.0), 9036: (0, 1605.0), 9095: (1, 1284.0), 9091: (1, 1284.0), 9096: (2, 3225.0)}


def build_schedule(n_rows, skip, on_dur, trough_lo, trough_hi, seed):
    """Return a list[bool] of length n_rows: True = green ON, False = trough.
    Synchronized across all DCs; trough durations random in [lo, hi]."""
    rng = random.Random(seed)
    on = [False] * n_rows
    r = skip
    green_now = True  # start in a green window
    while r < n_rows:
        if green_now:
            dur = on_dur
        else:
            dur = rng.randint(trough_lo, trough_hi)
        for j in range(r, min(r + dur, n_rows)):
            on[j] = green_now
        r += dur
        green_now = not green_now
    return on


def gen(out_dir, n_rows, skip, on_dur, trough_lo, trough_hi, amp_scale, seed, year=2021):
    os.makedirs(out_dir, exist_ok=True)
    sched = build_schedule(n_rows, skip, on_dur, trough_lo, trough_hi, seed)
    n_on = sum(sched)
    for tid, (dc, peak_kw) in TURBINES.items():
        peak = peak_kw * amp_scale
        rows = []
        for r in range(n_rows):
            p = peak if sched[r] else 0.0
            rows.append(f"2021-01-01 00:00:00,{p:.3f}")
        path = os.path.join(out_dir, f"Turbine_{tid}_{year}.csv")
        with open(path, "w") as f:
            f.write("timestamp,power_kw\n")
            f.write("\n".join(rows) + "\n")
        print(f"  Turbine_{tid} (DC{dc}): peak={peak:.0f}kW -> {path}")
    troughs = []
    r = skip; g = True
    rng = random.Random(seed)
    while r < n_rows:
        d = on_dur if g else rng.randint(trough_lo, trough_hi)
        if not g: troughs.append(d)
        r += d; g = not g
    print(f"schedule: {n_on}/{n_rows} rows green ({100*n_on/n_rows:.0f}%); "
          f"{len(troughs)} troughs, dur range [{trough_lo},{trough_hi}] "
          f"(median {sorted(troughs)[len(troughs)//2] if troughs else 0})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="../cloudsimplus-gateway/src/main/resources/windProduction/simplified")
    ap.add_argument("--rows", type=int, default=7400)
    ap.add_argument("--skip", type=int, default=12)
    ap.add_argument("--on-dur", type=int, default=120, help="green window length (steps)")
    ap.add_argument("--trough-lo", type=int, default=40, help="min trough length (< forecast horizon)")
    ap.add_argument("--trough-hi", type=int, default=480, help="max trough length (>> horizon)")
    ap.add_argument("--amp-scale", type=float, default=2.0, help="ON-period green as multiple of demand")
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    print(f"Square-trough green: on={a.on_dur} trough=[{a.trough_lo},{a.trough_hi}] amp={a.amp_scale} seed={a.seed}")
    gen(a.out, a.rows, a.skip, a.on_dur, a.trough_lo, a.trough_hi, a.amp_scale, a.seed)
    print("DONE. Rebuild the gateway jar (installDist) so the sim picks these up.")
