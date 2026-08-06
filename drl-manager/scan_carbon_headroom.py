#!/usr/bin/env python3
"""Analytic CARBON headroom scan (pure numpy, no sim, no GPU).

Question: is there any (green scarcity, phase offsets, job length, load) where a
FUTURE-aware router (oracle) burns much less BROWN than a current-green router
(greedy)? Green-capture headroom (the old metric) does NOT convert to carbon
unless green is scarce enough to force brown -- so here we measure brown/carbon
directly, using the verified CSV->sim green equation:
    green_i(t) = sum_turbines( CSV_kW[WARM + off_i + t] ) / DIV

Model (spatial forecast lever): each arriving batch of size D commits to ONE DC
and runs L steps there.
  greedy : pick the DC minimising CURRENT-step carbon  max(0, D-green_i(t))*bcf_i
  oracle : pick the DC minimising RUN-WINDOW carbon over [t, t+L)
Both are charged the TRUE run-window carbon at their chosen DC.
  headroom = (carbon_greedy - carbon_oracle) / carbon_greedy
A candidate is interesting only if greedy actually burns brown (scarcity binds)
AND demand does not chronically exceed total green (else it is overload, not a
forecast story).
"""
import csv, itertools
from pathlib import Path
import numpy as np

GATE = Path(__file__).resolve().parent.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
WARM = 13
# green DCs: name -> (turbine ids, brown carbon factor)
GREEN = {"Nordic": ([7012, 7036], 0.08), "Germany": ([7095, 7091], 0.35),
         "US_East": ([7096], 0.55), "Nordic2": ([7101, 7103], 0.28)}
T = 7000  # episode steps to score

def turb(tid):
    rows = list(csv.DictReader(open(GATE / f"Turbine_{tid}_2021.csv")))
    return np.array([float(r["power_kw"]) for r in rows])

RAW = {k: sum(turb(t) for t in ids) for k, (ids, _) in GREEN.items()}
BCF = np.array([GREEN[k][1] for k in GREEN])
NAMES = list(GREEN)

def build_green(offs, DIV):
    return np.stack([RAW[k][WARM + offs[i]: WARM + offs[i] + T] / DIV for i, k in enumerate(NAMES)], 1)  # T x nDC

def score(green, D, L, stride=25):
    nD = green.shape[1]
    cg = co = 0.0
    n = 0
    for t in range(0, T - L, stride):
        win = green[t:t + L]                       # L x nD  true future
        run_carbon = (np.maximum(0.0, D - win).sum(0)) * BCF   # per-DC true run-window carbon
        # greedy: choose DC by CURRENT step only
        cur_carbon_now = np.maximum(0.0, D - green[t]) * BCF
        g = int(cur_carbon_now.argmin())
        # oracle: choose DC by true run-window carbon
        o = int(run_carbon.argmin())
        cg += run_carbon[g]; co += run_carbon[o]; n += 1
    tot_green_mean = green.sum(1).mean()
    return cg / n, co / n, tot_green_mean

def main():
    OFFSETS = {"aligned": [0, 18, 54, 36], "antiphase": [0, 0, 1000, 100],
               "spread1k": [0, 350, 700, 1050], "spread2k": [0, 700, 1400, 2100],
               "spread3k": [0, 1000, 2000, 3000], "spread4k": [0, 1400, 2800, 4200]}
    DIVS = [1500.0, 3000.0, 4500.0, 6000.0]
    LS = [50, 150, 400, 700]
    print(f"{'offsets':10s} {'DIV':>6s} {'L':>4s} {'D':>7s} {'greedyC':>8s} {'oracleC':>8s} "
          f"{'headroom':>9s} {'green_mn':>8s} {'D/green':>7s}")
    rows = []
    for oname, offs in OFFSETS.items():
        for DIV in DIVS:
            g = build_green(offs, DIV)
            gmean = g.sum(1).mean()
            for D in [0.5 * gmean, 1.0 * gmean, 1.5 * gmean]:
                for L in LS:
                    cg, co, gm = score(g, D, L)
                    hr = (cg - co) / cg if cg > 1e-9 else 0.0
                    rows.append((hr, oname, DIV, L, D, cg, co, gm, D / gm))
    rows.sort(reverse=True)
    for hr, oname, DIV, L, D, cg, co, gm, dr in rows[:15]:
        print(f"{oname:10s} {DIV:6.0f} {L:4d} {D:7.4f} {cg:8.4f} {co:8.4f} "
              f"{100*hr:8.1f}% {gm:8.4f} {dr:7.2f}")
    print("\nInterpretation: want high headroom AND greedyC clearly > 0 (scarcity binds) "
          "AND D/green not >>1 (not pure overload). If nothing clears ~20% headroom with "
          "greedyC>0 at D/green<=1, the spatial forecast lever has little carbon to give in this data.")

if __name__ == "__main__":
    main()
