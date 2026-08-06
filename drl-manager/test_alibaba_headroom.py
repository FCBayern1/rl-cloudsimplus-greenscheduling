#!/usr/bin/env python3
"""THE decisive test: does the REAL Alibaba job-length distribution (with its
long tail, p90=201s p99=1043s) give a carbon-weighted forecast headroom?

The GPU stress test used dc8_light's distribution (all ~2 steps, tail truncated)
-> 0%. But our conversion CAPPED length at 2M MI (~50 steps), killing the tail.
Real Alibaba v2018 batch_task durations have a genuine long tail. Long jobs burn
MORE carbon (run longer), so a CARBON-WEIGHTED aggregate can be dominated by the
tail even if the median job is short.

Headroom = sum_jobs(carbon_greedy - carbon_oracle) / sum_jobs(carbon_greedy),
i.e. carbon-weighted, NOT mean of per-job ratios. 1 sim step = 1 second, so job
length in steps = real duration in seconds.
"""
import csv
from pathlib import Path
import numpy as np

GATE = Path(__file__).resolve().parent.parent / "cloudsimplus-gateway/src/main/resources/windProduction/simplified"
WARM = 13
GREEN = {"Nordic": ([7012, 7036], 0.08), "Germany": ([7095, 7091], 0.35),
         "US_East": ([7096], 0.55), "Nordic2": ([7101, 7103], 0.28)}
T = 7000
NAMES = list(GREEN)
BCF = np.array([GREEN[k][1] for k in GREEN])

def turb(tid):
    return np.array([float(r["power_kw"]) for r in csv.DictReader(open(GATE / f"Turbine_{tid}_2021.csv"))])
RAW = {k: sum(turb(t) for t in ids) for k, (ids, _) in GREEN.items()}

def build_green(offs, DIV):
    return np.stack([RAW[k][WARM + offs[i]: WARM + offs[i] + T] / DIV for i, k in enumerate(NAMES)], 1)

def load_alibaba_durations(n=500000):
    d = []
    with open(Path(__file__).resolve().parent.parent / "data/alibaba_v2018/batch_task.csv") as f:
        for i, row in enumerate(csv.reader(f)):
            if i >= n: break
            try:
                dur = float(row[6]) - float(row[5])
                if 1 <= dur < T: d.append(dur)
            except Exception: pass
    return np.array(d)

def weighted_headroom(green, durs, D, njobs, rng, cap=None):
    """Sample njobs jobs; each: random arrival t, length L~Alibaba, demand D/step.
    Carbon-weighted headroom (greedy=current-green choice, oracle=future-window)."""
    Cg = Co = 0.0
    Ls = rng.choice(durs, size=njobs)
    if cap: Ls = np.minimum(Ls, cap)
    for L in Ls:
        L = int(L)
        t = int(rng.integers(0, T - L - 1))
        win = green[t:t + L]                              # L x nDC true future
        run_carbon = np.maximum(0.0, D - win).sum(0) * BCF
        g = int((np.maximum(0.0, D - green[t]) * BCF).argmin())   # greedy: current step
        o = int(run_carbon.argmin())                              # oracle: full window
        Cg += run_carbon[g]; Co += run_carbon[o]
    return (Cg - Co) / Cg if Cg > 1e-9 else 0.0, Cg / njobs

def main():
    durs = load_alibaba_durations()
    print(f"Alibaba durations (steps=sec): n={len(durs)} p50={np.percentile(durs,50):.0f} "
          f"p90={np.percentile(durs,90):.0f} p99={np.percentile(durs,99):.0f} max={durs.max():.0f}")
    rng = np.random.default_rng(7)
    OFFSETS = {"aligned": [0, 18, 54, 36], "spread2k": [0, 700, 1400, 2100],
               "spread3k": [0, 1000, 2000, 3000]}
    DIV = 1500.0
    print(f"\n{'offsets':9s} {'cap(steps)':>10s} {'D/green':>7s} {'wtd_headroom':>13s}")
    for oname, offs in OFFSETS.items():
        g = build_green(offs, DIV); gmean = g.sum(1).mean()
        for cap in [50, 200, None]:   # 50=our current truncation, 200~p90, None=full real tail
            for dr in [0.5, 1.0]:
                hr, cavg = weighted_headroom(g, durs, dr * gmean, 4000, rng, cap=cap)
                capn = "full" if cap is None else str(cap)
                print(f"{oname:9s} {capn:>10s} {dr:7.1f} {100*hr:12.1f}%")
    print("\nRead: 'full' = real Alibaba tail (no truncation). If spread3k/full gives a large "
          "carbon-weighted headroom while cap=50 (our current trace) gives ~0, then the forecast "
          "value is REAL under a faithful long-job workload -- the tail we truncated was the signal.")

if __name__ == "__main__":
    main()
