#!/usr/bin/env python3
"""Offline horizon-coverage scan for obs_v32_forecast_horizon_steps.

Freezes the forecast span by EVIDENCE (eighth-review demand) instead of
argument: for each candidate horizon H, at every job arrival (sampled across
closed-book offsets), compare the H-truncated wait decision and captured gain
against the full-slack reference.

Metrics per H:
  nonzero_gain : fraction of jobs whose H-window offers any improvement
  agreement    : defer/route decision agreement with the full-slack planner
  gain_capture : mean(gain_H) / mean(gain_full) - how much of the teacher's
                 achievable improvement the truncated window still sees
"""
import csv, sys
from pathlib import Path
import numpy as np

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources"
import yaml
cfg = yaml.safe_load(open(REPO / "config_C.yml"))
exp = dict(cfg["common"]); exp.update(cfg["experiment_v3_2_oracle"])
div = float(exp["compressed_power_divisor"])
tot = None
for d in exp["datacenters"]:
    for t in d.get("turbine_ids") or []:
        v = np.array([float(r["power_kw"]) for r in
                      csv.DictReader(open(GATE / f"windProduction/simplified/Turbine_{t}_2021.csv"))])
        tot = v if tot is None else tot + v
green = tot * 1000.0 / div
tr = list(csv.DictReader(open(GATE / "traces" / Path(exp["cloudlet_trace_file"]).name)))
arr = np.array([int(r["arrival_time"]) for r in tr])
dl = np.array([int(r["deadline"]) for r in tr])
rt = np.array([max(1, round(float(r["length"]) / 40000)) for r in tr])
slack = np.maximum(0, dl - arr - rt - 120)          # wait budget (margin 120)
THETA = 0.5
offsets = [(1009 * k) % 4800 for k in range(0, 64, 8)]   # 8 closed-book offsets
HS = [120, 300, 600, 1200, 3000, 3600]

def binned_max(series, t0, span, bins):
    if span <= 0: return series[t0]
    offs = np.unique(np.rint(np.linspace(1, max(1, span), bins)).astype(int))
    idx = np.minimum(t0 + offs, len(series) - 1)
    return series[idx].max(initial=series[t0])

rows = []
for H, B in [(120,16),(300,16),(600,16),(1200,16),(3000,16),(3600,16),(3000,20),(3600,20)]:
    agree = nz = 0; gH_sum = gF_sum = 0.0; n = 0
    for off in offsets:
        base = 13 + off
        for a, s in zip(arr, slack):
            if s <= 0:
                continue
            t0 = base + a
            gnow = green[t0]
            bF = green[t0:t0 + int(s)].max(initial=gnow)      # full-res reference
            bH = binned_max(green, t0, int(min(s, H)), B)     # what the FEATURE sees
            dF, dH = gnow < THETA * bF, gnow < THETA * bH
            agree += (dF == dH); nz += (bH > gnow * 1.001)
            gF_sum += max(0.0, bF - gnow); gH_sum += max(0.0, bH - gnow)
            n += 1
    rows.append((H, B, nz / n, agree / n, gH_sum / max(1e-9, gF_sum)))
print(f"{'H(s)':>6} {'bins':>5} {'nonzero_gain':>13} {'agreement':>10} {'gain_capture':>13}")
for H, B, nzf, ag, gc in rows:
    print(f"{H:>6} {B:>5} {nzf:>12.1%} {ag:>10.1%} {gc:>13.1%}")
