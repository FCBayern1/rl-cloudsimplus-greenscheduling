#!/usr/bin/env python3
"""Pre-flight audit for a forecast-value testbed. Run BEFORE spending GPU hours.

Every check here corresponds to a failure mode that actually killed one of the
six earlier testbeds. The rule learned the hard way: a scenario that fails any
check produces an uninterpretable result, because a negative outcome cannot be
attributed to the scenario rather than to the design gap.

Usage: preflight_scenario.py <oracle_experiment> <blind_experiment>
"""
import csv
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
GATE = REPO / "cloudsimplus-gateway/src/main/resources"


def load(name):
    return yaml.safe_load(open(REPO / "config_C.yml"))[name]


def peak_stats(blk):
    tot = None
    for dc in blk["datacenters"]:
        for t in dc["turbine_ids"]:
            v = np.array([float(r["power_kw"]) for r in
                          csv.DictReader(open(GATE / f"windProduction/simplified/Turbine_{t}_2021.csv"))])
            tot = v if tot is None else tot + v
    g = tot[13:13 + 7200] * 1000 / blk["compressed_power_divisor"]
    demand = g.mean() / 1.29           # rho by construction
    above = g >= demand
    runs, cur = [], 0
    for x in above:
        if x:
            cur += 1
        elif cur:
            runs.append(cur); cur = 0
    if cur:
        runs.append(cur)
    runs = np.array(runs if runs else [1])
    return float(np.median(runs)), len(runs), 7200.0 / max(1, len(runs)), len(tot)


def main():
    o_name, n_name = sys.argv[1], sys.argv[2]
    O, N = load(o_name), load(n_name)
    tr = list(csv.DictReader(open(GATE / "traces" / Path(N["cloudlet_trace_file"]).name)))
    L = np.array([max(1, round(float(r["length"]) / 40000)) for r in tr])
    MI = np.array([float(r["length"]) for r in tr])
    arr = np.array([int(r["arrival_time"]) for r in tr])
    dl = np.array([int(r["deadline"]) for r in tr])
    pk, npk, cycle, wrows = peak_stats(N)

    fails = []

    def chk(name, ok, msg):
        print(f"[{'PASS' if ok else '**FAIL**'}] {name:30s} {msg}")
        if not ok:
            fails.append(name)

    ob = N.get("obs_cloudlet_mi_high")
    chk("obs bound >= max MI", bool(ob) and ob >= MI.max(), f"bound={ob} vs max MI={MI.max():.0f}")
    chk("closed-book offsets on", N.get("green_episode_offset_range", 0) > 0,
        f"range={N.get('green_episode_offset_range')}")
    need = 13 + int(N.get("green_episode_offset_range", 0)) + 7200
    chk("wind rows cover offsets", wrows >= need, f"{wrows} rows, need {need}")
    tz = [d.get("time_zone_offset_rows", 0) for d in N["datacenters"] if d["turbine_ids"]]
    chk("green DCs synchronised", len(set(tz)) == 1, f"tz offsets={tz}")
    bf = [d["brown_carbon_factor"] for d in N["datacenters"] if d["turbine_ids"]]
    chk("no cheap-brown cushion", len(set(bf)) == 1 and min(bf) >= 0.5, f"green-DC brown={bf}")
    chk("blind reward has no leak", N.get("window_carbon_source") == "persistence",
        f"blind={N.get('window_carbon_source')}, oracle={O.get('window_carbon_source')}")

    # The core geometry: a job must not fit inside one peak (else any peak will
    # do), yet must fit inside one peak-to-peak cycle (else every start time
    # averages over the same peaks and troughs and timing stops mattering).
    med = float(np.median(L))
    chk("job longer than a peak", med > pk, f"L median={med:.0f} vs peak={pk:.0f} ({med/pk:.2f}x)")
    chk("job shorter than a cycle", med < 0.6 * cycle,
        f"L median={med:.0f} vs cycle={cycle:.0f} ({med/cycle:.2f} of a cycle)")

    slack = dl - arr - L
    chk("slack near one cycle", float(np.median(slack)) / cycle <= 2.5,
        f"slack median={np.median(slack):.0f} = {np.median(slack)/cycle:.1f} cycles")
    conc = np.zeros(8300)
    for a, l in zip(arr, L):
        conc[a:a + l] += 1
    gv = sum(d.get("initial_s_vm_count", 0) + d.get("initial_m_vm_count", 0) + d.get("initial_l_vm_count", 0)
             for d in N["datacenters"] if d["turbine_ids"])
    chk("green capacity > peak load", conc.max() < gv, f"peak concurrency={conc.max():.0f} vs green VMs={gv}")
    chk("deadlines land in-episode", bool((dl - L < 7200).all()), f"max latest start={int((dl-L).max())}")
    sh = N["datacenters"][0].get("short_term_rows", 0)
    chk("short horizon covers peak", sh >= pk, f"short_term_rows={sh} vs peak={pk:.0f}")
    diff = {k for k in set(O) | set(N) if O.get(k) != N.get(k)}
    allowed = {"forecast_mode", "window_carbon_source", "green_oracle_mode", "experiment_name"}
    chk("arms differ only as intended", diff <= allowed, f"diffs={sorted(diff)}")

    print()
    if fails:
        print(f"AUDIT FAILED ({len(fails)}): " + ", ".join(fails))
        sys.exit(1)
    print("AUDIT PASSED - scenario is interpretable, training may start")


if __name__ == "__main__":
    main()
