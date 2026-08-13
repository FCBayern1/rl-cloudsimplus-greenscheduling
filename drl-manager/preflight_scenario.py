#!/usr/bin/env python3
"""Pre-flight audit for a forecast-value testbed. Run BEFORE spending GPU hours.

Every check here corresponds to a failure mode that actually killed one of the
six earlier testbeds. The rule learned the hard way: a scenario that fails any
check produces an uninterpretable result, because a negative outcome cannot be
attributed to the scenario rather than to the design gap.

Usage: preflight_scenario.py <oracle_experiment> <blind_experiment> [--v31-cert]

--v31-cert additionally enforces the V3.1 certification invariants (fixed
local drain + dispatch_rate). Template configs with switches still OFF pass
without the flag; the final cert configs must pass WITH it.
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
    # V3.1 gate (2026-08-13): the two-yardstick lesson. window_carbon_source
    # differing between arms made v3 a TWO-variable ablation (obs forecast AND
    # reward pricing) — an unconditional price discount the policy cannot act
    # on, which also made cross-arm returns unreadable. Same source, both arms.
    chk("arms share one carbon yardstick",
        O.get("window_carbon_source") == N.get("window_carbon_source"),
        f"oracle={O.get('window_carbon_source')} blind={N.get('window_carbon_source')}")
    # V3.1 gate: reward-surgery switches must be SYMMETRIC (they are recipe,
    # not treatment; the treatment is forecast_mode alone).
    v31_keys = ("per_action_completion_mode", "defer_cost_mode",
                "per_action_carbon_norm", "per_action_carbon_mu",
                "per_action_carbon_sigma", "fixed_local_scheduler",
                "obs_v31_features")
    asym = [k for k in v31_keys if O.get(k) != N.get(k)]
    chk("v3.1 switches symmetric", not asym, f"asymmetric={asym or 'none'}")

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
    # V3.1: window_carbon_source REMOVED from the whitelist — allowing it to
    # differ is exactly the hole the two-yardstick bug walked through.
    diff = {k for k in set(O) | set(N) if O.get(k) != N.get(k)}
    allowed = {"forecast_mode", "green_oracle_mode", "experiment_name", "simulation_name"}
    chk("arms differ only as intended", diff <= allowed, f"diffs={sorted(diff)}")

    # V3.1 gate: when the new observation features are on, report the trace
    # stats the clipping design must cover (time_to_deadline goes NEGATIVE at
    # runtime; wait_age can reach the episode length — [0, p99] bounds are
    # wrong by construction, see docs/V31_WORK_ORDERS.md 工单2).
    if N.get("obs_v31_features"):
        sl = dl - arr - L
        print(f"[info ] obs_v31 trace stats: init-slack p99={np.percentile(sl,99):.0f}s "
              f"max={sl.max():.0f}s; wait_age can reach ~7200s; slack can go negative")

    # --v31-cert: certification invariants (only for the FINAL cert configs).
    if "--v31-cert" in sys.argv:
        chk("cert: local drain fixed",
            O.get("fixed_local_scheduler") == "drain" and N.get("fixed_local_scheduler") == "drain",
            f"oracle={O.get('fixed_local_scheduler')} blind={N.get('fixed_local_scheduler')}")
        lm = (O.get("local_dispatch_mode", "vm_placement"), N.get("local_dispatch_mode", "vm_placement"))
        chk("cert: dispatch_rate mode", lm == ("dispatch_rate", "dispatch_rate"),
            f"local_dispatch_mode={lm} (drain override is only meaningful under dispatch_rate)")
        chk("cert: zscore has calibrated sigma",
            N.get("per_action_carbon_norm") != "centered_zscore"
            or float(N.get("per_action_carbon_sigma", 1.0)) != 1.0,
            "centered_zscore with sigma=1.0 means the calibration artifact was never applied")

    print()
    if fails:
        print(f"AUDIT FAILED ({len(fails)}): " + ", ".join(fails))
        sys.exit(1)
    print("AUDIT PASSED - scenario is interpretable, training may start")


if __name__ == "__main__":
    main()
