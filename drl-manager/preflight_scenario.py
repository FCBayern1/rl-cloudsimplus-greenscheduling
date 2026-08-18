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


def sqt2_trough_exposure(arrivals, runtimes, deadlines, mi, troughs,
                         offsets, margin=120.0, warmup=13, kwargs_tight=None):
    """SQT2 decision-exposure (Codex adjudication 2026-08-18): among jobs that
    ARRIVE INSIDE a trough (the population that actually faces the wait
    decision), classify wait-worthy vs not-worth using the REMAINING trough
    length at arrival (residual-to-ON), never the total trough length:
        worthy iff (trough_end - arrival_row) <= deadline-arrival-runtime-margin
    MI-weighted shares aggregated over the pre-registered anchor offsets.
    Also returns the all-jobs split (ON arrivals counted not-worth) as a
    reported secondary view."""
    iv = [(t["start"], t["start"] + t["dur"]) for t in troughs]
    tight_flags = kwargs_tight if kwargs_tight is not None else [False] * len(mi)
    cell = {("tight", True): 0.0, ("tight", False): 0.0,
            ("loose", True): 0.0, ("loose", False): 0.0}
    on_mi = 0.0
    hit_troughs = set()
    for off in offsets:
        for a, rt, dl, m, tg in zip(arrivals, runtimes, deadlines, mi, tight_flags):
            row = warmup + off + int(a)
            hit = next(((s, e) for (s, e) in iv if s <= row < e), None)
            if hit is None:
                on_mi += m
                continue
            hit_troughs.add(hit[0])
            residual = hit[1] - row
            budget = dl - a - rt - margin      # == latest-start backstop budget
            worthy = budget > 0 and residual <= budget
            cell[("tight" if tg else "loose", worthy)] += m
    worthy_mi = cell[("tight", True)] + cell[("loose", True)]
    notworth_mi = cell[("tight", False)] + cell[("loose", False)]
    trough_tot = worthy_mi + notworth_mi
    tight_tot = cell[("tight", True)] + cell[("tight", False)]
    return {"worthy_share": worthy_mi / trough_tot if trough_tot else 0.0,
            "notworth_share": notworth_mi / trough_tot if trough_tot else 0.0,
            "trough_arrival_mi_frac": trough_tot / max(1e-9, trough_tot + on_mi),
            "p_worthy_given_tight": (cell[("tight", True)] / tight_tot
                                     if tight_tot else float("nan")),
            "cell_mi": {f"{c}_{'worthy' if w else 'notworth'}": v
                        for (c, w), v in cell.items()},
            "distinct_troughs_hit": len(hit_troughs)}


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
    profile_o = str(O.get("preflight_temporal_profile") or "").strip()
    profile_n = str(N.get("preflight_temporal_profile") or "").strip()
    sqt2 = profile_o.startswith("sqt2_trough_") or ("--sqt2-cert" in sys.argv)

    def na(name, msg):
        print(f"[ N/A ] {name:30s} {msg}")

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
                "obs_v31_features",
                # V3.2 switches — same rule: recipe, not treatment
                "obs_v32_job_forecast", "obs_v32_forecast_bin_count",
                "obs_v32_forecast_horizon_steps",
                "per_action_spatial_center", "per_action_spatial_weight",
                "per_action_spatial_sigma")
    asym = [k for k in v31_keys if O.get(k) != N.get(k)]
    chk("v3.1 switches symmetric", not asym, f"asymmetric={asym or 'none'}")

    # The core geometry: a job must not fit inside one peak (else any peak will
    # do), yet must fit inside one peak-to-peak cycle (else every start time
    # averages over the same peaks and troughs and timing stops mattering).
    med = float(np.median(L))
    slack = dl - arr - L
    if sqt2:
        # v3-lever premises (job-spans-peak commitment) do not map to the
        # SQT2 wait-across-trough lever; marked N/A per Codex adjudication,
        # replaced by the sqt2_trough_v1 block below. NOT deleted globally.
        na("job longer than a peak", f"{profile_o}: commitment = wait span, not job span")
        na("job shorter than a cycle", profile_o)
        na("slack near one cycle", profile_o)
    else:
        chk("job longer than a peak", med > pk, f"L median={med:.0f} vs peak={pk:.0f} ({med/pk:.2f}x)")
        chk("job shorter than a cycle", med < 0.6 * cycle,
            f"L median={med:.0f} vs cycle={cycle:.0f} ({med/cycle:.2f} of a cycle)")
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
    if sqt2:
        na("short horizon covers peak", f"{profile_o}: v32 bins horizon is the decision channel")
    else:
        chk("short horizon covers peak", sh >= pk, f"short_term_rows={sh} vs peak={pk:.0f}")
    # V3.1: window_carbon_source REMOVED from the whitelist — allowing it to
    # differ is exactly the hole the two-yardstick bug walked through.
    diff = {k for k in set(O) | set(N) if O.get(k) != N.get(k)}
    allowed = {"forecast_mode", "green_oracle_mode", "experiment_name", "simulation_name"}
    chk("arms differ only as intended", diff <= allowed, f"diffs={sorted(diff)}")

    if sqt2:
        import json as _json
        chk("sqt2: profile symmetric", profile_o == profile_n == "sqt2_trough_v2",
            f"oracle={profile_o!r} blind={profile_n!r} (v2 required)")
        for key in ("defer_deadline_force_mode", "defer_deadline_slack_sec",
                    "obs_v32_demand_model", "obs_v32_deadline_margin_sec",
                    "cloudlet_cpu_utilization"):
            chk(f"sqt2: {key} symmetric", O.get(key) == N.get(key),
                f"oracle={O.get(key)} blind={N.get(key)}")
        chk("sqt2: latest-start backstop on",
            O.get("defer_deadline_force_mode") == "latest_start"
            and float(O.get("defer_deadline_slack_sec", 0)) == 120.0,
            f"mode={O.get('defer_deadline_force_mode')} slack={O.get('defer_deadline_slack_sec')}")
        chk("sqt2: obs margin == backstop slack (SQT2.2-Clean)",
            float(O.get("obs_v32_deadline_margin_sec", 0)) == 120.0,
            f"obs_v32_deadline_margin_sec={O.get('obs_v32_deadline_margin_sec')}")
        chk("sqt2: full CPU utilization (physics == registered maths)",
            float(O.get("cloudlet_cpu_utilization", 0.5)) == 1.0,
            f"cloudlet_cpu_utilization={O.get('cloudlet_cpu_utilization')}"
            " (0.5 stretches runtimes ~2.5x and voids every budget check)")
        trace_name = Path(N["cloudlet_trace_file"]).name
        prefix = trace_name.split("_n1200_")[0]        # sqt2 | sqt2ho
        sched_art = "calib/sqt2ho_schedule.json" if prefix == "sqt2ho" else "calib/sqt2_schedule.json"
        art = _json.loads((Path(__file__).resolve().parent / sched_art).read_text())
        tag = trace_name.replace(f"{prefix}_n1200_", "").replace(".csv", "")
        tr_art = _json.loads((Path(__file__).resolve().parent
                              / f"calib/{prefix}_trace_{tag}.json").read_text())
        tight_ids = set(tr_art["tight_cloudlet_ids"])
        tight_flags = [r["cloudlet_id"] in tight_ids for r in tr]
        short_max = art["off_short"][1]
        long_min, long_max = art["off_long"]
        on_min = art["on_range"][0]
        horizon = int(N.get("obs_v32_forecast_horizon_steps", 0))
        bins = int(N.get("obs_v32_forecast_bin_count", 1))
        # class-conditional structure on the EXECUTABLE wait budget
        # B = deadline - arrival - runtime - 120 (== latest-start backstop)
        B = dl - arr - L - 120.0
        Bt = B[np.asarray(tight_flags)]
        Bl = B[~np.asarray(tight_flags)]
        chk("sqt2v2: tight budget positive", float(np.percentile(Bt, 5)) > 0,
            f"B_p05(tight)={np.percentile(Bt, 5):.0f} > 0")
        chk("sqt2v2: tight cannot outwait short troughs",
            float(np.percentile(Bt, 95)) < short_max,
            f"B_p95(tight)={np.percentile(Bt, 95):.0f} < short_max={short_max}")
        chk("sqt2v2: loose budget between classes",
            short_max <= float(np.median(Bl)) < long_min,
            f"short_max={short_max} <= B_p50(loose)={np.median(Bl):.0f} < long_min={long_min}")
        anchors = [(1009 * k) % int(N["green_episode_offset_range"])
                   for k in (0, 20, 40, 59, 79, 99, 119, 138, 158, 178)]
        exp = sqt2_trough_exposure(arr, L, dl, MI, art["troughs"], anchors,
                                   kwargs_tight=tight_flags)
        chk("sqt2v2: decision exposure (MAIN)",
            0.35 <= exp["worthy_share"] <= 0.65,
            f"worthy={exp['worthy_share']:.2f} not-worth={exp['notworth_share']:.2f} "
            f"(trough-arrival MI frac={exp['trough_arrival_mi_frac']:.2f})")
        chk("sqt2v2: P(worthy|tight) in band",
            0.25 <= exp["p_worthy_given_tight"] <= 0.75,
            f"P(worthy|tight)={exp['p_worthy_given_tight']:.2f}")
        cells = {k: round(v / 1e9, 1) for k, v in exp["cell_mi"].items()}
        print(f"[info ] sqt2v2 exposure cells (GMI): {cells}")
        chk("sqt2v2: distinct trough coverage",
            exp["distinct_troughs_hit"] >= 8,
            f"{exp['distinct_troughs_hit']} distinct troughs across anchors")
        chk("sqt2: cashability", int(L.max()) <= on_min,
            f"runtime_max={int(L.max())} <= ON_min={on_min}")
        slack_p95 = float(np.percentile(slack, 95))
        chk("sqt2: horizon covers waitable+slack",
            horizon >= max(short_max, slack_p95),
            f"horizon={horizon} >= max(short_max={short_max}, slack_p95={slack_p95:.0f})")
        gap = horizon / max(1, bins - 1)
        chk("sqt2: bin resolution", gap < on_min,
            f"max bin gap={gap:.0f}s < ON_min={on_min}")
        longs = [tt for tt in art["troughs"] if tt["kind"] == "long"]
        beyond = sum(1 for tt in longs if tt["dur"] > horizon)
        chk("sqt2: long troughs beyond horizon",
            longs and beyond / len(longs) >= 0.2,
            f"{beyond}/{len(longs)} long troughs exceed horizon {horizon}")

    # V3.1 gate: when the new observation features are on, report the trace
    # stats the clipping design must cover (time_to_deadline goes NEGATIVE at
    # runtime; wait_age can reach the episode length — [0, p99] bounds are
    # wrong by construction, see docs/V31_WORK_ORDERS.md 工单2).
    if N.get("obs_v31_features"):
        sl = dl - arr - L
        print(f"[info ] obs_v31 trace stats: init-slack p99={np.percentile(sl,99):.0f}s "
              f"max={sl.max():.0f}s; wait_age can reach ~7200s; slack can go negative")

    # --v32-cert: V3.2 certification invariants (superset of --v31-cert).
    if "--v32-cert" in sys.argv:
        sys.argv.append("--v31-cert")  # inherit all v31 gates
        chk("cert: v32 job forecast on",
            O.get("obs_v32_job_forecast") is True and N.get("obs_v32_job_forecast") is True,
            f"oracle={O.get('obs_v32_job_forecast')} blind={N.get('obs_v32_job_forecast')}")
        chk("cert: spatial term calibrated",
            O.get("per_action_spatial_center") == "candidate_mean"
            and float(O.get("per_action_spatial_sigma", 1.0)) != 1.0,
            f"center={O.get('per_action_spatial_center')} sigma={O.get('per_action_spatial_sigma')}"
            " (sigma=1.0 means the sigma_spatial artifact was never applied)")
        # the gate flag lives in the model config block, not env config — the
        # runner asserts it separately; here we pin the observation halves.

    # --p02-cert: P0-2 SLA alignment (V32B_ANNEAL_SPEC R1 item 2). The
    # TRAINING constraint must equal the EVALUATION contract - the 600k FT
    # arms trained with deadline_miss+0.05 tolerance, lambda stayed 0.0 for
    # the whole run, and PPO legally traded completion for carbon.
    if "--p02-cert" in sys.argv:
        for tag, cfg_ in (("oracle", O), ("blind", N)):
            chk(f"p02: {tag} sla_mode is completion",
                cfg_.get("sla_mode") == "completion",
                f"sla_mode={cfg_.get('sla_mode')}")
            chk(f"p02: {tag} training target == 99.5% eval contract",
                abs(float(cfg_.get("sla_target", 0.0)) - 0.995) < 1e-9,
                f"sla_target={cfg_.get('sla_target')}")
            lag = cfg_.get("lagrangian") or {}
            chk(f"p02: {tag} c_ep_tolerance is zero",
                float(lag.get("c_ep_tolerance", 1.0)) == 0.0,
                f"c_ep_tolerance={lag.get('c_ep_tolerance')}")

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
