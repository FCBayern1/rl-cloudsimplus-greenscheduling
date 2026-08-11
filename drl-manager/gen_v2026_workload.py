#!/usr/bin/env python3
"""Derive a LONG-job CloudSim workload from the Alibaba v2026 GPU cluster trace.

WHY: forecast value in scheduling needs a long commitment window (a job, once
routed, runs L steps there; only if L spans green transitions does the forecast
beat current-green). Our old traces had ~2-step jobs -> forecast worthless. The
v2026 job_execution_summary has real AI-training durations (LP median ~1220s,
p90 ~11600s) -> genuinely long jobs.

DERIVATION (see --help for knobs):
  - source: v2026 duration_hours, filtered to priority_class == LP (deferrable).
  - time scale S sec/step: length_steps = clip(round(dur_sec / S), 1, Lmax).
    S=20 -> median ~61 steps (analytic carbon-weighted headroom ~44% at spread3k).
  - MI = length_steps * ref_mips (pes=1); with ref_mips == vm_pe_mips the sim
    runtime equals length_steps exactly.
  - arrival ~ uniform(0, sim_duration - length_steps): every job finishes in-episode.
  - N is the LOAD knob, calibrated empirically in the sim (D/green~0.5), NOT here.
Verifies: prints the resulting length-in-steps distribution to check it matches
the source LP distribution (scaled/capped).
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import pyarrow.parquet as pq


def load_lp_durations_sec(parquet):
    t = pq.read_table(parquet, columns=["duration_hours", "priority_class"])
    d = t.column("duration_hours").to_numpy(zero_copy_only=False).astype(float)
    pc = t.column("priority_class").to_numpy(zero_copy_only=False)
    d = d[(pc == "LP") & np.isfinite(d) & (d > 0)]
    return d * 3600.0


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--parquet", default="../data/v2026_gpu/asi_opensource_job_execution_summary/part-000.parquet")
    p.add_argument("--out", required=True, help="output trace CSV (under traces/)")
    p.add_argument("--n", type=int, default=2000, help="job count (LOAD knob, smoke-calibrated)")
    p.add_argument("--time-scale", type=float, default=20.0, help="sec per sim step")
    p.add_argument("--max-length-steps", type=int, default=1000, help="cap job length in steps")
    p.add_argument("--sim-duration", type=int, default=7200, help="episode length in steps")
    p.add_argument("--ref-mips", type=int, default=40000, help="= vm_pe_mips, so runtime==length_steps")
    p.add_argument("--deadline-slack", type=int, default=1200,
                   help="deadline = arrival + length_steps + slack. REQUIRED for the "
                        "defer deadline-backstop (deadline<=0 rows are dropped from the "
                        "deadline map in Java -> unbounded deferral -> starvation risk). "
                        "1200 matches the dc8_light convention.")
    p.add_argument("--deadline-slack-max", type=int, default=0,
                   help="if >0, per-job slack ~ U[deadline-slack, this] and the arrival "
                        "window is coupled (arrival <= sim_duration - L - slack) so every "
                        "job's LATEST start lies inside the episode - the temporal-gamble "
                        "design (design_temporal_gamble.py PASS-STABLE cell).")
    p.add_argument("--min-duration-sec", type=float, default=0.0,
                   help="drop source LP jobs shorter than this before sampling. The v3 "
                        "lever for making job length comparable to the green-peak duration, "
                        "so that WHICH peak a job fits becomes the binding question "
                        "(scan_voi_v3.py Lmin axis).")
    p.add_argument("--seed", type=int, default=42)
    a = p.parse_args()

    rng = np.random.default_rng(a.seed)
    dur = load_lp_durations_sec(a.parquet)
    if a.min_duration_sec > 0:
        n_before = len(dur)
        dur = dur[dur >= a.min_duration_sec]
        print(f"min-duration filter {a.min_duration_sec:.0f}s: {n_before} -> {len(dur)} source jobs")
    print(f"source LP durations (sec): n={len(dur)} p50={np.percentile(dur,50):.0f} "
          f"p90={np.percentile(dur,90):.0f} p99={np.percentile(dur,99):.0f}")

    sample = rng.choice(dur, size=a.n)
    L = np.clip(np.round(sample / a.time_scale).astype(int), 1, a.max_length_steps)  # steps
    MI = L * a.ref_mips
    if a.deadline_slack_max > 0:
        slack = rng.integers(a.deadline_slack, a.deadline_slack_max + 1, a.n)
        # couple the arrival window so the latest start (deadline - L) is in-episode
        hi = np.maximum(1, a.sim_duration - L - slack)
    else:
        slack = np.full(a.n, a.deadline_slack)
        # arrival uniform in [0, sim_duration - L] so every job finishes in-episode
        hi = np.maximum(1, a.sim_duration - L)
    arrival = (rng.random(a.n) * hi).astype(int)

    order = np.argsort(arrival)
    rows = []
    for cid, i in enumerate(order):
        length = int(MI[i]); pes = 1
        fs = max(100, length // 1000); os_ = max(50, fs // 2)
        # deadline: absolute sim-time, feasible by construction (covers own runtime)
        ddl = int(arrival[i]) + int(L[i]) + int(slack[i])
        rows.append([cid, int(arrival[i]), length, pes, fs, os_, ddl])

    outp = Path("../cloudsimplus-gateway/src/main/resources") / a.out
    outp.parent.mkdir(parents=True, exist_ok=True)
    with open(outp, "w", newline="") as f:
        w = csv.writer(f); w.writerow(["cloudlet_id","arrival_time","length","pes_required","file_size","output_size","deadline"])
        w.writerows(rows)

    # ---- verification ----
    print(f"\nwrote {outp}  ({a.n} jobs, time_scale={a.time_scale}s/step, cap={a.max_length_steps} steps)")
    print(f"length in STEPS  p10/p50/p90/p99/max: "
          f"{[int(np.percentile(L,q)) for q in (10,50,90,99,100)]}")
    print(f"length in MI     p50/p99/max: {int(np.percentile(MI,50))}/{int(np.percentile(MI,99))}/{int(MI.max())}")
    print(f"  -> max MI {MI.max():,} ; set obs_cloudlet_mi_high >= this")
    print(f"arrival span: {arrival.min()}..{arrival.max()} (episode {a.sim_duration})")
    print(f"total PE-steps (sum L): {int(L.sum()):,}  mean concurrent jobs ~ {L.sum()/a.sim_duration:.1f}")


if __name__ == "__main__":
    main()
