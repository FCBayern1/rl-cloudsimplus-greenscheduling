#!/usr/bin/env python3
"""SQT2 formal capacity calibration (docs/SQT2_SCENARIO_SPEC.md line ~108).

green_executable_MI = sum over ON rows, over green DCs, of
    min(G_d / W_per_PE_eff_d, PE_d) * VM_MIPS * 1s
with W_per_PE_eff = idle-per-PE share + dynamic per-PE watts from the
HostProfile SPEC constants (RS500A: 64 PEs, 214W peak, 24.0% idle;
RS700A: 128 PEs, 430W peak, 24.7% idle). The shipping requirement is
green_executable_MI >= 1.2 x total job MI - checked BOTH over the whole
series scaled to one episode and, stricter, at every anchor's 7200 s
decision window. Emits calib/{sqt2,sqt2ho}_capacity_audit.json.
"""
import argparse
import csv
import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from oracle_slack_planner import WARMUP_ROWS  # noqa: E402
from teacher_reward_audit import episode_offset  # noqa: E402
from gen_sqt2 import H_D_W  # noqa: E402
from sqt2_prescreen import ANCHORS, HORIZON_S  # noqa: E402

VM_MIPS = 40000.0
RATIO_MIN = 1.2
# HostProfile SPEC constants: pes, peak W, idle percent
PROFILES = {"rs500a": (64, 214.0, 24.0), "rs700a": (128, 430.0, 24.7)}


def w_per_pe_eff(profile: str) -> float:
    pes, peak, idle_pct = PROFILES[profile]
    idle = peak * idle_pct / 100.0
    return idle / pes + (peak - idle) / pes


def dc_profile(dc_cfg: dict):
    """(profile, host_count) from a config datacenter dict."""
    for prof in PROFILES:
        key = f"host_count_spec_asus_{prof}"
        if dc_cfg.get(key):
            return prof, int(dc_cfg[key])
    raise ValueError(f"no known host spec in DC {dc_cfg.get('datacenter_id')}")


def executable_mi_per_on_second(datacenters) -> float:
    """Green-DC executable MI in one synchronized ON second."""
    total = 0.0
    for dc in datacenters:
        dcid = int(dc["datacenter_id"])
        if dcid not in H_D_W:
            continue
        prof, hosts = dc_profile(dc)
        pes_cap = PROFILES[prof][0] * hosts
        w_eff = w_per_pe_eff(prof)
        total += min(H_D_W[dcid] / w_eff, pes_cap) * VM_MIPS
    return total


def audit(schedule_art: dict, trace_rows, datacenters, off_range: int):
    # ON flags = complement of the artifact's trough intervals
    on = [True] * schedule_art["rows"]
    for t in schedule_art["troughs"]:
        for j in range(t["start"], min(t["start"] + t["dur"],
                                       schedule_art["rows"])):
            on[j] = False
    per_sec = executable_mi_per_on_second(datacenters)
    job_mi = sum(float(r["length"]) for r in trace_rows)
    windows = []
    for k in ANCHORS:
        off = episode_offset(k, off_range)
        s = WARMUP_ROWS + off
        on_count = sum(1 for j in range(s, min(s + int(HORIZON_S), len(on)))
                       if on[j])
        exe = on_count * per_sec
        windows.append({"anchor": k, "offset": off, "on_seconds": on_count,
                        "green_executable_mi": exe,
                        "ratio": exe / max(1.0, RATIO_MIN * job_mi) * RATIO_MIN})
    worst = min(w["ratio"] for w in windows)
    return {"per_on_second_mi": per_sec, "total_job_mi": job_mi,
            "windows": windows, "worst_window_ratio": worst,
            "requirement": RATIO_MIN, "pass": worst >= RATIO_MIN}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--schedule", choices=("cal", "ho"), default="cal")
    args = ap.parse_args()
    from src.baselines.evaluate import load_config
    repo = pathlib.Path(__file__).resolve().parent
    prefix = "sqt2" if args.schedule == "cal" else "sqt2ho"
    exp = ("experiment_sqt2_noforecast" if args.schedule == "cal"
           else "experiment_sqt2ho_noforecast")
    cfg = load_config(exp)
    art = json.loads((repo / f"calib/{prefix}_schedule.json").read_text())
    trace = (repo.parent / "cloudsimplus-gateway/src/main/resources"
             / cfg["cloudlet_trace_file"])
    rows = list(csv.DictReader(open(trace)))
    res = audit(art, rows, cfg["datacenters"],
                int(cfg.get("green_episode_offset_range", 0) or 0))
    out = repo / f"calib/{prefix}_capacity_audit.json"
    out.write_text(json.dumps(res, indent=1))
    print(f"per-ON-second executable MI: {res['per_on_second_mi']:.3e}")
    print(f"total job MI: {res['total_job_mi']:.3e}")
    print(f"worst anchor-window ratio: {res['worst_window_ratio']:.2f} "
          f"(require >= {RATIO_MIN}) -> {'PASS' if res['pass'] else 'FAIL'}")
    if not res["pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
