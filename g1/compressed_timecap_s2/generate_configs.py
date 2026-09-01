#!/usr/bin/env python3
"""Programmatic derivation of every scheme-2 experiment block from one frozen base.

Nothing here is hand written. Section 8 of the work order forbids copying a batch of YAML
by hand and comparing it by eye, because the conclusion of this line is attributable only
if the arms differ in the variables under test and in nothing else. So every block is a
deep copy of

    g1/config_C_2020.yml :: experiment_g1eval_matchedvan

with a closed, tested set of rewrites applied on top. The test module pins both directions:
a key outside the whitelist that drifts is a failure, and a key inside the whitelist that
does not carry its registered value is also a failure.

Stage A (one block per cell) rewrites only:

    experiment_name / simulation_name       identity
    cloudlet_trace_file                     the cell's generated trace
    max_episode_length                      the cell's realised last finish plus the drain
    green_episode_offset_range              from windows.json, so every reachable offset
                                            is inside the 2021 series by construction
    wind_csv_year                           2021 — the scheduler evaluation year. The base
                                            block runs on 2020, which is the year reserved
                                            for TimeCAP training, so leaving it would break
                                            the isolation this whole scheme rests on.
    green_interpolation_mode                STEP, block level and per DC. A row is one
                                            control epoch; SPLINE would interpolate across
                                            epoch boundaries and blur the thing being
                                            forecast.
    green_power_scale                       1.0, pinned rather than defaulted. In
                                            COMPRESSED the divisor carries the scaling and
                                            a stray multiplier here would silently rescale
                                            every green number.
    cloudlet_cpu_utilization                1.0, so runtime = MI / (PES * MIPS) exactly.
                                            The base block inherits the 0.5 legacy default,
                                            which stretches every job to twice its
                                            registered runtime and voids the closure
                                            condition.
    defer_deadline_force_mode               latest_start, explicit. This is the key the
                                            work order singles out: on the legacy fixed-lead
                                            rule the backstop fires 600 rows early and
                                            force-routes the whole 144-row exam before the
                                            scheduler ever gets to decide.
    defer_deadline_slack_sec                1.0 row, so the forced start lands on the
                                            registered latest legal start and not before.
    defer_urgency_window_sec                144.0 rows, the horizon, instead of an
                                            inherited 3600 that means nothing here.
    obs_cloudlet_mi_high                    1.25 x the largest MI in the whole grid, frozen
                                            grid-wide so cells stay comparable. The base
                                            bound clips the r=72 pes=4 job.
    forecast_mode / green_oracle_mode       the registered blind pairing; Stage A's arms
                                            read the wind CSV themselves and no predictor
                                            may be loaded.
    timecap                                 removed. Stage A must not be able to touch a
                                            checkpoint even by accident.

Stage C adds the forecast arms on top of one frozen Stage A cell, and may differ between
arms only in forecast_mode, green_oracle_mode, the timecap sub-block, and the two names.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import constants as C                                            # noqa: E402
import workload as W                                             # noqa: E402

REPO = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_DIR = os.path.join(REPO, "g1/compressed_timecap_s2")
TRACES_DIR = os.path.join(REPO, "cloudsimplus-gateway/src/main/resources/traces")
WINDOWS_JSON = os.path.join(OUT_DIR, "windows.json")

# Every top-level key Stage A is allowed to differ from the base in. Anything else is drift.
STAGE_A_WHITELIST = {
    "experiment_name", "simulation_name", "cloudlet_trace_file", "max_episode_length",
    "green_episode_offset_range", "wind_csv_year", "green_interpolation_mode",
    "green_power_scale", "cloudlet_cpu_utilization", "defer_deadline_force_mode",
    "defer_deadline_slack_sec", "defer_urgency_window_sec", "obs_cloudlet_mi_high",
    "forecast_mode", "green_oracle_mode", "timecap", "datacenters",
}
# Within datacenters[*], Stage A may only touch these.
STAGE_A_DC_WHITELIST = {"green_interpolation_mode", "green_power_scale"}

# Stage C arms may differ from one another only in these. Same set preflight_scenario.py
# already enforces, plus the timecap sub-block that carries the negative-control switch.
STAGE_C_ARM_WHITELIST = {
    "forecast_mode", "green_oracle_mode", "timecap", "experiment_name", "simulation_name",
}
STAGE_C_ARMS = ("noforecast", "clean", "shuffle", "anti")

GREEN_INTERPOLATION_MODE = "STEP"
GREEN_POWER_SCALE = 1.0
BLIND_FORECAST_MODE = "none"
BLIND_ORACLE_MODE = "godeye"


def load_base(path=None):
    path = path or os.path.join(REPO, C.BASE_CONFIG_REL)
    cfg = yaml.safe_load(open(path))
    if C.BASE_BLOCK not in cfg:
        raise KeyError(f"base block missing: {C.BASE_BLOCK} in {path}")
    return cfg[C.BASE_BLOCK]


def load_windows(path=None):
    return json.load(open(path or WINDOWS_JSON))


def block_name(cell):
    return f"{C.BLOCK_PREFIX}_{C.cell_key(cell)}"


def derive_stage_a(base, cell, windows, wl=None):
    """One Stage A block. Pure: the base is never mutated."""
    wl = wl if wl is not None else W.draw(cell)
    b = copy.deepcopy(base)
    key = C.cell_key(cell)

    b["experiment_name"] = f"cts2_{key}"
    b["simulation_name"] = f"CTS2_{key}"
    b["cloudlet_trace_file"] = f"traces/{W.trace_name(cell)}"
    b["max_episode_length"] = W.episode_steps(wl)
    b["green_episode_offset_range"] = int(windows["green_episode_offset_range"])
    b["wind_csv_year"] = C.YEAR_SCHEDULER_EVAL
    b["green_interpolation_mode"] = GREEN_INTERPOLATION_MODE
    b["green_power_scale"] = GREEN_POWER_SCALE
    b["cloudlet_cpu_utilization"] = C.CPU_UTIL
    b["defer_deadline_force_mode"] = "latest_start"
    b["defer_deadline_slack_sec"] = C.DEFER_SLACK_ROWS
    b["defer_urgency_window_sec"] = C.DEFER_URGENCY_WINDOW_ROWS
    b["obs_cloudlet_mi_high"] = C.obs_cloudlet_mi_high()
    b["forecast_mode"] = BLIND_FORECAST_MODE
    b["green_oracle_mode"] = BLIND_ORACLE_MODE
    b.pop("timecap", None)

    for dc in b["datacenters"]:
        dc["green_interpolation_mode"] = GREEN_INTERPOLATION_MODE
        dc["green_power_scale"] = GREEN_POWER_SCALE
    return b


def derive_stage_c(stage_a_block, arm, timecap_checkpoint=None):
    """One Stage C arm on top of an already frozen Stage A block.

    The no-forecast arm is the Stage A block itself under a different name: matched means
    matched, so it keeps the same trace, the same episode length, the same backstop and the
    same window. clean / shuffle / anti differ from it only by switching the predictor on
    and by the perturbation applied to the predictor's output.

    timecap.forecast_perturbation is the switch the negative controls ride on. The gateway
    does not implement it yet; Stage C may not start until it does and until a test pins
    that shuffle and anti actually change the served forecast. Emitting the key here fixes
    its name and its values before any result is seen.
    """
    if arm not in STAGE_C_ARMS:
        raise ValueError(f"unknown Stage C arm: {arm!r}")
    b = copy.deepcopy(stage_a_block)
    base_name = b["experiment_name"]
    b["experiment_name"] = f"{base_name}_{arm}"
    b["simulation_name"] = f"CTS2_{b['experiment_name']}"
    if arm == "noforecast":
        return b
    b["forecast_mode"] = "full"
    b["green_oracle_mode"] = "timecap"
    tc = {"csv_year": C.YEAR_SCHEDULER_EVAL, "device": "cpu", "feature_set": "v1",
          "forecast_every": 6, "warmup_on_reset": True,
          "forecast_perturbation": {"none": "none", "clean": "none",
                                    "shuffle": "shuffle", "anti": "anti"}[arm]}
    if timecap_checkpoint:
        tc["checkpoint"] = timecap_checkpoint
    b["timecap"] = tc
    return b


def diff_keys(a, b):
    return {k for k in set(a) | set(b) if a.get(k) != b.get(k)}


def build_stage_a(base=None, windows=None):
    base = base if base is not None else load_base()
    windows = windows if windows is not None else load_windows()
    blocks, reports = {}, []
    for cell in C.cells():
        wl = W.draw(cell)
        checks, ok = W.assertions(wl)
        if not ok:
            raise AssertionError(f"workload invariants failed for {C.cell_key(cell)}: "
                                 f"{[k for k, v in checks.items() if not v]}")
        blocks[block_name(cell)] = derive_stage_a(base, cell, windows, wl)
        reports.append(W.report(wl))
    return blocks, reports


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


HEADER = (
    "# " + "-" * 76 + "\n"
    "# COMPRESSED TimeCAP scheme 2 — Stage A blocks (accelerated-weather synthetic\n"
    "# mechanism positive control). Generated by g1/compressed_timecap_s2/generate_configs.py\n"
    "# from g1/config_C_2020.yml :: experiment_g1eval_matchedvan. Do not hand edit: the\n"
    "# whitelist test in test_generate_configs.py will fail and the run will be void.\n"
    "# One wind row = one synthetic control epoch = one simulation second. Not ten minutes.\n"
    "# " + "-" * 76 + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="write the config, the traces and the manifest (default: dry run)")
    ap.add_argument("--out", default=os.path.join(OUT_DIR, "config_cts2_stage_a.yml"))
    a = ap.parse_args()

    windows = load_windows()
    blocks, reports = build_stage_a(windows=windows)
    print(f"cells {len(blocks)}   offset range {windows['green_episode_offset_range']}   "
          f"discovery {windows['discovery']}   confirmation {windows['confirmation']}")
    if not a.write:
        print("dry run; pass --write to emit traces, config and manifest")
        return

    for cell in C.cells():
        W.write_trace(W.draw(cell), TRACES_DIR)

    with open(a.out, "w") as f:
        f.write(HEADER)
        yaml.safe_dump(blocks, f, default_flow_style=False, sort_keys=True,
                       allow_unicode=True, width=4096)

    manifest = {
        "identity": "accelerated-weather synthetic mechanism positive control (scheme 2)",
        "base_config": C.BASE_CONFIG_REL, "base_block": C.BASE_BLOCK,
        "windows_selection_hash": windows["selection_hash"],
        "stage_a_config": os.path.relpath(a.out, REPO),
        "stage_a_config_sha256": sha256_file(a.out),
        "n_cells": len(blocks),
        "obs_cloudlet_mi_high": C.obs_cloudlet_mi_high(),
        "base_seed": C.BASE_SEED,
        "workloads": sorted(reports, key=lambda r: r["key"]),
    }
    mpath = os.path.join(OUT_DIR, "workloads.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"wrote {a.out}\nwrote {mpath}\nwrote {len(blocks)} traces to {TRACES_DIR}")


if __name__ == "__main__":
    main()
