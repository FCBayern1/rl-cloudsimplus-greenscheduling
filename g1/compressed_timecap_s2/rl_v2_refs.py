"""RL_V2 smoke references (reports/RL_V2_SMOKE_PREREG.md §5): on the six reading windows,
cover_argmax (index ties) on every tier of the full channel and on the hollow channel, and the
offline flat planner on the two validation windows (the four test windows have it from the
F_FITS_V2 pass). Usage: python rl_v2_refs.py [flat|cover|all]"""
from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ladder_run as lr  # noqa: E402
import f_v2_run as fv2  # noqa: E402

OUT = os.path.join(HERE, "stage_a_out", "rl_v2")
EVAL_CFG = os.path.join(HERE, "config_rl_v2_eval.yml")
TIERS = ("godeye", "shrink75", "shrink50", "shrink25", "shrink0", "shuffle", "anti")
K = 73


def read_windows():
    man = json.load(open(os.path.join(OUT, "manifest.json")))
    return man["windows"]["read"]                 # k=0..5 of the read allowlist = F_FITS_V2 k12..k17


def cover():
    d = os.path.join(OUT, "ref"); os.makedirs(d, exist_ok=True)
    for i, off in enumerate(read_windows()):
        kk = 12 + i
        for chan in ("full", "none"):
            for tier in (TIERS if chan == "full" else ("godeye",)):
                cell = f"rl2e_{chan}_{tier}"
                out_csv = os.path.join(d, f"cover_{chan}_{tier}_k{kk}.csv"); dump = os.path.join(d, f"cover_{chan}_{tier}_k{kk}_decisions.csv")
                for p in (out_csv, dump, dump.replace(".csv", "_obs.npz")):
                    if os.path.exists(p):
                        os.remove(p)
                ok = lr._evaluate(EVAL_CFG, cell, i, off, "cover_argmax", out_csv,
                                  {"OFFSET_GRID_DENSE": "1", "COVER_TIE": "index", "EVAL_DECISION_DUMP": dump})
                row = list(csv.DictReader(open(out_csv)))[-1] if ok else {}
                print(f"cover {chan} {tier} k{kk}: {'ok' if ok else 'FAILED'} carbon {row.get('total_carbon_kg')} ontime {row.get('ontime_mi_share')}", flush=True)


def flat():
    """Offline flat (lambda = 0) planner, closed by replay, on the two validation windows (F_FITS_V2 k12, k13)."""
    from ladder_planner import build_instance, solve_milp, settle
    d = os.path.join(OUT, "flat"); os.makedirs(d, exist_ok=True)
    cfg_def, cell_def = fv2.twin("defer"); cfg_off, cell_off = fv2.twin("F1")
    cfg_all = yaml.safe_load(open(cfg_def)); blk = cfg_all[cell_def]; sites = lr.sites_from_config(cfg_all, blk)
    mips = float(blk["datacenters"][0].get("vm_pe_mips", 40000)); u = float(blk.get("cloudlet_cpu_utilization", 1.0))
    roles = fv2.windows()
    for k, (role, off) in enumerate(roles):
        if role != "val":
            continue
        dump = os.path.join(d, f"k{k}_dump_decisions.csv")
        for p in (dump, dump.replace(".csv", "_obs.npz")):
            if os.path.exists(p):
                os.remove(p)
        lr._evaluate(cfg_def, cell_def, k, off, "reactive_wait_planner", os.path.join(d, f"k{k}_dump.csv"), {"EVAL_DECISION_DUMP": dump, "EVAL_DECISION_DUMP_OBS": "1"})
        rows = list(csv.DictReader(open(dump))); jobs = lr.jobs_from_dump(rows, mips, u)
        need = max(j.latest + j.runtime for j in jobs) + 1
        truth, _ = lr.truth_curve(blk, off, need); mu = lr._mu_w(blk)
        G = lr.rung_curve(truth, "shrink_0", mu, seed_key=f"ladder:{off}")
        res = solve_milp(build_instance(jobs, sites, G), time_limit_s=3600)
        sj = os.path.join(d, f"k{k}_flat_schedule.json")
        json.dump({"schedule": {str(i): list(v) for i, v in res.get("schedule", {}).items()}, "grid": list(range(K))}, open(sj, "w"))
        ok = res.get("status") == "OPTIMAL" and lr._evaluate(cfg_off, cell_off, k, off, "schedule_replay", os.path.join(d, f"k{k}.csv"), lr.replay_env(sj))
        row = list(csv.DictReader(open(os.path.join(d, f"k{k}.csv"))))[-1] if ok else {}
        cm = settle(build_instance(jobs, sites, truth), res["schedule"])["C_kg"] if res.get("status") == "OPTIMAL" else None
        print(f"flat k{k} ({off}): {res.get('status')} C_model {cm} C_sim {row.get('total_carbon_kg')} ontime {row.get('ontime_mi_share')}", flush=True)


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("flat", "all"):
        flat()
    if what in ("cover", "all"):
        cover()
