"""Post-STOP feasibility diagnostic. Does not reopen Round 1 and does not touch carbon.

Round 1 Phase A stopped because no causal blind honoured the contract on every required
instance. Four pooled totals of None only say that each arm failed at least one cell; they
do not say whether the instances were schedulable at all. This module answers that and
nothing else.

    part one   a 1,296 x 4 failure matrix with the reason each arm gave, stratified by the
               registered axes, carbon never read
    part two   a pure feasibility model over the same instances: arrivals, runtimes,
               deadlines, per-site capacity, per-job wait caps and the total delay budget,
               with a constant objective

The result decides whether TB13 failed at scenario generation or at causal-baseline
design. It cannot revive the STOP, and it produces no EVPI.
"""
from __future__ import annotations

import collections
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import causal_blinds as cbl  # noqa: E402
import instance_gen as ig  # noqa: E402
import round1 as r1  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
TIME_LIMIT_S = 20.0
OUTER_WORKERS = 2


def feasibility_only(sc, time_limit_s=TIME_LIMIT_S):
    """Is there ANY schedule satisfying the contract? No carbon, no objective.

    Returns FEASIBLE, INFEASIBLE or UNKNOWN. UNKNOWN is not evidence of infeasibility.
    """
    m = cp_model.CpModel()
    x = {(i, d, s): m.NewBoolVar(f"x_{i}_{d}_{s}")
         for i in range(sc.n) for d in range(sc.n_dc) for s in sc.starts(i)}
    for i in range(sc.n):
        opts = [x[(i, d, s)] for d in range(sc.n_dc) for s in sc.starts(i)]
        if not opts:
            return "INFEASIBLE"          # this job has no admissible start at all
        m.AddExactlyOne(opts)
    for d in range(sc.n_dc):
        for t in range(sc.T):
            active = [sc.p[i] * x[(i, d, s)] for i in range(sc.n) for s in sc.starts(i)
                      if s <= t < s + sc.r[i]]
            if active:
                m.Add(sum(active) <= int(sc.cap[d]))
    m.Add(sum((s - int(sc.a[i])) * x[(i, d, s)]
              for i in range(sc.n) for d in range(sc.n_dc) for s in sc.starts(i)) <= sc.B)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "FEASIBLE"
    if st == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    return "UNKNOWN"


def _one(ax):
    sc, prov = ig.build_instance(ax, seed=0)
    row = {"axes": {k: v for k, v in ax.items() if k != "runtime_set"}, "blinds": {}}
    for name, fn in cbl.BLINDS.items():
        c, a, diag = fn(sc, prov["clim_residual_green"], diagnose=True)
        row["blinds"][name] = {"ok": c is not None, "reason": diag["reason"],
                               "at_epoch": diag["at_epoch"],
                               "pending_at_failure": diag["pending_at_failure"]}
    row["feasibility"] = feasibility_only(sc)
    row["rho_residual"] = prov["rho_residual"]
    return row


def main(out_dir=None):
    out_dir = out_dir or os.path.join(HERE, "poststop_out")
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    commit, shas, manifest = r1.preflight(os.path.join(HERE, "round0_out"))
    inst = r1.build_instances(os.path.join(HERE, "round0_out"))
    with ProcessPoolExecutor(max_workers=OUTER_WORKERS) as ex:
        rows = list(ex.map(_one, inst, chunksize=4))

    names = list(cbl.BLINDS)
    fail_sets = {n: {i for i, r in enumerate(rows) if not r["blinds"][n]["ok"]}
                 for n in names}
    reasons = {n: dict(collections.Counter(
        r["blinds"][n]["reason"] for r in rows if not r["blinds"][n]["ok"])) for n in names}
    feas = collections.Counter(r["feasibility"] for r in rows)

    def strat(field):
        out = {}
        for r in rows:
            k = str(r["axes"][field])
            e = out.setdefault(k, {"n": 0, "infeasible": 0,
                                   **{n: 0 for n in names}})
            e["n"] += 1
            e["infeasible"] += int(r["feasibility"] == "INFEASIBLE")
            for n in names:
                e[n] += int(not r["blinds"][n]["ok"])
        return out

    summary = {
        "note": "post-STOP diagnostic; Round 1 remains STOPPED and no carbon was read",
        "instances": len(rows),
        "feasibility": dict(feas),
        "blind_failures": {n: len(fail_sets[n]) for n in names},
        "blind_failure_reasons": reasons,
        "failed_by_all_four": len(set.intersection(*fail_sets.values())) if names else 0,
        "failed_by_at_least_one": len(set.union(*fail_sets.values())) if names else 0,
        "infeasible_and_blind_failed": len(
            {i for i, r in enumerate(rows) if r["feasibility"] == "INFEASIBLE"}
            & set.union(*fail_sets.values())) if names else 0,
        "feasible_but_all_blinds_failed": len(
            {i for i, r in enumerate(rows) if r["feasibility"] == "FEASIBLE"}
            & set.intersection(*fail_sets.values())) if names else 0,
        "strata": {f: strat(f) for f in ("n_jobs", "pes_per_job", "concurrency",
                                         "wait_cap", "budget_fraction")},
        "wall_seconds": round(time.time() - t0, 2),
        "provenance": {"commit": commit, "file_shas": shas},
    }
    r1._write(os.path.join(out_dir, "poststop_rows.jsonl"), rows, lines=True)
    r1._write(os.path.join(out_dir, "poststop_summary.json"), summary)
    return summary


if __name__ == "__main__":
    s = main()
    print(json.dumps({k: v for k, v in s.items() if k not in ("strata", "provenance")},
                     sort_keys=True, indent=2))
