"""Schedule-only feasibility and a contract-safe reservation policy.

Codex 2026-09-01. Both routines here decide whether a load can be run at all. If either
could see the weather, the set of accepted workloads would depend on it, and the screen
would be choosing its own instances by how windy they are.

Determinism is by deterministic time and a fixed solver seed, not by wall clock: a retry
accepted because one machine was faster would make the frozen seed sequence meaningless.
"""
from __future__ import annotations

import os
import sys

import numpy as np
from ortools.sat.python import cp_model

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_gen as ig  # noqa: E402

N_DC = ig.N_DC
CAP = ig.CAP_PES_PER_SITE
RANDOM_SEED = 20260901
MAX_DETERMINISTIC_TIME = 30.0


def latest_start(w, i):
    return int(min(w["deadline"][i] - w["runtime"][i],
                   w["horizon"] - w["runtime"][i],
                   w["arrival"][i] + w["wait_cap"]))


def capacity_ok(w, budget):
    """Is there ANY schedule meeting capacity, deadlines, wait caps and the budget?

    Pure constraints, no objective. FEASIBLE or OPTIMAL both mean a witness exists.
    UNKNOWN means the search ran out of deterministic budget and says nothing either way.
    """
    n, T = len(w["arrival"]), w["horizon"]
    m = cp_model.CpModel()
    x = {}
    for i in range(n):
        lo, hi = int(w["arrival"][i]), latest_start(w, i)
        opts = []
        for d in range(N_DC):
            for s in range(lo, hi + 1):
                v = m.NewBoolVar(f"x_{i}_{d}_{s}")
                x[(i, d, s)] = v
                opts.append(v)
        if not opts:
            return "INFEASIBLE"
        m.AddExactlyOne(opts)
    for d in range(N_DC):
        for t in range(T):
            active = [int(w["pes"][i]) * x[(i, d, s)] for (i, dd, s) in x
                      if dd == d and s <= t < s + w["runtime"][i]]
            if active:
                m.Add(sum(active) <= CAP)
    m.Add(sum((s - int(w["arrival"][i])) * v for (i, _d, s), v in x.items()) <= int(budget))
    solver = cp_model.CpSolver()
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = RANDOM_SEED
    solver.parameters.max_deterministic_time = MAX_DETERMINISTIC_TIME
    st = solver.Solve(m)
    if st in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return "FEASIBLE"
    if st == cp_model.INFEASIBLE:
        return "INFEASIBLE"
    return "UNKNOWN"


def reservation_edf(w, budget):
    """Earliest-deadline-first with a persistent, irrevocable capacity reservation.

    On arrival a job takes the earliest start it can hold, on the lowest-indexed site that
    can hold it, and never gives it back. Ties among simultaneous arrivals go to the
    earlier deadline and then the smaller job id. If no reservation exists the policy fails
    at once; it does not backtrack or reorder.
    """
    n, T = len(w["arrival"]), w["horizon"]
    used = np.zeros((N_DC, T), dtype=int)
    assign, spent = {}, 0
    order = sorted(range(n), key=lambda i: (int(w["arrival"][i]),
                                            int(w["deadline"][i]), i))
    for i in order:
        lo, hi, r, p = int(w["arrival"][i]), latest_start(w, i), int(w["runtime"][i]), int(w["pes"][i])
        placed = None
        for s in range(lo, hi + 1):
            if spent + (s - lo) > budget:
                break                      # any later start overruns the shared budget
            for d in range(N_DC):
                if np.all(used[d, s:s + r] + p <= CAP):
                    placed = (d, s)
                    break
            if placed:
                break
        if placed is None:
            return None, None
        d, s = placed
        used[d, s:s + r] += p
        assign[i] = (d, s)
        spent += s - lo
    if spent > budget:
        return None, None
    return assign, spent
