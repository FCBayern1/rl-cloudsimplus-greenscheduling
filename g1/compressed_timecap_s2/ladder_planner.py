"""Dominance-safe planner for the forecast-quality ladder (reports/ERROR_LADDER_PLANNER_PREREG.md,
frozen at 24b5de60: body < Addendum A < B < C).

Exact time-indexed CP-SAT model, one window at a time:
  x[j, d, s] in {0,1}: job j starts at site d at step s (s in [a_j + lag, L_j], L_j = D_j - r_j - eps)
  capacity per (d, t):  sum_j p_j x <= cap_d
  load_mW[d, t] = 2020 * sum_j p_j x  (2.02 W per PE, MIPS-based utilisation of RS500A_DYN)
  hosts h[d, t] >= sum_j p_j x / 64   (active-host floor, ceil, an approximation bounded by
                                        the 3 % model-simulator closure)
  brown_mW[d, t] >= load + 1000 h - G_mW[d, t], brown >= 0; green = load + 1000 h - brown
  J_int = sum_{d,t} (50 * brown + 1 * green)   (exact: 0.5 / 0.01 = 50);  C_kg = J_int / (3.6e11)
All curves are integers in mW; the only quantisation is the green curve's rounding, bounded by
  delta <= sites * steps * 0.5 mW * (f_b - f_g) / 3.6e9 kg  and required <= 0.1 % * C_brown_ref.
Every rung must be OPTIMAL (600 s); a non-optimal solve is STOP_SOLVER_RUNG_UNRESOLVED.

Pure functions: build_instance, settle (model settlement of a fixed schedule on any curve),
quantisation_bound, preflight_factors; solve() wraps ortools.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MW_PER_PE = 2020            # 2.02 W per PE: 161.6 W * 40000 / (64 * 50000)
HOST_FLOOR_MW = 1000        # 1 W technical floor per active host
HOST_PES = 64
F_BROWN, F_GREEN = 0.5, 0.01
RATIO = 50                  # F_BROWN / F_GREEN, exact
KG_PER_UNIT = 1.0 / 3.6e11  # one mW*s at 0.01 kg/kWh
TIME_LIMIT_S = 600.0
EPS_STEPS, LAG = 2, 1


@dataclass
class Job:
    id: int
    arrival: int          # step
    runtime: int          # steps
    pes: int
    deadline: int         # step (absolute)

    @property
    def latest(self) -> int:
        return self.deadline - self.runtime - EPS_STEPS


@dataclass
class Instance:
    jobs: List[Job]
    cap: List[int]                       # PEs per site
    curves_mw: np.ndarray                # (sites, T) integer mW
    T: int
    starts: Dict[int, range] = field(default_factory=dict)   # job id -> legal starts


def preflight_factors(datacenters: Sequence[dict]) -> None:
    for d in datacenters:
        if abs(float(d.get("brown_carbon_factor", -1)) - F_BROWN) > 1e-12 or \
           abs(float(d.get("green_carbon_factor", -1)) - F_GREEN) > 1e-12:
            raise RuntimeError(f"exact integer objective needs factors 0.5/0.01 on every site; got {d}")


def runtime_steps(mi: float, vm_pe_mips: float, cpu_util: float, timestep_sec: float = 1.0) -> int:
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_util)))
    return max(1, int(math.ceil(float(mi) / rate / timestep_sec)))


def build_instance(jobs: List[Job], cap: Sequence[int], curves_w: np.ndarray) -> Instance:
    """curves_w: (sites, T) green power in W (float); rounded to integer mW here."""
    curves_mw = np.rint(np.asarray(curves_w, dtype=np.float64) * 1000.0).astype(np.int64)
    T = int(curves_mw.shape[1])
    inst = Instance(jobs=list(jobs), cap=[int(c) for c in cap], curves_mw=curves_mw, T=T)
    for j in inst.jobs:
        lo = j.arrival + LAG
        hi = min(j.latest, T - j.runtime)
        inst.starts[j.id] = range(lo, hi + 1) if hi >= lo else range(0)
    return inst


def quantisation_bound_kg(n_sites: int, n_steps: int) -> float:
    return n_sites * n_steps * 0.5 * (F_BROWN - F_GREEN) / 3.6e9


def settle(inst: Instance, schedule: Dict[int, Tuple[int, int]], curves_mw: Optional[np.ndarray] = None) -> dict:
    """Model settlement of a fixed schedule {job id: (site, start)} on a curve (default: the
    instance's own). Returns J_int, C_kg and the per-(site, step) brown/green in mW."""
    G = inst.curves_mw if curves_mw is None else np.asarray(curves_mw, dtype=np.int64)
    n, T = G.shape
    pes = np.zeros((n, T), dtype=np.int64)
    by_id = {j.id: j for j in inst.jobs}
    for jid, (d, s) in schedule.items():
        j = by_id[jid]
        pes[d, s:s + j.runtime] += j.pes
    hosts = np.ceil(pes / HOST_PES).astype(np.int64)
    draw = pes * MW_PER_PE + hosts * HOST_FLOOR_MW
    brown = np.maximum(0, draw - G)
    green = draw - brown
    J = int(RATIO * brown.sum() + green.sum())
    return {"J_int": J, "C_kg": J * KG_PER_UNIT, "brown_mw": brown, "green_mw": green,
            "draw_mw": draw, "hosts": hosts, "pes": pes}


def solve(inst: Instance, time_limit_s: float = TIME_LIMIT_S, workers: int = 8) -> dict:
    """CP-SAT on the instance's curve. Returns status, schedule, J_int, C_kg, wall time."""
    from ortools.sat.python import cp_model
    m = cp_model.CpModel()
    n, T = inst.curves_mw.shape
    x = {}
    for j in inst.jobs:
        for d in range(n):
            for s in inst.starts[j.id]:
                x[(j.id, d, s)] = m.NewBoolVar(f"x_{j.id}_{d}_{s}")
        if not inst.starts[j.id]:
            return {"status": "INFEASIBLE", "reason": f"job {j.id} has no legal start"}
        m.AddExactlyOne(x[(j.id, d, s)] for d in range(n) for s in inst.starts[j.id])
    # per (d, t): running PEs
    running = {(d, t): [] for d in range(n) for t in range(T)}
    for j in inst.jobs:
        for d in range(n):
            for s in inst.starts[j.id]:
                for t in range(s, min(T, s + j.runtime)):
                    running[(d, t)].append((j.pes, x[(j.id, d, s)]))
    objective_terms = []
    max_draw = max(inst.cap) * MW_PER_PE + math.ceil(max(inst.cap) / HOST_PES) * HOST_FLOOR_MW
    for d in range(n):
        for t in range(T):
            terms = running[(d, t)]
            if not terms:
                continue
            pes_expr = sum(p * v for p, v in terms)
            m.Add(pes_expr <= inst.cap[d])
            h = m.NewIntVar(0, math.ceil(inst.cap[d] / HOST_PES), f"h_{d}_{t}")
            m.Add(HOST_PES * h >= pes_expr)
            draw = MW_PER_PE * pes_expr + HOST_FLOOR_MW * h
            brown = m.NewIntVar(0, max_draw, f"b_{d}_{t}")
            m.Add(brown >= draw - int(inst.curves_mw[d, t]))
            # J = 50 brown + green = 50 brown + (draw - brown) = 49 brown + draw
            objective_terms.append((RATIO - 1) * brown + draw)
    m.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = int(workers)
    status = solver.Solve(m)
    name = solver.StatusName(status)
    out = {"status": name, "wall_s": solver.WallTime()}
    if status in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        sched = {}
        for (jid, d, s), v in x.items():
            if solver.Value(v):
                sched[jid] = (d, s)
        out["schedule"] = sched
        out["J_int"] = int(round(solver.ObjectiveValue()))
        out["C_kg"] = out["J_int"] * KG_PER_UNIT
        out["bound"] = solver.BestObjectiveBound()
    return out
