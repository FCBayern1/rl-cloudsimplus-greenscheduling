"""Dominance-safe planner for the forecast-quality ladder. MODEL_VERSION 2 (2026-09-06).

Version 1 (reports/ERROR_LADDER_PLANNER_PREREG.md, frozen 24b5de60, closed by Addendum E with
two STOPs, archived in reports/manifests/ladder_v3/run1) charged ⌈PEs/64⌉ hosts per site and one
RS500A power function everywhere. The settlement diagnostic (reports/SETTLEMENT_DIAG_2026_09_06.md)
attributed the whole model-vs-simulator gap to four terms; this version closes the two that are
the model's (A, D) and keeps the interface of version 1:

  A  the simulator runs every job on its own host: VM i of a site sits on host i mod H (fixed
     topology) and the placement ledger takes the most-free fitting VM, lowest id, so a job is
     assigned VM id j only when VMs 0..j-1 are all busy; hence every VM id ever used is
     <= (max concurrency - 1), and with concurrency <= H all concurrent jobs sit on distinct
     hosts (`placement_hosts` reproduces the rule, `verify_schedule` enforces the premise).
     Model: active hosts = running jobs; the premise "concurrent jobs <= H_d" is a hard
     constraint of the planner and a fail-fast check of every settlement.
  D  per-site host profile: P_job(d, p) = idle_w + (max_w - idle_w) * p * vm_mips / (host_pes *
     host_mips), in integer mW (65,640 mW on RS500A_DYN, 65,600 mW on RS700A_DYN for 32 PEs).

Exact time-indexed model, one window at a time:
  x[j, d, s] in {0,1}: job j starts at site d at step s (s in [a_j + lag, L_j], L_j = D_j - r_j - eps)
  PEs per (d, t) <= cap_d;  running jobs per (d, t) <= H_d   (premise A)
  draw_mW[d, t] = sum_j P_job(d, p_j) x                      (linear: no host variables)
  brown_mW[d, t] >= draw - G_mW[d, t], brown >= 0; green = draw - brown
  J_int = sum_{d,t} (50 * brown + 1 * green) = sum 49 brown + draw;  C_kg = J_int / 3.6e11
All curves are integers in mW; the only quantisation is the green curve's rounding.

Pure functions: build_instance, settle, verify_schedule, placement_hosts, quantisation_bound,
preflight_factors; solve_milp (HiGHS) and solve (CP-SAT) wrap the solvers.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

MODEL_VERSION = 2
F_BROWN, F_GREEN = 0.5, 0.01
RATIO = 50                  # F_BROWN / F_GREEN, exact
KG_PER_UNIT = 1.0 / 3.6e11  # one mW*s at 0.01 kg/kWh
TIME_LIMIT_S = 600.0
EPS_STEPS, LAG = 2, 1

# host profiles of the gateway (HostProfile.java): PEs, MIPS per PE, full-load dynamic W, idle W
HOST_PROFILES = {
    "SPEC_ASUS_RS500A_DYN": {"pes": 64, "mips": 50000, "max_w": 214.0 - 51.4, "idle_w": 1.0},
    "SPEC_ASUS_RS700A_DYN": {"pes": 128, "mips": 50000, "max_w": 430.0 - 106.0, "idle_w": 1.0},
}
# legacy constants of version 1, kept for the archived stage's readers
MW_PER_PE = 2020
HOST_FLOOR_MW = 1000
HOST_PES = 64


@dataclass(frozen=True)
class Site:
    name: str
    profile: str
    hosts: int
    vms: int
    vm_pes: int
    vm_mips: float
    host_pes: int = 64
    host_mips: float = 50000.0
    max_w: float = 214.0 - 51.4
    idle_w: float = 1.0

    @property
    def cap(self) -> int:
        """PE capacity seen by the planner: the site's VMs."""
        return self.vms * self.vm_pes

    def job_power_mw(self, pes: int) -> int:
        """Power of one job of `pes` PEs on its own host, integer mW (idle floor + dynamic share)."""
        u = pes * self.vm_mips / (self.host_pes * self.host_mips)
        return int(round(1000.0 * (self.idle_w + (self.max_w - self.idle_w) * u)))


def site_from_profile(name: str, profile: str, hosts: int, vms: int, vm_pes: int = 32, vm_mips: float = 40000.0) -> Site:
    p = HOST_PROFILES[profile]
    return Site(name=name, profile=profile, hosts=int(hosts), vms=int(vms), vm_pes=int(vm_pes), vm_mips=float(vm_mips),
                host_pes=p["pes"], host_mips=p["mips"], max_w=p["max_w"], idle_w=p["idle_w"])


def sites_from_caps(caps: Sequence[int], vm_pes: int = 32) -> List[Site]:
    """Legacy helper: RS500A sites with as many hosts as 64-PE blocks and one 32-PE VM per 32 PEs."""
    return [site_from_profile(f"site{d}", "SPEC_ASUS_RS500A_DYN", hosts=max(1, int(c) // 64), vms=int(c) // vm_pes, vm_pes=vm_pes)
            for d, c in enumerate(caps)]


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
    sites: List[Site]
    curves_mw: np.ndarray                # (sites, T) integer mW
    T: int
    starts: Dict[int, range] = field(default_factory=dict)   # job id -> legal starts

    @property
    def cap(self) -> List[int]:
        return [s.cap for s in self.sites]


def preflight_factors(datacenters: Sequence[dict]) -> None:
    for d in datacenters:
        if abs(float(d.get("brown_carbon_factor", -1)) - F_BROWN) > 1e-12 or \
           abs(float(d.get("green_carbon_factor", -1)) - F_GREEN) > 1e-12:
            raise RuntimeError(f"exact integer objective needs factors 0.5/0.01 on every site; got {d}")


def runtime_steps(mi: float, vm_pe_mips: float, cpu_util: float, timestep_sec: float = 1.0) -> int:
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_util)))
    return max(1, int(math.ceil(float(mi) / rate / timestep_sec)))


def build_instance(jobs: List[Job], sites: Sequence = None, curves_w: np.ndarray = None, cap: Sequence[int] = None) -> Instance:
    """curves_w: (sites, T) green power in W (float); rounded to integer mW here. `sites` is a
    list of Site, or of PE capacities (legacy RS500A sites, see sites_from_caps); `cap` is the
    version-1 keyword for the latter."""
    if sites is None:
        sites = cap
    if sites and not isinstance(sites[0], Site):
        sites = sites_from_caps([int(c) for c in sites])
    curves_mw = np.rint(np.asarray(curves_w, dtype=np.float64) * 1000.0).astype(np.int64)
    T = int(curves_mw.shape[1])
    if curves_mw.shape[0] != len(sites):
        raise ValueError(f"curve has {curves_mw.shape[0]} sites, topology has {len(sites)}")
    inst = Instance(jobs=list(jobs), sites=list(sites), curves_mw=curves_mw, T=T)
    for j in inst.jobs:
        lo = j.arrival + LAG
        hi = min(j.latest, T - j.runtime)
        inst.starts[j.id] = range(lo, hi + 1) if hi >= lo else range(0)
    return inst


def quantisation_bound_kg(n_sites: int, n_steps: int) -> float:
    return n_sites * n_steps * 0.5 * (F_BROWN - F_GREEN) / 3.6e9


def placement_hosts(schedule: Dict[int, Tuple[int, int]], jobs: Sequence[Job], sites: Sequence[Site]) -> Dict[int, Tuple[int, int]]:
    """The simulator's placement rule reproduced: per site, VM ids 0..vms-1 on host (id mod H);
    at each start step, in id order, the job takes the lowest free VM (most-free fitting with
    all free VMs equal, lowest id). Returns {job id: (vm, host)}; raises if a job finds no VM."""
    by_id = {j.id: j for j in jobs}
    busy_until: Dict[int, Dict[int, int]] = {d: {} for d in range(len(sites))}   # site -> vm -> end step
    out = {}
    for jid, (d, s) in sorted(schedule.items(), key=lambda kv: (kv[1][1], kv[0])):
        j = by_id[jid]
        free = [v for v in range(sites[d].vms) if busy_until[d].get(v, -1) <= s]
        if not free:
            raise ValueError(f"job {jid}: no free VM on site {d} at step {s}")
        v = min(free)
        busy_until[d][v] = s + j.runtime
        out[jid] = (v, v % sites[d].hosts)
    return out


def settle(inst: Instance, schedule: Dict[int, Tuple[int, int]], curves_mw: Optional[np.ndarray] = None) -> dict:
    """Model settlement of a fixed schedule {job id: (site, start)} on a curve (default: the
    instance's own). Returns J_int, C_kg, per-(site, step) draw/brown/green/hosts/pes/jobs and
    `premise_ok` (concurrency <= hosts on every site and step)."""
    G = inst.curves_mw if curves_mw is None else np.asarray(curves_mw, dtype=np.int64)
    n, T = G.shape
    pes = np.zeros((n, T), dtype=np.int64)
    jobs = np.zeros((n, T), dtype=np.int64)
    draw = np.zeros((n, T), dtype=np.int64)
    by_id = {j.id: j for j in inst.jobs}
    for jid, (d, s) in schedule.items():
        j = by_id[jid]
        pes[d, s:s + j.runtime] += j.pes
        jobs[d, s:s + j.runtime] += 1
        draw[d, s:s + j.runtime] += inst.sites[d].job_power_mw(j.pes)
    hosts = jobs.copy()
    premise_ok = all(int(jobs[d].max()) <= inst.sites[d].hosts for d in range(n)) if T else True
    brown = np.maximum(0, draw - G)
    green = draw - brown
    J = int(RATIO * brown.sum() + green.sum())
    return {"J_int": J, "C_kg": J * KG_PER_UNIT, "brown_mw": brown, "green_mw": green,
            "draw_mw": draw, "hosts": hosts, "pes": pes, "jobs": jobs, "premise_ok": bool(premise_ok),
            "model_version": MODEL_VERSION}


def verify_schedule(inst: Instance, schedule: Dict[int, Tuple[int, int]]) -> List[str]:
    """Integer re-check of a full schedule (Addendum D3 + premise A): exactly one assignment per
    job, start >= arrival + lag, start <= latest, PE capacity and concurrency <= hosts per
    (site, step). Returns violations."""
    v = []
    ids = {j.id for j in inst.jobs}
    if set(schedule) != ids:
        v.append(f"assignments {sorted(set(schedule) ^ ids)} missing or extra")
    n, T = inst.curves_mw.shape
    pes = np.zeros((n, T), dtype=np.int64)
    cnt = np.zeros((n, T), dtype=np.int64)
    for j in inst.jobs:
        if j.id not in schedule:
            continue
        d, s = schedule[j.id]
        if not (0 <= d < n):
            v.append(f"job {j.id} site {d} out of range"); continue
        if s < j.arrival + LAG:
            v.append(f"job {j.id} starts {s} before arrival + lag {j.arrival + LAG}")
        if s > j.latest:
            v.append(f"job {j.id} starts {s} after latest {j.latest}")
        if s + j.runtime > T:
            v.append(f"job {j.id} runs past the horizon")
        pes[d, s:s + j.runtime] += j.pes
        cnt[d, s:s + j.runtime] += 1
    for d in range(n):
        over = np.where(pes[d] > inst.sites[d].cap)[0]
        if over.size:
            v.append(f"site {d} over capacity at steps {over[:5].tolist()}")
        crowd = np.where(cnt[d] > inst.sites[d].hosts)[0]
        if crowd.size:
            v.append(f"site {d} more running jobs than hosts ({inst.sites[d].hosts}) at steps {crowd[:5].tolist()}")
    return v


def schedule_hash(schedule: Dict[int, Tuple[int, int]]) -> str:
    import hashlib
    sig = ";".join(f"{k}:{d}:{s}" for k, (d, s) in sorted(schedule.items()))
    return hashlib.sha256(sig.encode()).hexdigest()[:16]


def solve_milp(inst: Instance, time_limit_s: float = TIME_LIMIT_S, mip_gap: float = 0.0) -> dict:
    """The model as a MIP for HiGHS (scipy.optimize.milp): x binary, brown continuous (integral
    at optimality since every coefficient is integer). Exactness: OPTIMAL iff HiGHS reports
    optimal with the relative gap set to 0 and the compound checks hold."""
    import time
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    n, T = inst.curves_mw.shape
    if any(not inst.starts[j.id] for j in inst.jobs):
        return {"status": "INFEASIBLE", "reason": "a job has no legal start"}
    xs = [(j.id, d, s) for j in inst.jobs for d in range(n) for s in inst.starts[j.id]]
    xi = {key: i for i, key in enumerate(xs)}
    cells = [(d, t) for d in range(n) for t in range(T)]
    nx, nc = len(xs), len(cells)
    boff = nx
    nvar = nx + nc
    by_id = {j.id: j for j in inst.jobs}
    # objective: sum_{d,t} 49 * brown + draw, draw = sum_j P_job(d, p_j) * x over the running steps
    c = np.zeros(nvar)
    for (jid, d, s), i in xi.items():
        j = by_id[jid]
        c[i] += inst.sites[d].job_power_mw(j.pes) * min(j.runtime, T - s)
    c[boff:boff + nc] = RATIO - 1
    lo, hi = [], []
    A = lil_matrix((len(inst.jobs) + 3 * nc, nvar))
    r = 0
    for j in inst.jobs:                                             # exactly one start
        for d in range(n):
            for s in inst.starts[j.id]:
                A[r, xi[(j.id, d, s)]] = 1.0
        lo.append(1.0); hi.append(1.0); r += 1
    pes_row = {cell: r + k for k, cell in enumerate(cells)}
    cnt_row = {cell: r + nc + k for k, cell in enumerate(cells)}
    brown_row = {cell: r + 2 * nc + k for k, cell in enumerate(cells)}
    for (jid, d, s), i in xi.items():
        j = by_id[jid]
        pw = inst.sites[d].job_power_mw(j.pes)
        for t in range(s, min(T, s + j.runtime)):
            A[pes_row[(d, t)], i] += j.pes
            A[cnt_row[(d, t)], i] += 1.0
            A[brown_row[(d, t)], i] += -pw
    for (d, t) in cells:                                            # PEs <= cap
        lo.append(-np.inf); hi.append(float(inst.sites[d].cap))
    for (d, t) in cells:                                            # running jobs <= hosts (premise A)
        lo.append(-np.inf); hi.append(float(inst.sites[d].hosts))
    for k, (d, t) in enumerate(cells):                              # brown - draw >= -G
        A[brown_row[(d, t)], boff + k] = 1.0
        lo.append(-float(inst.curves_mw[d, t])); hi.append(np.inf)
    integrality = np.zeros(nvar); integrality[:nx] = 1
    ub = np.full(nvar, np.inf); ub[:nx] = 1.0
    t0 = time.time()
    res = milp(c, constraints=LinearConstraint(A.tocsr(), np.array(lo), np.array(hi)), integrality=integrality,
               bounds=Bounds(np.zeros(nvar), ub), options={"time_limit": float(time_limit_s), "mip_rel_gap": float(mip_gap), "disp": False})
    out = {"wall_s": time.time() - t0, "milp_status": int(res.status), "milp_message": str(res.message),
           "fun": (float(res.fun) if getattr(res, "fun", None) is not None else None),
           "mip_gap": (float(res.mip_gap) if getattr(res, "mip_gap", None) is not None else None),
           "mip_dual_bound": (float(res.mip_dual_bound) if getattr(res, "mip_dual_bound", None) is not None else None),
           "mip_node_count": (int(res.mip_node_count) if getattr(res, "mip_node_count", None) is not None else None),
           "model_version": MODEL_VERSION, "n_binaries": nx, "n_cells": nc}
    if res.x is None:
        out["status"] = "INFEASIBLE" if res.status == 2 else "UNKNOWN"
        return out
    sched = {}
    for (jid, d, s), i in xi.items():
        if res.x[i] > 0.5:
            sched[jid] = (d, s)
    out["schedule"] = sched
    out["schedule_hash"] = schedule_hash(sched)
    st = settle(inst, sched)                  # exact integer objective from the schedule itself
    out["J_int"], out["C_kg"] = st["J_int"], st["C_kg"]
    out["bound"] = out["mip_dual_bound"]
    checks = {"highs_optimal": res.status == 0,
              "schedule_valid": not verify_schedule(inst, sched),
              "objective_matches": out["fun"] is not None and abs(out["fun"] - out["J_int"]) < 0.5,
              "bound_closes": (out["mip_dual_bound"] is not None and math.isfinite(out["mip_dual_bound"])
                               and out["J_int"] - out["mip_dual_bound"] < 1.0),
              "gap_finite": out["mip_gap"] is not None and math.isfinite(out["mip_gap"]),
              "premise_ok": st["premise_ok"]}
    out["checks"] = checks
    out["verify_violations"] = verify_schedule(inst, sched)
    out["status"] = "OPTIMAL" if all(checks.values()) else ("FEASIBLE" if res.status in (0, 1) else "UNKNOWN")
    return out


def solve(inst: Instance, time_limit_s: float = TIME_LIMIT_S, workers: int = 8, linearization_level: int = 2) -> dict:
    """CP-SAT on the instance's curve (cross-check solver). Returns status, schedule, J_int,
    C_kg, wall time."""
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
    running = {(d, t): [] for d in range(n) for t in range(T)}
    for j in inst.jobs:
        for d in range(n):
            pw = inst.sites[d].job_power_mw(j.pes)
            for s in inst.starts[j.id]:
                for t in range(s, min(T, s + j.runtime)):
                    running[(d, t)].append((j.pes, pw, x[(j.id, d, s)]))
    objective_terms = []
    for d in range(n):
        max_draw = inst.sites[d].hosts * max(inst.sites[d].job_power_mw(j.pes) for j in inst.jobs)
        for t in range(T):
            terms = running[(d, t)]
            if not terms:
                continue
            m.Add(sum(p * v for p, _, v in terms) <= inst.sites[d].cap)
            m.Add(sum(v for _, _, v in terms) <= inst.sites[d].hosts)
            draw = sum(pw * v for _, pw, v in terms)
            brown = m.NewIntVar(0, max_draw, f"b_{d}_{t}")
            m.Add(brown >= draw - int(inst.curves_mw[d, t]))
            objective_terms.append((RATIO - 1) * brown + draw)
    m.Minimize(sum(objective_terms))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = float(time_limit_s)
    solver.parameters.num_workers = int(workers)
    solver.parameters.linearization_level = int(linearization_level)
    status = solver.Solve(m)
    name = solver.StatusName(status)
    out = {"status": name, "wall_s": solver.WallTime(), "model_version": MODEL_VERSION}
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
