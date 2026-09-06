"""Dominance-safe planner for the forecast-quality ladder. MODEL_VERSION 2 (2026-09-06).

Version 1 (reports/ERROR_LADDER_PLANNER_PREREG.md, frozen 24b5de60, closed by Addendum E with
two STOPs, archived in reports/manifests/ladder_v3/run1) charged ⌈PEs/64⌉ hosts per site and one
RS500A power function everywhere. The settlement diagnostic (reports/SETTLEMENT_DIAG_2026_09_06.md)
attributed the whole model-vs-simulator gap to four terms; this version closes the two that are
the model's (A, D) and keeps the interface of version 1:

  A  the simulator runs every job on its own host: VM i of a site sits on host i mod H (fixed
     topology) and the placement ledger takes the most-free fitting VM, lowest id; a VM whose
     job ends at row e is free again from row e + 1 (at row e the finish and the new routing
     share the clock and the VM still counts as taken: k0 job 28). So a job is assigned VM id j
     only when VMs 0..j-1 are busy or just freed, hence every VM id used is <= (occupancy - 1)
     with occupancy(d, t) = jobs running at t + jobs ending at t, and with occupancy <= H all
     concurrent jobs sit on distinct hosts (`placement_hosts` reproduces the rule,
     `verify_schedule` enforces the premise). Model: active hosts = running jobs; the premise
     "occupancy(d, t) <= H_d" is a hard constraint of the planner and a fail-fast check of
     every settlement.
  D  per-site host profile: P_job(d, p) = idle_w + (max_w - idle_w) * p * vm_mips / (host_pes *
     host_mips), in integer mW (65,640 mW on RS500A_DYN, 65,600 mW on RS700A_DYN for 32 PEs).

Exact time-indexed model, one window at a time:
  x[j, d, s] in {0,1}: job j starts at site d at step s (s in [a_j + LAG(2), L_j], L_j = D_j - r_j - eps)
  PEs per (d, t) <= cap_d;  occupancy per (d, t) <= H_d   (premise A: running + ending jobs)
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
# LAG: earliest executable start after a job's first sighting, in observation rows. Measured on
# the certification twin (k0 closure, 2026-09-06): a job routed at its first sighting (row a,
# the route-now path, kappa 0) begins executing at clock a + 2 = row a + 2; a held job (kappa
# >= 1, released at row s - 1) begins exactly at row s. Version 1 used 1 and the two route-now
# jobs of k0 ran one row late (the only residual of the closure). With 2 every planned start is
# reached through the hold path and lands exactly.
EPS_STEPS, LAG = 2, 2

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
    # causal rolling expert (2026-09-06): load already committed on the reservation grid, per
    # (site, row): draw in mW and occupancy (running + ending jobs); the new jobs are planned
    # on top of it. None = nothing committed (the offline instance).
    base_draw_mw: Optional[np.ndarray] = None
    base_occ: Optional[np.ndarray] = None
    # optional per-(job, site) legal starts (the executor's legality mask); default: starts
    starts_by_site: Optional[Dict[int, Dict[int, Sequence[int]]]] = None

    def legal_starts(self, jid: int, d: int):
        if self.starts_by_site is not None and jid in self.starts_by_site:
            return list(self.starts_by_site[jid].get(d, []))
        return self.starts[jid]

    def eff_curve_mw(self) -> np.ndarray:
        """Green left for new jobs: curve minus committed draw (may be negative)."""
        return self.curves_mw if self.base_draw_mw is None else self.curves_mw - self.base_draw_mw

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


def build_instance(jobs: List[Job], sites: Sequence = None, curves_w: np.ndarray = None, cap: Sequence[int] = None,
                   base_draw_mw: Optional[np.ndarray] = None, base_occ: Optional[np.ndarray] = None,
                   starts_by_site: Optional[Dict[int, Dict[int, Sequence[int]]]] = None) -> Instance:
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
    inst = Instance(jobs=list(jobs), sites=list(sites), curves_mw=curves_mw, T=T,
                    base_draw_mw=None if base_draw_mw is None else np.asarray(base_draw_mw, dtype=np.int64),
                    base_occ=None if base_occ is None else np.asarray(base_occ, dtype=np.int64),
                    starts_by_site=starts_by_site)
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
    all free VMs equal, lowest id); a VM freed at row e is free from row e + 1. Returns
    {job id: (vm, host)}; raises if a job finds no VM."""
    by_id = {j.id: j for j in jobs}
    busy_until: Dict[int, Dict[int, int]] = {d: {} for d in range(len(sites))}   # site -> vm -> end step
    out = {}
    for jid, (d, s) in sorted(schedule.items(), key=lambda kv: (kv[1][1], kv[0])):
        j = by_id[jid]
        free = [v for v in range(sites[d].vms) if busy_until[d].get(v, -10) + 1 <= s]
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
    occ = np.zeros((n, T + 1), dtype=np.int64)          # running + ending (premise A)
    draw = np.zeros((n, T), dtype=np.int64)
    by_id = {j.id: j for j in inst.jobs}
    for jid, (d, s) in schedule.items():
        j = by_id[jid]
        pes[d, s:s + j.runtime] += j.pes
        jobs[d, s:s + j.runtime] += 1
        occ[d, s:s + j.runtime + 1] += 1
        draw[d, s:s + j.runtime] += inst.sites[d].job_power_mw(j.pes)
    if inst.base_draw_mw is not None:
        draw = draw + inst.base_draw_mw[:, :T]
    if inst.base_occ is not None:
        occ[:, :T] += inst.base_occ[:, :T]
    hosts = jobs.copy()
    premise_ok = all(int(occ[d].max()) <= inst.sites[d].hosts for d in range(n)) if T else True
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
    cnt = np.zeros((n, T + 1), dtype=np.int64)
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
        if inst.starts_by_site is not None and s not in set(inst.legal_starts(j.id, d)):
            v.append(f"job {j.id} start {s} on site {d} is not a legal (masked) start")
        pes[d, s:s + j.runtime] += j.pes
        cnt[d, s:s + j.runtime + 1] += 1
    if inst.base_occ is not None:
        cnt[:, :T] += inst.base_occ[:, :T]
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


def solve_milp(inst: Instance, time_limit_s: float = TIME_LIMIT_S, mip_gap: float = 0.0, envelope_cuts: bool = True) -> dict:
    """The model as a MIP for HiGHS (scipy.optimize.milp): x binary, brown continuous (integral
    at optimality since every coefficient is integer). Exactness: OPTIMAL iff HiGHS reports
    optimal with the relative gap set to 0 and the compound checks hold."""
    import time
    from scipy.optimize import milp, LinearConstraint, Bounds
    from scipy.sparse import lil_matrix
    n, T = inst.curves_mw.shape
    xs = [(j.id, d, s) for j in inst.jobs for d in range(n) for s in inst.legal_starts(j.id, d)]
    if any(not any(inst.legal_starts(j.id, d) for d in range(n)) for j in inst.jobs):
        return {"status": "INFEASIBLE", "reason": "a job has no legal start"}
    Geff = inst.eff_curve_mw()
    base_occ = inst.base_occ if inst.base_occ is not None else np.zeros((n, T + 1), dtype=np.int64)
    xi = {key: i for i, key in enumerate(xs)}
    cells = [(d, t) for d in range(n) for t in range(T)]
    ci = {c: k for k, c in enumerate(cells)}
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
            for s in inst.legal_starts(j.id, d):
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
            A[brown_row[(d, t)], i] += -pw
        for t in range(s, min(T, s + j.runtime + 1)):        # occupancy: running + ending (premise A)
            A[cnt_row[(d, t)], i] += 1.0
    for (d, t) in cells:                                            # PEs <= cap
        lo.append(-np.inf); hi.append(float(inst.sites[d].cap))
    for (d, t) in cells:                                            # occupancy <= hosts (premise A), minus committed
        lo.append(-np.inf); hi.append(float(inst.sites[d].hosts - int(base_occ[d, t])))
    for k, (d, t) in enumerate(cells):                              # brown - draw_new >= -(G - committed draw)
        A[brown_row[(d, t)], boff + k] = 1.0
        lo.append(-float(Geff[d, t])); hi.append(np.inf)
    # Equivalent tightening (2026-09-06): the convex envelope of brown = max(0, P n - G) over the
    # INTEGER job counts n of a cell. With m = floor(G / P) the envelope between n = m and m + 1
    # is brown >= (P (m + 1) - G) (n - m), stronger than the LP line where G is not a multiple of
    # P (validity: RHS - true brown = (P m - G)(n - m - 1) <= 0 for n >= m + 1, RHS <= 0 for
    # n <= m). Applied on cells whose jobs all draw the same P (true for a one-PE-class trace);
    # it changes no integer solution, only the relaxation. Row count: one per such cell.
    if envelope_cuts:
        cut_rows = []
        for (d, t) in cells:
            i_list = [(xi[(jid, dd, ss)], by_id[jid]) for (jid, dd, ss) in xs if dd == d and ss <= t < ss + by_id[jid].runtime]
            if not i_list:
                continue
            P = {inst.sites[d].job_power_mw(j.pes) for _, j in i_list}
            if len(P) != 1:
                continue
            P = P.pop(); G = int(Geff[d, t]); m = G // P
            slope = P * (m + 1) - G
            if slope <= 0 or slope >= P:
                continue
            cut_rows.append((d, t, slope, m, [i for i, _ in i_list]))
        if cut_rows:
            from scipy.sparse import vstack
            C = lil_matrix((len(cut_rows), nvar))
            for r_i, (d, t, slope, m, idx) in enumerate(cut_rows):
                C[r_i, boff + ci[(d, t)]] = 1.0
                for i in idx:
                    C[r_i, i] = -slope
                lo.append(-float(slope * m)); hi.append(np.inf)
            A = vstack([A.tocsr(), C.tocsr()]).tolil()
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
           "model_version": MODEL_VERSION, "n_binaries": nx, "n_cells": nc, "envelope_cuts": bool(envelope_cuts)}
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
    # with committed load the MILP objective omits the constant "draw of the committed jobs"
    # (brown is on the full draw); settle counts it, so compare after adding it back
    const = int(inst.base_draw_mw[:, :T].sum()) if inst.base_draw_mw is not None else 0
    out["J_const_committed"] = const
    out["bound"] = None if out["mip_dual_bound"] is None else out["mip_dual_bound"] + const
    checks = {"highs_optimal": res.status == 0,
              "schedule_valid": not verify_schedule(inst, sched),
              "objective_matches": out["fun"] is not None and abs(out["fun"] + const - out["J_int"]) < 0.5,
              "bound_closes": (out["mip_dual_bound"] is not None and math.isfinite(out["mip_dual_bound"])
                               and out["J_int"] - (out["mip_dual_bound"] + const) < 1.0),
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
    if inst.base_draw_mw is not None or inst.base_occ is not None:
        raise NotImplementedError("CP-SAT cross-check has no committed-load support; use solve_milp")
    x = {}
    for j in inst.jobs:
        for d in range(n):
            for s in inst.legal_starts(j.id, d):
                x[(j.id, d, s)] = m.NewBoolVar(f"x_{j.id}_{d}_{s}")
        if not any(inst.legal_starts(j.id, d) for d in range(n)):
            return {"status": "INFEASIBLE", "reason": f"job {j.id} has no legal start"}
        m.AddExactlyOne(x[(j.id, d, s)] for d in range(n) for s in inst.legal_starts(j.id, d))
    running = {(d, t): [] for d in range(n) for t in range(T)}
    occupying = {(d, t): [] for d in range(n) for t in range(T)}
    for j in inst.jobs:
        for d in range(n):
            pw = inst.sites[d].job_power_mw(j.pes)
            for s in inst.legal_starts(j.id, d):
                for t in range(s, min(T, s + j.runtime)):
                    running[(d, t)].append((j.pes, pw, x[(j.id, d, s)]))
                for t in range(s, min(T, s + j.runtime + 1)):
                    occupying[(d, t)].append(x[(j.id, d, s)])
    objective_terms = []
    for d in range(n):
        max_draw = inst.sites[d].hosts * max(inst.sites[d].job_power_mw(j.pes) for j in inst.jobs)
        for t in range(T):
            terms = running[(d, t)]
            if occupying[(d, t)]:
                m.Add(sum(occupying[(d, t)]) <= inst.sites[d].hosts)
            if not terms:
                continue
            m.Add(sum(p * v for p, _, v in terms) <= inst.sites[d].cap)
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
