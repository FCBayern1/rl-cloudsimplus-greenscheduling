"""Option executor for HOLD_FOR_GREEN(d) (reports/OPTION_ACTION_DESIGN.md §2, Addenda A3 and B).

One executor, shared by every arm, living in the env layer. It owns

  * the reservation grid `occ[d, step]` (PEs committed at each site per step) with the
    planner's semantics: every route seen by the env is held on the grid for its runtime,
    every HOLD books a fallback reservation at the latest feasible start not later than
    the job's latest start, and nothing else writes to the grid;
  * the hold ledger: one persistent commitment per held job (site fixed at creation);
  * the termination rule, evaluated at every decision point BEFORE the batch is routed,
    tightest latest-start first, with same-step accumulators:
        T2  the fallback reservation is due (t + lag >= s_f)            -> release, "margin"
        T1  residual realised green at d covers the job's draw AND d has the free PEs now
            (both after the releases already made this step)             -> release, "green"
    and no other exit;
  * the legality of HOLD(d) per (slot, site): the deadline rule (given) AND a fallback
    reservation exists at d on the current grid;
  * the per-site held observation (count, PEs, tightest margin) and the ledger rows.

It reads the current clock, realised green, free PEs and the jobs' own deadline arithmetic;
no forecast, no future truth, no comparison of candidate starts.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

REASON_GREEN, REASON_MARGIN, REASON_OFFSET = "green", "margin", "offset"


def offset_grid(wait_cap_steps: int):
    """K(W) = {0} ∪ {2^q : 2^q < W} ∪ {W} (OPTION_ACTION_DESIGN Addendum A5), ascending.

    Diagnostic only: with OFFSET_GRID_DENSE=1 in the environment the grid is every step
    0..W. Never set in a gate run; the frozen grid is the dyadic one."""
    import os
    W = int(wait_cap_steps)
    if os.environ.get("OFFSET_GRID_DENSE", "") == "1":
        return list(range(W + 1))
    ks = {0, W}
    q = 1
    while q < W:
        ks.add(q)
        q *= 2
    return sorted(ks)


def runtime_steps(mi: float, vm_pe_mips: float, cpu_util: float, timestep_sec: float) -> int:
    """Steps a cloudlet occupies its site: ceil(mi / (mips * u) / timestep), PEs independent
    (the backstop's and the deadline mask's runtime unit)."""
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_util)))
    sec = max(0.0, float(mi)) / rate
    return max(1, int(np.ceil(sec / max(1e-9, float(timestep_sec)))))


DYN_MW_PER_PE_MODEL = 2.02        # W per PE under the RS500A_DYN MIPS utilisation (the planner's model)
HOST_FLOOR_W_MODEL = 1.0
HOST_PES_MODEL = 64


def cand_green_cover(future_w, committed_pes, pes, mi, ids, t_now, grid, vm_pe_mips, cpu_util,
                     static_w=None, lag=1, timestep_sec=1.0):
    """(NB, n * |K|) float32: for job j at site d with dispatch offset κ, the share of the job's
    dynamic energy over its runtime that the arm's forecast green at d covers after the site's
    static draw and the load already committed on the reservation grid before this decision
    (SCENE_INTERFACE_DESIGN §4.4, Addenda A3, B2). Energy-weighted by construction (draw × steps);
    same-batch jobs may co-claim the same residual (audited elsewhere). Pure.

    future_w: (n, H) forecast green in W for steps t_now .. t_now + H - 1 (the arm's own curve);
    committed_pes: (n, T) PEs committed on the executor grid (absolute steps);
    beyond the forecast horizon the residual is taken as zero (nothing is claimed there).
    """
    future = np.asarray(future_w, dtype=np.float64)
    n, H = future.shape
    ids = np.asarray(ids).reshape(-1)
    nb, K = ids.shape[0], len(grid)
    out = np.zeros((nb, n * K), dtype=np.float32)
    static = np.zeros(n) if static_w is None else np.asarray(static_w, dtype=np.float64).reshape(n)
    occ = np.asarray(committed_pes, dtype=np.float64)
    T = occ.shape[1]
    rate = max(1.0, float(vm_pe_mips)) * min(1.0, max(1e-6, float(cpu_util)))
    for j in range(nb):
        if int(ids[j]) < 0 or float(pes[j]) <= 0 or float(mi[j]) <= 0:
            continue
        p = float(pes[j])
        r = max(1, int(np.ceil(float(mi[j]) / rate / timestep_sec)))
        draw = p * DYN_MW_PER_PE_MODEL
        for d in range(n):
            for i, k in enumerate(grid):
                s = int(t_now) + int(k) + int(lag)               # absolute start step
                covered = 0.0
                for step in range(s, s + r):
                    h = step - int(t_now)                          # index into the forecast
                    if h < 0 or h >= H:
                        continue                                   # beyond the horizon: nothing claimed
                    c_pes = occ[d, step] if step < T else 0.0
                    hosts = np.ceil(c_pes / HOST_PES_MODEL)
                    resid = future[d, h] - static[d] - c_pes * DYN_MW_PER_PE_MODEL - hosts * HOST_FLOOR_W_MODEL
                    covered += min(draw, max(0.0, resid))
                out[j, d * K + i] = covered / (draw * r)
    return np.clip(out, 0.0, 1.0)


def residual_green(green_now_w: float, static_w: float, occupied_pes: float,
                   dyn_per_pe_w: float, cpu_util: float) -> float:
    """Green left on the meter at a site after its static draw and the dynamic draw of the
    PEs committed at the start step; the planner's `_reactive_choice` quantity."""
    return max(0.0, float(green_now_w) - float(static_w)
               - float(occupied_pes) * float(dyn_per_pe_w) * float(cpu_util))


@dataclass
class HeldJob:
    id: int
    dc: int
    pes: int
    mi: float
    r: int
    t_c: int
    latest: int
    s_f: int
    t_release: Optional[int] = None
    reason: Optional[str] = None
    r_release: Optional[float] = None
    extra: dict = field(default_factory=dict)


class OptionExecutor:
    def __init__(self, num_dcs: int, cap_pes, horizon_steps: int, dyn_per_pe_w: float,
                 static_w, cpu_util: float, vm_pe_mips: float, timestep_sec: float = 1.0,
                 eps_steps: int = 2, start_lag: int = 1):
        self.n = int(num_dcs)
        self.cap = np.asarray(cap_pes, dtype=np.float64).reshape(self.n)
        self.T = int(horizon_steps)
        self.dyn = float(dyn_per_pe_w)
        self.static = np.asarray(static_w, dtype=np.float64).reshape(self.n)
        self.u = float(cpu_util)
        self.mips = float(vm_pe_mips)
        self.dt = float(timestep_sec)
        self.eps = int(eps_steps)
        self.lag = int(start_lag)
        self.reset()

    # ── state ────────────────────────────────────────────────────────────────────────
    def reset(self):
        self.occ = np.zeros((self.n, self.T), dtype=np.float64)
        self.held: Dict[int, HeldJob] = {}
        self.done: Dict[int, HeldJob] = {}
        self.active: Dict[int, Tuple[int, int, int, int]] = {}
        self.n_created = 0
        self.n_refused = 0
        self.n_term_green = 0
        self.n_term_margin = 0
        self.n_route_seen = 0

    # ── arithmetic ───────────────────────────────────────────────────────────────────
    def runtime(self, mi) -> int:
        return runtime_steps(mi, self.mips, self.u, self.dt)

    def latest_start(self, t: int, ttd_sec: float, present: bool, r: int) -> int:
        """Last step at which the job can start and still clear the deadline with the frozen
        eps: D - (r + eps); no deadline -> the horizon minus its runtime."""
        if not present:
            return self.T - r - 1
        deadline = int(t) + int(np.floor(float(ttd_sec) / self.dt))
        return deadline - (int(r) + self.eps)

    def feasible(self, d: int, s: int, r: int, p: float) -> bool:
        if s < 0 or s + r > self.T:
            return False
        return bool(np.all(self.occ[d, s:s + r] + p <= self.cap[d]))

    def fallback_start(self, d: int, t: int, latest: int, r: int, p: float) -> Optional[int]:
        """Latest feasible start in [t + lag, latest] at site d, or None."""
        lo = int(t) + self.lag
        for s in range(int(latest), lo - 1, -1):
            if self.feasible(d, s, r, p):
                return s
        return None

    def _hold(self, d, s, e, p):
        self.occ[d, max(0, s):max(0, min(self.T, e))] += p

    def _free(self, d, s, e, p):
        self.occ[d, max(0, s):max(0, min(self.T, e))] -= p

    # ── legality ─────────────────────────────────────────────────────────────────────
    def hold_allowed(self, t: int, ids, pes, mi, ttd, present, deadline_allowed) -> np.ndarray:
        """(NB, n) float32: 1 iff the slot holds a real job, the deadline rule allows a wait,
        and a fallback reservation exists at d on the current grid."""
        ids = np.asarray(ids).reshape(-1)
        nb = ids.shape[0]
        out = np.zeros((nb, self.n), dtype=np.float32)
        for j in range(nb):
            if int(ids[j]) < 0 or float(pes[j]) <= 0 or float(mi[j]) <= 0:
                continue
            if float(deadline_allowed[j]) < 0.5:
                continue
            r = self.runtime(mi[j])
            latest = self.latest_start(t, ttd[j], bool(float(present[j]) >= 0.5), r)
            for d in range(self.n):
                if self.fallback_start(d, t, latest, r, float(pes[j])) is not None:
                    out[j, d] = 1.0
        return out

    # ── lifecycle ────────────────────────────────────────────────────────────────────
    def note_route(self, jid: int, d: int, t: int, pes, mi):
        """Any route seen by the env (ROUTE_NOW of any arm, or a release) occupies the grid
        from t + lag for its runtime."""
        r = self.runtime(mi)
        s = int(t) + self.lag
        self._hold(d, s, s + r, float(pes))
        self.active[int(jid)] = (int(d), s, s + r, int(pes))
        self.n_route_seen += 1

    def create(self, jid: int, d: int, t: int, pes, mi, ttd, present) -> bool:
        """HOLD(d) at step t. Books the fallback reservation; False (refused) when none fits."""
        r = self.runtime(mi)
        latest = self.latest_start(t, ttd, bool(present), r)
        s_f = self.fallback_start(d, t, latest, r, float(pes))
        if s_f is None:
            self.n_refused += 1
            return False
        self._hold(d, s_f, s_f + r, float(pes))
        self.held[int(jid)] = HeldJob(int(jid), int(d), int(pes), float(mi), r, int(t), latest, s_f)
        self.n_created += 1
        return True

    def releases(self, t: int, green_now_w, free_pes) -> List[Tuple[int, int, str]]:
        """Decision point t, before the batch: which held jobs start now and why."""
        green = np.asarray(green_now_w, dtype=np.float64).reshape(self.n)
        free = np.asarray(free_pes, dtype=np.float64).reshape(self.n).copy()
        s_now = int(t) + self.lag
        out = []
        for jid in sorted(self.held, key=lambda i: (self.held[i].latest, i)):
            h = self.held[jid]
            d, p = h.dc, float(h.pes)
            reason = None
            if s_now >= h.s_f:
                reason = REASON_MARGIN
            else:
                occ_at = self.occ[d, s_now] if s_now < self.T else self.cap[d]
                if (residual_green(green[d], self.static[d], occ_at, self.dyn, self.u) >= p * self.dyn * self.u
                        and free[d] >= p and self.feasible(d, s_now, h.r, p)):
                    reason = REASON_GREEN
            if reason is None:
                continue
            # the reservation becomes the execution (margin) or moves to now (green)
            self._free(d, h.s_f, h.s_f + h.r, p)
            self._hold(d, s_now, s_now + h.r, p)
            free[d] -= p
            self.active[jid] = (d, s_now, s_now + h.r, h.pes)
            h.t_release, h.reason = int(t), reason
            if reason == REASON_GREEN:
                self.n_term_green += 1
            else:
                self.n_term_margin += 1
            self.done[jid] = h
            del self.held[jid]
            out.append((jid, d, reason))
        return out

    # ── (DC, dispatch-offset) fallback (Addenda A5, C1–C2) ──────────────────────────
    # κ is the interval from creation to the route call; the job starts at t + κ + lag on
    # the grid. Legality (deadline, capacity) is decided at creation; the release depends
    # on t_c + κ only, never on green, occupancy or deadline.
    def offset_legal(self, d: int, t: int, kappa: int, r: int, p: float, latest: int) -> bool:
        s = int(t) + int(kappa) + self.lag
        return s <= int(latest) and self.feasible(d, s, r, p)

    def offset_allowed(self, t: int, ids, pes, mi, ttd, present, grid) -> np.ndarray:
        """(NB, n * |K|) float32, index d * |K| + i for κ = grid[i]: 1 iff the slot holds a
        real job and (d, κ) is legal now. Reads no green."""
        ids = np.asarray(ids).reshape(-1)
        nb, K = ids.shape[0], len(grid)
        out = np.zeros((nb, self.n * K), dtype=np.float32)
        for j in range(nb):
            if int(ids[j]) < 0 or float(pes[j]) <= 0 or float(mi[j]) <= 0:
                continue
            r = self.runtime(mi[j])
            latest = self.latest_start(t, ttd[j], bool(float(present[j]) >= 0.5), r)
            for d in range(self.n):
                for i, k in enumerate(grid):
                    if self.offset_legal(d, t, k, r, float(pes[j]), latest):
                        out[j, d * K + i] = 1.0
        return out

    def create_fixed(self, jid: int, d: int, t: int, kappa: int, pes, mi, ttd, present) -> bool:
        """HOLD with a fixed dispatch offset κ > 0: reservation at t + κ + lag; False when
        (d, κ) is not legal (nothing is clipped)."""
        r = self.runtime(mi)
        latest = self.latest_start(t, ttd, bool(present), r)
        if int(kappa) <= 0 or not self.offset_legal(d, t, kappa, r, float(pes), latest):
            self.n_refused += 1
            return False
        s = int(t) + int(kappa) + self.lag
        self._hold(d, s, s + r, float(pes))
        h = HeldJob(int(jid), int(d), int(pes), float(mi), r, int(t), latest, s)
        h.extra["kappa"] = int(kappa)
        self.held[int(jid)] = h
        self.n_created += 1
        return True

    def releases_fixed(self, t: int) -> List[Tuple[int, int, str]]:
        """Decision point t: release every held job whose t_c + κ has arrived. Time only."""
        out = []
        for jid in sorted(self.held, key=lambda i: (self.held[i].t_c + self.held[i].extra.get("kappa", 0), i)):
            h = self.held[jid]
            if h.t_c + int(h.extra.get("kappa", 0)) > int(t):
                continue
            h.t_release, h.reason = int(t), REASON_OFFSET
            self.active[jid] = (h.dc, h.s_f, h.s_f + h.r, h.pes)
            self.n_term_margin += 1
            self.done[jid] = h
            del self.held[jid]
            out.append((jid, h.dc, REASON_OFFSET))
        return out

    def record_release_reward(self, jid: int, r: float):
        h = self.done.get(int(jid))
        if h is not None:
            h.r_release = float(r) if r is not None else None

    # ── observation and ledger ───────────────────────────────────────────────────────
    def observation(self, t: int, no_hold_margin: float):
        """Per site: held count, held PEs, tightest margin (min latest - t; `no_hold_margin`
        when nothing is held)."""
        cnt = np.zeros(self.n, dtype=np.float32)
        pes = np.zeros(self.n, dtype=np.float32)
        tight = np.full(self.n, float(no_hold_margin), dtype=np.float32)
        for h in self.held.values():
            cnt[h.dc] += 1.0
            pes[h.dc] += float(h.pes)
            tight[h.dc] = min(tight[h.dc], float(h.latest - int(t)))
        return cnt, pes, tight

    def rows(self, start_times: Optional[Dict[int, float]] = None, clock0: float = 0.0) -> List[dict]:
        """Ledger rows, one per created option, with the simulator's execution start when
        known (t_s in steps from the clock origin) and k = t_s - t_c."""
        st = start_times or {}
        rows = []
        for jid, h in list(self.done.items()) + list(self.held.items()):
            ts = st.get(jid)
            t_s = None if ts is None else float(ts - clock0) / self.dt
            rows.append({"id": jid, "dc": h.dc, "pes": h.pes, "mi": h.mi, "runtime_steps": h.r,
                         "t_c": h.t_c, "latest": h.latest, "s_f": h.s_f, "kappa": h.extra.get("kappa", ""),
                         "t_release": h.t_release, "reason": h.reason, "r_release": h.r_release,
                         "t_s": t_s, "k": (None if t_s is None else t_s - h.t_c),
                         "route_to_start_steps": (None if t_s is None or h.t_release is None
                                                   else t_s - h.t_release),
                         "stale": h.t_release is None})
        rows.sort(key=lambda x: x["id"])
        return rows

    def counters(self) -> dict:
        return {"opt_created": self.n_created, "opt_refused": self.n_refused,
                "opt_term_green": self.n_term_green, "opt_term_margin": self.n_term_margin,
                "opt_held_open": len(self.held), "opt_routes_seen": self.n_route_seen}
