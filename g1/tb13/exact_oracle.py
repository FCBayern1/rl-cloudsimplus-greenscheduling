"""Exact constrained scheduling model for TB13 screening (CP-SAT).

Codex 2026-09-01: a dominance-safe oracle cannot be a heuristic comparing its own internal
estimates, because that model can still miss packing, queueing or static energy. It has to
be a constrained optimum over carbon with the nowait schedule admitted as feasible, so

    C_optimum <= C_nowait

holds by construction. Waiting is made scarce by a registered delay budget rather than by
destroying work: no job is dropped, completion and punctuality stay contracted, and the
shadow price of the budget pushes the decision boundary into the interior.

Green is settled on aggregate load per site and epoch, never per job. An earlier version
asked each job separately how much of the site's residual green covered it, so two
concurrent jobs at one site both claimed the same watts. On a site with 20 W spare and two
20 W jobs that model returned zero carbon where the correct aggregate answer is 13.33.

    L[d,t] = static[d] + dyn * sum_{i active at (d,t)} p_i
    B[d,t] = max(L[d,t] - G[d,t], 0)
    carbon = sum_{d,t} [ cb_d * B[d,t] + cg_d * (L[d,t] - B[d,t]) ] * epoch_hours

Scope of the word exact: static power is modelled as fixed per site and independent of the
schedule. Where host power-down makes static depend on which hosts are up, this is an
offline abstraction rather than exact terminal carbon, and the dominance relation must be
confirmed in the real simulator. See `STATIC_IS_SCHEDULE_INDEPENDENT`.
"""
from __future__ import annotations

import numpy as np
from ortools.sat.python import cp_model

STATIC_IS_SCHEDULE_INDEPENDENT = True   # flip and rename the claim if host power-down is modelled
W_SCALE = 1000                          # watts -> milliwatts, keeps CP-SAT integral
EPOCH_HOURS = 600.0 / 3600.0


class Scenario:
    """A TB13 candidate. All times are in wind rows; one row is one 600 s epoch."""

    def __init__(self, green_w, static_w, brown_factor, green_factor,
                 cap_pes, arrival, runtime, pes, deadline, dyn_w_per_pe,
                 per_job_wait_max, budget_total):
        self.green = np.asarray(green_w, dtype=float)
        self.static = np.asarray(static_w, dtype=float)
        self.cb = np.asarray(brown_factor, dtype=float)
        self.cg = np.asarray(green_factor, dtype=float)
        self.cap = np.asarray(cap_pes, dtype=int)
        self.a = np.asarray(arrival, dtype=int)
        self.r = np.asarray(runtime, dtype=int)
        self.p = np.asarray(pes, dtype=int)
        self.dl = np.asarray(deadline, dtype=int)
        self.dyn = float(dyn_w_per_pe)
        self.wmax = int(per_job_wait_max)
        self.B = int(budget_total)
        self.n_dc, self.T = self.green.shape
        self.n = len(self.a)
        self._validate()

    def _validate(self):
        """Reject inputs the model cannot represent, rather than solving the wrong thing.

        brown >= load - green is only pushed to its lower bound because the objective
        weight on brown exceeds the weight on green. With cb < cg the solver would be
        rewarded for inflating brown and the relaxation would be unsound.
        """
        for name, arr, n in (("static_w", self.static, self.n_dc),
                             ("brown_factor", self.cb, self.n_dc),
                             ("green_factor", self.cg, self.n_dc),
                             ("cap_pes", self.cap, self.n_dc),
                             ("arrival", self.a, self.n), ("runtime", self.r, self.n),
                             ("pes", self.p, self.n), ("deadline", self.dl, self.n)):
            if arr.shape != (n,):
                raise ValueError(f"{name} has shape {arr.shape}, expected {(n,)}")
        if np.any(self.cb < self.cg):
            raise ValueError(
                f"brown_factor must be at least green_factor for the brown relaxation to "
                f"be tight; got cb={self.cb.tolist()} cg={self.cg.tolist()}")
        if np.any(self.cap < 0) or np.any(self.r < 1) or np.any(self.p < 1):
            raise ValueError("capacity must be non-negative, runtime and PES at least one")
        if np.any(self.a < 0) or np.any(self.a >= self.T):
            raise ValueError(f"arrivals must lie inside the horizon 0..{self.T - 1}")
        if np.any(self.dl > self.T) or np.any(self.dl < self.a + self.r):
            raise ValueError("deadlines must be inside the horizon and reachable")
        if self.wmax < 0 or self.B < 0:
            raise ValueError("the delay budget and per-job wait cap must be non-negative")

    def latest_start(self, i):
        return min(self.dl[i] - self.r[i], self.T - self.r[i], self.a[i] + self.wmax)

    def starts(self, i):
        return range(int(self.a[i]), int(self.latest_start(i)) + 1)

    def carbon_of(self, assign):
        """Aggregate carbon of a full assignment {job: (site, start)}, the ground truth."""
        load = np.tile(self.static.reshape(-1, 1), (1, self.T)).astype(float)
        for i, (d, s) in assign.items():
            load[d, s:s + self.r[i]] += self.p[i] * self.dyn
        brown = np.maximum(load - self.green, 0.0)
        green = load - brown
        return float(np.sum(self.cb.reshape(-1, 1) * brown
                            + self.cg.reshape(-1, 1) * green)) * EPOCH_HOURS


def validate_assignment(sc: Scenario, assign, budget=None):
    """Check an externally produced schedule against every constraint the model imposes.

    Used to prove that the nowait schedule really is a feasible candidate, rather than
    inferring it from a solver run that was free to move jobs to another site.
    Returns (ok, reasons).
    """
    reasons = []
    if set(assign) != set(range(sc.n)):
        reasons.append("not every job is assigned exactly once")
    for i, (d, s) in assign.items():
        if not (0 <= d < sc.n_dc):
            reasons.append(f"job {i}: site {d} does not exist")
        if s not in sc.starts(i):
            reasons.append(f"job {i}: start {s} outside the candidate set "
                           f"{sc.a[i]}..{sc.latest_start(i)}")
        if s + sc.r[i] > sc.dl[i]:
            reasons.append(f"job {i}: finishes after its deadline")
        if s - sc.a[i] > sc.wmax:
            reasons.append(f"job {i}: waits {s - sc.a[i]} beyond the per-job cap {sc.wmax}")
    used = np.zeros((sc.n_dc, sc.T), dtype=float)
    for i, (d, s) in assign.items():
        if 0 <= d < sc.n_dc:
            used[d, s:s + sc.r[i]] += sc.p[i]
    for d in range(sc.n_dc):
        if np.any(used[d] > sc.cap[d] + 1e-9):
            reasons.append(f"site {d} exceeds capacity {sc.cap[d]}")
    total_wait = sum(int(s - sc.a[i]) for i, (_d, s) in assign.items())
    cap_budget = sc.B if budget is None else budget
    if total_wait > cap_budget:
        reasons.append(f"total wait {total_wait} exceeds the budget {cap_budget}")
    return (not reasons), reasons


def _empty_result(status):
    return {"carbon_status": status, "wait_status": "NOT_RUN", "carbon": None,
            "assign": None, "carbon_gap": None, "carbon_bound": None,
            "wait_gap": None, "wait_bound": None, "total_wait": None,
            "exact": False, "wait_exact": False}


def _build(sc: Scenario):
    m = cp_model.CpModel()
    x = {(i, d, s): m.NewBoolVar(f"x_{i}_{d}_{s}")
         for i in range(sc.n) for d in range(sc.n_dc) for s in sc.starts(i)}
    for i in range(sc.n):
        opts = [x[(i, d, s)] for d in range(sc.n_dc) for s in sc.starts(i)]
        if not opts:
            return None          # no admissible (site, start) for this job
        m.AddExactlyOne(opts)

    dyn_s = int(round(sc.dyn * W_SCALE))
    terms = []
    for d in range(sc.n_dc):
        static_s = int(round(sc.static[d] * W_SCALE))
        for t in range(sc.T):
            active = [sc.p[i] * x[(i, d, s)]
                      for i in range(sc.n) for s in sc.starts(i)
                      if s <= t < s + sc.r[i]]
            # Aggregate load at this site and epoch, settled once for every job on it.
            load = m.NewIntVar(0, static_s + dyn_s * int(sc.cap[d]) + 1, f"L_{d}_{t}")
            m.Add(load == static_s + (dyn_s * sum(active) if active else 0))
            if active:
                m.Add(sum(active) <= int(sc.cap[d]))
            g_s = int(round(sc.green[d, t] * W_SCALE))
            brown = m.NewIntVar(0, static_s + dyn_s * int(sc.cap[d]) + 1, f"B_{d}_{t}")
            m.Add(brown >= load - g_s)
            # cb > cg makes the objective push brown to its lower bound max(load - g, 0).
            terms.append((sc.cg[d], load))
            terms.append((sc.cb[d] - sc.cg[d], brown))
    return m, x, terms


def _objective(m, terms, scale):
    return sum(int(round(c * scale)) * v for c, v in terms)


def solve(sc: Scenario, time_limit_s=30.0, scale=1000, log=False, lexicographic=True,
          pin=None):
    """Minimum aggregate carbon, then minimum total wait at that carbon.

    Returns a dict. Exact screening accepts only status OPTIMAL: a merely FEASIBLE answer
    is recorded as UNRESOLVED and must not enter a pass or fail decision, because the true
    optimum could be anywhere below the incumbent. The two stages report separately, so a
    run may have an optimal carbon and an unresolved minimum wait, and only the latter
    weakens the claim about waiting.
    """
    built = _build(sc)
    if built is None:
        return _empty_result("INFEASIBLE_JOB")
    m, x, terms = built
    waits = sum((s - int(sc.a[i])) * x[(i, d, s)]
                for i in range(sc.n) for d in range(sc.n_dc) for s in sc.starts(i))
    m.Add(waits <= sc.B)
    if pin is not None:
        # Fix the complete (site, start) of every job. A partial pin would leave the rest
        # free to move, which is not a test that the given schedule is admissible, so an
        # incomplete or infeasible pin is refused rather than silently relaxed.
        if set(pin) != set(range(sc.n)):
            return _empty_result("PIN_INVALID")
        ok, _why = validate_assignment(sc, pin)
        if not ok:
            return _empty_result("PIN_INVALID")
        for i, (d, s) in pin.items():
            if (i, d, s) not in x:
                return _empty_result("PIN_INVALID")
            m.Add(x[(i, d, s)] == 1)

    obj = _objective(m, terms, scale)
    m.Minimize(obj)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_search_workers = 4
    solver.parameters.log_search_progress = log
    st = solver.Solve(m)
    out = _empty_result(solver.StatusName(st))
    if st not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return out
    best = solver.ObjectiveValue()
    # The internal objective accumulates carbon-factor x milliwatts x scale, so the bound
    # needs the same two divisions and the epoch length to sit beside `carbon`.
    out["carbon_bound"] = (solver.BestObjectiveBound()
                           / (scale * W_SCALE) * EPOCH_HOURS)
    out["carbon_gap"] = (abs(best - solver.BestObjectiveBound()) / max(abs(best), 1e-12)
                         if best else 0.0)
    out["exact"] = st == cp_model.OPTIMAL
    if st != cp_model.OPTIMAL:
        out["carbon_status"] = "UNRESOLVED"

    if lexicographic and st == cp_model.OPTIMAL:
        # Same carbon, least waiting: waiting is scarce, not a free action. The carbon is
        # pinned with equality; a <= on a rounded float could admit a cheaper schedule that
        # the first stage had already proved impossible.
        m.Add(obj == int(round(best)))
        m.Minimize(waits)
        s2 = cp_model.CpSolver()
        s2.parameters.max_time_in_seconds = time_limit_s
        s2.parameters.num_search_workers = 4
        st2 = s2.Solve(m)
        out["wait_status"] = s2.StatusName(st2)
        if st2 in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            out["wait_bound"] = s2.BestObjectiveBound()
            out["wait_gap"] = (abs(s2.ObjectiveValue() - s2.BestObjectiveBound())
                               / max(abs(s2.ObjectiveValue()), 1e-12)
                               if s2.ObjectiveValue() else 0.0)
            out["wait_exact"] = st2 == cp_model.OPTIMAL
            if st2 != cp_model.OPTIMAL:
                out["wait_status"] = "UNRESOLVED"
            solver = s2

    assign = {i: (d, s) for (i, d, s), v in x.items() if solver.Value(v)}
    out["assign"] = assign
    out["carbon"] = sc.carbon_of(assign)
    out["total_wait"] = sum(int(s - sc.a[i]) for i, (_d, s) in assign.items())
    return out


def nowait_schedule(sc: Scenario):
    """Immediate dispatch at the site that is cheapest given what is already placed.

    Returns (carbon, assignment) or (None, None) when instantaneous capacity makes
    immediate dispatch infeasible. A scenario where this returns None must not be used to
    argue that nowait is a feasibility witness for the delay budget.
    """
    load = np.tile(sc.static.reshape(-1, 1), (1, sc.T)).astype(float)
    assign = {}
    for i in sorted(range(sc.n), key=lambda j: (sc.a[j], j)):
        s = int(sc.a[i])
        if s + sc.r[i] > sc.T:
            return None, None
        best = None
        for d in range(sc.n_dc):
            used = (load[d, s:s + sc.r[i]] - sc.static[d]) / sc.dyn
            if np.any(used + sc.p[i] > sc.cap[d] + 1e-9):
                continue
            trial = load.copy()
            trial[d, s:s + sc.r[i]] += sc.p[i] * sc.dyn
            brown = np.maximum(trial - sc.green, 0.0)
            c = float(np.sum(sc.cb.reshape(-1, 1) * brown
                             + sc.cg.reshape(-1, 1) * (trial - brown))) * EPOCH_HOURS
            if best is None or c < best[0]:
                best = (c, d)
        if best is None:
            return None, None
        d = best[1]
        load[d, s:s + sc.r[i]] += sc.p[i] * sc.dyn
        assign[i] = (d, s)
    return sc.carbon_of(assign), assign
