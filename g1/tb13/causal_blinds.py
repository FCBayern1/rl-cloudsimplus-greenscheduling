"""Causal blind policies for TB13 screening.

Codex 2026-09-01: `nowait_schedule` in the oracle module chooses its site by evaluating
carbon over the job's whole runtime, which reads wind the policy could not know. It stays
as the feasibility and dominance witness under the name `immediate_dispatch_witness`, but
it is not a blind and cannot stand in for C_strongest_blind.

Every policy here is simulated forward epoch by epoch and may read only

    the wind already observed, up to and including the current epoch
    a climatology calibrated strictly before the window
    its own placement decisions

None of them look at G[d, t'] for t' > now. Capacity, per-job wait caps and the delay
budget are enforced online exactly as the optimiser enforces them offline.
"""
from __future__ import annotations

import numpy as np

from exact_oracle import EPOCH_HOURS, Scenario


def _out(carbon, assign, diagnose, reason, at_t, job, n_pending, **detail):
    """Uniform return. Without `diagnose` the pair is exactly what it always was.

    The diagnostic channel tells a scenario that was never schedulable apart from a blind
    that was simply too weak. It never touches carbon.
    """
    if not diagnose:
        return carbon, assign
    d = {"reason": reason, "at_epoch": at_t, "job": job,
         "pending_at_failure": n_pending}
    d.update(detail)
    return carbon, assign, d


def _residual(sc: Scenario, load, d, t):
    """Green left at site d and epoch t after everything already placed there."""
    return max(0.0, sc.green[d, t] - load[d, t])


def _instant_rate(sc: Scenario, load, d, t, draw):
    """Carbon per hour of adding `draw` watts at site d, epoch t, given current load."""
    before_brown = max(0.0, load[d, t] - sc.green[d, t])
    after_brown = max(0.0, load[d, t] + draw - sc.green[d, t])
    d_brown = after_brown - before_brown
    d_green = draw - d_brown
    return sc.cb[d] * d_brown + sc.cg[d] * d_green


def _run(sc: Scenario, decide, climatology=None, diagnose=False):
    """Drive one policy forward. `decide(job, t, load, ctx, feasible, draw, must) -> site|None`.

    The delay budget is settled on the running total, not on each job at the moment it
    happens to leave. An earlier version debited a job's wait only when it was dispatched,
    so several jobs could sit waiting at once without any of them counting, and the final
    sum could exceed the budget. At every epoch the committed spend is

        sum over dispatched (s_i - a_i)  +  sum over still pending (t - a_i)

    and the second term is a lower bound on what those jobs will eventually cost, so a
    policy that lets it reach the budget must release work now.
    """
    load = np.tile(sc.static.reshape(-1, 1), (1, sc.T)).astype(float)
    assign = {}
    pending = sorted(range(sc.n), key=lambda i: (sc.a[i], i))
    spent_dispatched = 0
    ctx = {"climatology": climatology}
    for t in range(sc.T):
        ready = [i for i in pending if sc.a[i] <= t]
        ready.sort(key=lambda i: (sc.latest_start(i), i))
        for i in ready:
            latest = sc.latest_start(i)
            draw = sc.p[i] * sc.dyn
            feasible = []
            for d in range(sc.n_dc):
                if t + sc.r[i] > sc.T:
                    continue
                used = (load[d, t:t + sc.r[i]] - sc.static[d]) / sc.dyn
                if np.any(used + sc.p[i] > sc.cap[d] + 1e-9):
                    continue
                feasible.append(d)

            # Cheapest possible total if this job waits one more epoch: everything
            # already dispatched, plus one more epoch of waiting for every arrived job
            # still pending. Unarrived jobs contribute at least zero, so this is a valid
            # lower bound. Exceeding the budget means the job has to leave now.
            floor_if_wait = spent_dispatched + sum(
                (t + 1) - int(sc.a[j]) for j in pending if sc.a[j] <= t)
            by_deadline = t >= latest
            by_budget = floor_if_wait > sc.B
            must = by_deadline or by_budget

            if not feasible:
                if must:
                    # Why the job could not wait matters as much as that it could not be
                    # placed: a deadline and an exhausted delay budget are different
                    # scenario faults, and "no feasible site" alone conflates them.
                    free = [int(sc.cap[d] - ((load[d, t] - sc.static[d]) / sc.dyn))
                            for d in range(sc.n_dc)]
                    return _out(None, None, diagnose, "no_feasible_site_when_forced",
                                t, i, len(pending),
                                must_reason=("both" if by_deadline and by_budget
                                             else "deadline" if by_deadline else "budget"),
                                floor_if_wait=int(floor_if_wait), budget=int(sc.B),
                                latest=int(latest), free_pes=free,
                                need_pes=int(sc.p[i]), runtime=int(sc.r[i]))
                continue
            site = decide(i, t, load, ctx, feasible, draw, must)
            if site is None:
                continue
            load[site, t:t + sc.r[i]] += draw
            assign[i] = (site, t)
            spent_dispatched += t - int(sc.a[i])
            pending.remove(i)
        if not pending:
            break
    if pending:
        return _out(None, None, diagnose, "pending_at_horizon_end",
                    sc.T, int(pending[0]), len(pending))
    if sum(s - int(sc.a[i]) for i, (_d, s) in assign.items()) > sc.B:
        return _out(None, None, diagnose, "budget_exceeded_at_end", sc.T, None, 0)
    return _out(sc.carbon_of(assign), assign, diagnose, None, None, None, 0)


def immediate_current_only(sc: Scenario, diagnose=False):
    """Dispatch on arrival, ranking sites by the carbon rate visible at this instant."""
    def decide(i, t, load, ctx, feasible, draw, must):
        return min(feasible, key=lambda d: (_instant_rate(sc, load, d, t, draw), d))
    return _run(sc, decide, diagnose=diagnose)


def persistence(sc: Scenario, diagnose=False):
    """Assume the wind stays at what it reads now, and plan against that flat future.

    Under a flat future every start is priced alike, so waiting can never help and this
    policy is behaviourally identical to `immediate_current_only`. It is kept as a named
    arm because the planner family carries the same distinction, and the equivalence is
    asserted in the tests rather than left implicit.
    """
    def decide(i, t, load, ctx, feasible, draw, must):
        now_best = min(feasible, key=lambda d: (_instant_rate(sc, load, d, t, draw), d))
        if must:
            return now_best
        # Under persistence the future looks exactly like now, so waiting can never help.
        return now_best
    return _run(sc, decide, diagnose=diagnose)


def climatology(sc: Scenario, clim_residual_green, diagnose=False):
    """Believe the wind returns to a level calibrated strictly before the window.

    The level passed in is RESIDUAL green, with the site's static draw already removed by
    `instance_gen._climatology`. Subtracting static again here would double-count it, so
    the cost of a placement under the climatological view is simply

        cb * max(draw - clim_residual, 0) + cg * min(draw, clim_residual)
    """
    clim = np.asarray(clim_residual_green, dtype=float)

    def decide(i, t, load, ctx, feasible, draw, must):
        now = min(feasible, key=lambda d: (_instant_rate(sc, load, d, t, draw), d))
        if must:
            return now
        rate_now = _instant_rate(sc, load, now, t, draw)
        best_clim = min(sc.cb[d] * max(0.0, draw - clim[d])
                        + sc.cg[d] * min(draw, max(0.0, clim[d]))
                        for d in range(sc.n_dc))
        return None if best_clim < rate_now - 1e-12 else now
    return _run(sc, decide, climatology=clim, diagnose=diagnose)


def reactive_wait(sc: Scenario, diagnose=False):
    """Go when the green already on the meter covers the job, otherwise wait."""
    def decide(i, t, load, ctx, feasible, draw, must):
        covered = [d for d in feasible if _residual(sc, load, d, t) >= draw - 1e-12]
        if covered:
            return min(covered, key=lambda d: (_instant_rate(sc, load, d, t, draw), d))
        if must:
            return min(feasible, key=lambda d: (_instant_rate(sc, load, d, t, draw), d))
        return None
    return _run(sc, decide, diagnose=diagnose)


def reservation_edf_blind(sc: Scenario, diagnose=False):
    """Earliest-deadline-first with irrevocable reservations, registered in v2 section 5.

    The policy is the one `schedule_feasibility.reservation_edf` implements, whose source
    may not read a weather or an emissions field at all. It is in the candidate set to
    guarantee at least one contract-safe member, and it wins only if pooled carbon says so.
    """
    import schedule_feasibility as sf

    w = {"arrival": sc.a, "runtime": sc.r, "pes": sc.p, "deadline": sc.dl,
         "horizon": sc.T, "wait_cap": sc.wmax}
    assign, _spent = sf.reservation_edf(w, sc.B)
    if assign is None or len(assign) != sc.n:
        return _out(None, None, diagnose, "reservation_edf_found_no_schedule",
                    sc.T, None, sc.n)
    return _out(sc.carbon_of(assign), assign, diagnose, None, None, None, 0)


BLINDS = {
    "immediate_current_only": lambda sc, clim, **kw: immediate_current_only(sc, **kw),
    "persistence": lambda sc, clim, **kw: persistence(sc, **kw),
    "climatology": lambda sc, clim, **kw: climatology(sc, clim, **kw),
    "reactive_wait": lambda sc, clim, **kw: reactive_wait(sc, **kw),
    "reservation_edf_blind": lambda sc, clim, **kw: reservation_edf_blind(sc, **kw),
}


def blind_class_diagnostic(sc: Scenario, clim_level):
    """Per-instance best blind. A diagnostic, NOT a deployable policy.

    Picking the winner per instance uses the realised carbon of each arm, which no policy
    could do in advance. It gives a conservative lower envelope of the blind class and is
    reported as such. The blind used in a verdict must be a single arm frozen on DISCOVERY
    by pooled carbon and reused unchanged on CONFIRMATION; see `pooled_strongest`.
    """
    table = {}
    for name, fn in BLINDS.items():
        c, a = fn(sc, clim_level)
        table[name] = None if c is None else c
        if c is not None:
            table[name] = c
    valid = {k: v for k, v in table.items() if v is not None}
    if not valid:
        return None, None, None, table
    best = min(valid, key=lambda k: valid[k])
    c, a = BLINDS[best](sc, clim_level)
    return best, c, a, table


def pooled_strongest(instances):
    """Freeze one blind arm by pooled carbon over a set of DISCOVERY instances.

    `instances` is a sequence of (Scenario, climatology). An arm that fails to honour the
    contract on any instance is disqualified outright rather than averaged around.
    """
    totals = {name: 0.0 for name in BLINDS}
    disqualified = set()
    for sc, clim in instances:
        for name, fn in BLINDS.items():
            if name in disqualified:
                continue
            c, _a = fn(sc, clim)
            if c is None:
                disqualified.add(name)
            else:
                totals[name] += c
    live = {k: v for k, v in totals.items() if k not in disqualified}
    if not live:
        return None, totals, disqualified
    return min(live, key=lambda k: live[k]), totals, disqualified
