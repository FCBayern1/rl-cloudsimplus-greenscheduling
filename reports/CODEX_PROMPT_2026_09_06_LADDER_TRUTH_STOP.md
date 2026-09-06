# Ladder truth stage under Addendum D: HiGHS proves k0 but not k1 within 600 s → STOP_SOLVER_RUNG_UNRESOLVED; ruling on the solver budget needed (2026-09-06 02:30, written while k2–k5 still run)

Context: ERROR_LADDER_PLANNER_PREREG.md frozen at 24b5de60 with Addendum D (HiGHS the only judging solver; 600 s internal wall clock per (window, rung); OPTIMAL is the compound condition; a solve that reaches 600 s without proof is STOP_SOLVER_RUNG_UNRESOLVED at once, no extension, no retry). Machine otherwise idle (gradle daemon stopped, no evaluation or training), one solve at a time, fresh directory `stage_a_out/ladder_v2/`.

## The bad news

| window | offset | HiGHS truth rung | wall |
|---|---|---|---|
| k0 | 16477 | OPTIMAL (compound condition met), J = 222,043,518, C_model 0.000617 kg | 337 s |
| k1 | 4240 | FEASIBLE at the 600 s limit, no proof | 600 s |
| k2–k5 | | still running, each with the same 600 s limit; recorded when done | |

By D5 the truth rung is unresolved on k1, so the ladder stops before any other rung is generated. I am not extending, retrying or dropping the window. The formal record of this stage will be STOP_SOLVER_RUNG_UNRESOLVED once k2–k5 have run out their limits.

## What I did not touch and what I queued

- Nothing else in the ladder ran; no rung's carbon other than the truth model settlement exists.
- Queued after the formal stage, each labelled as what it is: (a) the D4 CP-SAT cross-check on the truth rungs; (b) the scene design's frozen steps 2b/2c (margin probe, P0′ analytic arms) on the development windows; (c) the §4.4 / A4 interface smoke (dense float32 first) on window k0; (d) a POST-RUN SOLVER DIAGNOSTIC: the unproven truth windows re-solved with a 3600 s limit in a scratch directory, to learn the actual proof times. (d) is not a result and is not archived as one.
- Earlier tonight, disclosed in the design log: stage-1 run 1 was invalid on a dump field (normalised deadline; fixed); CP-SAT run 2 interrupted on your ruling (k3 OPTIMAL in 294 s there; note that CP-SAT and HiGHS differ in which windows they close quickly).

## What I ask (Addendum E, before any further formal solve)

1. Confirm STOP_SOLVER_RUNG_UNRESOLVED as the frozen outcome of this stage.
2. Choose the continuation, all of which keep the model, the compound OPTIMAL condition and "no retry after the limit" intact:
   (a) a larger per-cell budget (e.g. 3600 s), set once from the diagnostic proof times, with the same no-extension rule at the new limit;
   (b) an equivalent tightening of the formulation (same feasible set and objective; e.g. per-site cumulative load expressed through aggregated running-job counts, or the LP-tightening cuts HiGHS accepts as constraints), verified on small instances to agree bitwise with the current model, then the same 600 s;
   (c) HiGHS with parallel threads through highspy (scipy's milp is single-threaded), same 600 s wall clock.
   My reading: (a) is the least manipulable, (b) the most likely to be decisive, (c) uncertain. I recommend (a) with the budget fixed from the diagnostic before it is read on any rung other than truth, and (b) only if (a) still leaves a window unproven.
3. Whether the diagnostic proof times may be read before the addendum is written (they are truth-rung solve times, not carbon), or whether the budget must be chosen blind.
