# Ladder stage 1: CP-SAT cannot prove the truth rung optimal in 600 s; HiGHS on the identical model can — ruling on the solver text (2026-09-06 02:20)

Context: ERROR_LADDER_PLANNER_PREREG.md frozen at 24b5de60 (body < A < B < C). §2.1 names the solver: "CP-SAT (ortools 9.15, installed)". Nothing about the model, the optimality requirement, the 600 s limit, the closure or the gates is touched here.

## What happened (stage 1, truth rung only; no other rung's carbon produced)

- Stage-1 run 1: every truth solve returned INFEASIBLE within seconds. Instrument bug on my side: the decision dump recorded the observation's normalised time-to-deadline (120 s → 0.017) instead of raw seconds; every job's latest start came out negative. Fixed (the dump now records `ttd_sec` from the planner channel, tested), disclosed, rerun.
- Stage-1 run 2 (CP-SAT, default parameters, 8 workers, 600 s per window): running while this note is written; the 30 s probe on window k0 shows why it will end STOP_SOLVER_RUNG_UNRESOLVED: after 30 s the incumbent is 2.33 × 10^8 with bound 1.14 × 10^8 (gap ≈ 50 %); with `linearization_level = 2` the gap after 120 s (4 workers) is 11.5 %. The default CP-SAT search does not close this time-indexed model.
- The identical model as a MIP (HiGHS through `scipy.optimize.milp`, `mip_rel_gap = 0`, same binaries, same integer host variables, brown continuous but integral at optimality since every coefficient is integer) proves **OPTIMAL on k0 in 362 s** (while sharing the CPU with the running CP-SAT stage), objective 222,043,518 = the CP-SAT lin2 incumbent. The two formulations are tested to agree on small instances (`test_milp_agrees_with_cpsat_on_small_instances`).

## What I ask

1. Record stage-1 run 2 as STOP_SOLVER_RUNG_UNRESOLVED under the frozen solver text when it finishes (expected), with the two probes above as the diagnosis.
2. Allow an Addendum D that changes only the solver sentence of §2.1: "the exact model is solved by HiGHS (scipy.optimize.milp, relative MIP gap 0, 600 s); a rung counts as OPTIMAL only when HiGHS reports optimal; CP-SAT with linearization_level 2 is run as a cross-check where it finishes within the limit and must agree on the objective". Model, OPTIMAL requirement, limit, closure, gates unchanged.
3. Confirm that the 600 s limit applies per (window, rung) solve, and whether it should be measured with the machine otherwise idle (the 362 s probe was under contention).

No rung other than truth has been solved or read; the six-window dumps and the truth incumbents are in `stage_a_out/ladder/`, nothing archived as a result yet.
