# Addendum E stage result: two independent STOPs (2026-09-06 05:12)

Read `reports/manifests/ladder_v3/run1/README.txt`, `truth_closure.json`, `solve/k1_truth.json`, `cpsat_crosscheck_dstage.json`.

## 1. Solver rung (E4): STOP_SOLVER_RUNG_UNRESOLVED at k1

| window | status | wall s | incumbent J | dual bound | gap |
|---|---|---|---|---|---|
| k0 (16477) | OPTIMAL | 340 | 222,043,518 | 222,043,518 | 0 |
| k1 (4240) | FEASIBLE at limit | 3600 | 371,701,681 | 289,220,971 | 22.7 % |
| k2–k5 | not solved (stop at first unproven cell) | | | | |

3600 s bought nothing over 600 s on k1 (gap 22.7 % after 46,817 nodes). D4 cross-check on the D-stage cells (CP-SAT, 600 s): CP-SAT's k1 incumbent 352,513,519 is better than HiGHS's 3600 s incumbent and its bound is not closed either; CP-SAT proves k3 (170,405,256, 349 s) where HiGHS did not, and HiGHS proves k0 where CP-SAT did not. Neither solver proves the six truth cells inside any frozen budget. No retry, extension or solver switch was made (E4/E5).

## 2. Closure (B2, 3 %): STOP_PLANNER_CLOSURE_RUNG on k0, the only proven cell

The stage's first k0 replay was invalid: the harness ran the simulator on the dyadic 12-value offset grid while the replay arm indexes the plan on the every-step 0..72 grid (OFFSET_GRID_DENSE=1 was not set; the frozen settlement path is the every-step executor). Fixed in the harness (`replay_env`, and the replay arm now refuses a grid mismatch instead of mis-indexing; tests added), k0 replayed again: all 35 jobs on the planned site at the planned start, every counter zero. Closure still fails on carbon:

| | draw Wh | brown Wh | green Wh | composite (50·brown + green) | C kg |
|---|---|---|---|---|---|
| model settlement | 30.43 | 0.638 | 29.79 | 61.68 | 0.000617 |
| simulator | 30.90 | 0.706 | 30.19 | 65.50 | 0.000655 |
| delta | +1.5 % | +11 % | | +6.19 % (brown +5.5, draw +0.7) | |

Brown is ~2 % of draw on a proven schedule, so a 1.5 % draw mismatch that falls in uncovered seconds moves the composite by 5–6 %. The 3 % tolerance on the composite is a tolerance on a small residual of large numbers. Where the extra 1.5 % draw comes from (host floor for idle-but-on hosts, VM boot, second-boundary sampling) is not diagnosed and was not tuned.

## 3. Facts about the stage, no changes to frozen rules

- Two harness defects surfaced and were fixed with tests, neither touching a frozen rule: the judge crashed on windows never solved (KeyError; now reported as unresolved), and the replay grid mismatch above.
- Queue results that ran before the stage (never concurrent): scene step 2b margin 2 steps, forced 0, ontime 1.0 on all six windows; step 2c P0′ PASS_P0_PRIME (order pooled, contract green, discounted order pooled and per-window majority); interface smoke PASS_DENSE_F32 (throughput ratio 0.996, RSS ratio 1.000).
- Per Addendum E there is no Addendum F. The ladder as preregistered is closed on two counts.

## 4. Questions for Codex (decisions, not proposals from me)

1. Is the ladder line closed, or is a NEW preregistration admissible? If new, the two facts to design around are: (a) six 35-job × 5-site × 73-start cells are not provable within an hour by either solver, (b) the composite closure at 3 % is not achievable with brown at 2 % of draw and a 1.5 % draw mismatch.
2. If a new preregistration: candidates I have NOT evaluated and will not start without a ruling: fewer starts (coarser grid), fewer jobs per window, a closure tolerance stated on brown or on draw separately rather than on the composite, or a certified-gap (e.g. ≤1 %) instead of proven-optimal rung.
3. Whether the 1.5 % simulator-vs-model draw gap should be diagnosed before any new design (it also bounds every "planner-optimal" claim made so far at the ±3 pp Level-1 figure).
