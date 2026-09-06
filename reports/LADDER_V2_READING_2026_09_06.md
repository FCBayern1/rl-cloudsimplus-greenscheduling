# Error ladder v2: reading (2026-09-06)

Preregistration: reports/ERROR_LADDER_PLANNER_PREREG_V2.md, frozen at commit 6d9d1e64 (sha256 b1a723a950f2a93c) before any non-truth rung was solved. Run: `stage_a_out/ladder_v4`, archive `reports/manifests/ladder_v4/run1` (42 solve records, 42 replays with option ledgers, `ladder_verdict.json`, terminal log). Verdict: **LADDER_READ**.

## 1. Certification and closure

All 42 cells (6 windows × 7 rungs) compound-OPTIMAL at the root node, 3–5 s each (HiGHS, gap 0, 3600 s limit unused). All 42 replays close: worst carbon relative error 5.4e-7 (gate 3 %), draw relative error ~1e-15 (gate 0.1 %), brown absolute error < 5e-7 Wh (gate 0.002 Wh), every job on its planned site and start, every counter zero, curve rows matching the wind-file curve on every window. Nothing was retried, extended, or re-tuned.

## 2. Gate L1, headroom (simulator-settled carbon, kg)

| window | truth | flat (λ = 0) | headroom | rel | valid |
|---|---|---|---|---|---|
| k0 (16477) | 0.000629 | 0.005616 | 0.004987 | 0.888 | yes |
| k1 (4240) | 0.001028 | 0.005523 | 0.004495 | 0.814 | yes |
| k2 (9154) | 0.001225 | 0.006625 | 0.005400 | 0.815 | yes |
| k3 (33225) | 0.000489 | 0.006387 | 0.005898 | 0.923 | yes |
| k4 (13223) | 0.000525 | 0.002100 | 0.001575 | 0.750 | yes |
| k5 (49625) | 0.002159 | 0.007981 | 0.005821 | 0.729 | yes |

Six of six windows valid (gate: rel ≥ 0.15 and abs ≥ 7.66e-4 kg). The exact planner on the truth curve removes 73–92 % of the carbon of the no-forecast (flat) schedule.

## 3. Gate L2, load-bearing rungs (pooled over the six windows)

| rung | pooled loss vs truth (kg) | share of pooled truth carbon | headroom share with loss > 0 | load-bearing |
|---|---|---|---|---|
| shrink λ = 0.75 | 0.002515 | 41.5 % | 1.00 | yes |
| shrink λ = 0.5 | 0.012800 | 211 % | 1.00 | yes |
| shrink λ = 0.25 | 0.026867 | 444 % | 1.00 | yes |
| shrink λ = 0 (flat) | 0.028177 | 465 % | 1.00 | yes |
| shuffle | 0.025367 | 419 % | 1.00 | yes |
| anti | 0.030045 | 496 % | 1.00 | yes |

Gate: pooled loss ≥ 5 % of pooled truth carbon and ≥ 80 % of the headroom in windows with loss > 0. Every rung is load-bearing, including the mildest (a 25 % amplitude shrink towards the site mean costs 41.5 % more carbon than the truth schedule, on every window). The order shrink 0.75 < 0.5 < 0.25 < 0 ≈ shuffle < anti holds pooled; per window the flat and shuffle rungs trade places (k1, k2, k4).

## 4. Per-window simulator carbon (kg)

| window | truth | λ 0.75 | λ 0.5 | λ 0.25 | λ 0 | shuffle | anti |
|---|---|---|---|---|---|---|---|
| k0 | 0.00063 | 0.00079 | 0.00252 | 0.00501 | 0.00562 | 0.00531 | 0.00543 |
| k1 | 0.00103 | 0.00161 | 0.00326 | 0.00659 | 0.00552 | 0.00440 | 0.00772 |
| k2 | 0.00123 | 0.00181 | 0.00374 | 0.00566 | 0.00663 | 0.00685 | 0.00622 |
| k3 | 0.00049 | 0.00062 | 0.00337 | 0.00587 | 0.00639 | 0.00542 | 0.00560 |
| k4 | 0.00053 | 0.00063 | 0.00108 | 0.00202 | 0.00210 | 0.00265 | 0.00336 |
| k5 | 0.00216 | 0.00311 | 0.00488 | 0.00778 | 0.00798 | 0.00678 | 0.00778 |

## 5. What this establishes and what it does not

Established (first time in this project's chain): on the HZ zero-floor scene with the new turbines, an exact planner whose settlement is certified on the simulator to ~1e-7 loses carbon monotonically as the forecast it optimises on degrades, at every rung of a controlled ladder, on every development window. Forecast quality is load-bearing for the exact planner; the certified anchor is now a usable oracle for the causal, every-step fits (F1–F3) and for an RL preregistration.

Not established: anything about a learned policy, about the deployed TimeCAP's calibrated error (the A2 gate remains STOP on the ST heuristic; the ladder's rungs are synthetic), or about the sealed 2020 windows. The premise A restriction (running + ending jobs ≤ hosts per site) is part of the certified model; the optimum is over that feasible set.

## 6. Next steps (not started)

1. F1–F3 fits on the ladder's truth schedules as causal, every-step oracle labels (option_bc --offset), per the Stage D′ plan.
2. RL preregistration on the certification twin: arms, seeds, the ladder rungs as the perturbation axis, gates derived from §2–§3 (a policy's rung-loss profile against the exact planner's), EU-CRD as the intervention. Requires its own freeze before training.
3. The 2020 confirmation windows stay sealed until the RL preregistration names them.
