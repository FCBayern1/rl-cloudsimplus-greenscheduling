# Option four-gate result: gate 1 fails, fallback (DC, start-offset) starts (2026-09-05 21:10)

Context: `reports/OPTION_ACTION_DESIGN.md` v1 + Addenda A/B (frozen), STAGE_D_PRIME_DESIGN.md §29–§32. Zero training throughout. Artefacts under `reports/manifests/stage_d/dprime/option/`.

## 1. What was implemented (all committed, all tested)

Java hold ledger (hold / release at the decision clock with the route reward / execution-start times per id / `ep_opt_*`), the Python executor (reservation grid with the planner's constants, fallback reservation at the latest feasible start, (slot, site) legality, T1 green + capacity with same-step accumulators, T2 fallback), env mode `option_v1` (2n actions, hold mask, per-site held keys, one translation rule for every arm, ledger CSV), the score-based module's option mode, the analytic option arms of §5, `always_hold`, the option config on the same six train windows, the judges for gates 3 → 1 → 2 and gate 4 (fit + executed arm). Addendum B records that the reservation arithmetic lives in the env executor while Java keeps the primitives and the timing truth.

## 2. Instrument repairs, all disclosed (design §30–§31)

- Py4J passes Python ints as Integer; the release API cast to Long (crash) → typed on Number.
- A Long-keyed Java map cannot be looked up from Python ints → start times returned as a list aligned with the ids.
- The episode-end test did not count held cloudlets as unfinished; the first six-window run ended two episodes with options open (climatology_opt k1: five stale holds, completion 0.857; shrink_opt k1: one). Repaired once, as gate 3 allows; every row and the gate-4 corpus regenerated under the repaired jar. Gates 1–2 were not read from that run.

## 3. Results (run 2, jar 16df1990…)

Gate 3 smoke (k0, nine arms) and the full gate 3 (54 rows): PASS. Contract 1.000 / 1.000 everywhere, forced 0, no refused or masked hold, every option released and found in the simulator's start events, route→start 0 steps.

Gate 1 (oracle-driven option vs B = reactive_wait, ST = reserving godeye; P0′ run-6 rows):

| window | C_B | C_ST | C_oracle_opt | capture |
|---|---|---|---|---|
| k0 | 0.002634 | 0.001476 | 0.002114 | 0.449 |
| k1 | 0.005079 | 0.003191 | 0.004481 | 0.317 |
| k2 | 0.003876 | 0.003870 | 0.003608 | invalid denominator |
| k3 | 0.002718 | 0.000894 | 0.002188 | 0.290 |
| k4 | 0.001946 | 0.001600 | 0.001980 | −0.098 |
| k5 | 0.000422 | 0.000309 | 0.000634 | −1.883 |
| pooled | 0.016674 | 0.011339 | 0.015006 | **0.313** |

Need 0.80 pooled and 0.70 on all but one valid window; 0 of 5 reach 0.70. **Verdict: STOP_GATE1_FAIL_FALLBACK_OFFSET.** Gates 2 and 4 not read. Descriptive reading: releasing at the first moment the meter covers the job cannot express "wait past the first green for a better window", which is the reserving planner's timing; on the two smallest-gap windows the option is worse than the blind waiter.

## 4. What happens next by the frozen rule

The option is rejected as written and not modified. The preregistered fallback (§8 as amended by A5) starts: action (d, κ), κ ∈ K(72) = {0, 1, 2, 4, 8, 16, 32, 64, 72}; the executor is a reservation ledger with a fixed start, reading no green; offsets beyond the latest start or without a fitting reservation are masked, nothing is clipped; the oracle quantises its planned start down to the largest legal offset; arms oracle_off / shuffle_off / anti_off / persistence_off / climatology_off / nowait_off (κ = 0) / always_max (κ = 72 where legal); the same four gates, same order, same numbers, same references B and ST.

## 5. Rulings requested before implementing the fallback

1. Confirm the start of the fallback under §8 / A5 and that the option results above are archived, not revisited.
2. The fallback's executor reuses the Java hold primitives (hold at creation, release at t + κ_eff at the committed site) and the Python reservation grid; the only rule change is the termination (fixed step instead of green / margin). Confirm this reuse is acceptable, or require a separate implementation.
3. Timing truth: κ is a route offset; the ledger records both the release step and the execution start, and gate 3 keeps route→start ≤ 1 step. Confirm.
4. Gate 2's blind* for the fallback: the blind offset arms decide with their own curve view through the same rule (persistence: flat future, so the earliest cheapest start; climatology: mean curve). If both collapse to κ = 0 in practice, blind* is effectively nowait_off; state whether that is acceptable or whether a "latest legal offset" blind should be added as a further no-forecast option before freezing.
