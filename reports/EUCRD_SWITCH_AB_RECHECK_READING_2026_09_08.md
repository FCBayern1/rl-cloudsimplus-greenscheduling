# Switch comparison, light independent re-check under Addendum B: SWITCH_AB_PASS (2026-09-08, 21:5x)

Registered object: the policy surrogate gradient (reports/EUCRD_SWITCH_AB_PREREG.md Addendum B, 0dbb43b5, a development revision made after the decomposition was read and disclosed as such; threshold 0.9999 and max-over-calls aggregation carried over unchanged). Light protocol: the frozen G5-err global module restored in full (145 + 40 = 185 tensors, strict), learning rate zero, one sampling iteration, seed 20260909 (the decomposition used 20260908), 102 loss calls, the first excluded. No training. This is the re-check; the decomposition run is the engineering check of the same judge and is not a second confirmation.

## Gates

| gate | requirement | re-check | |
|---|---|---|---|
| W1 weights differ | mean \|Δw\| > 1e-3, max > 1e-2 | 0.0139 / 0.475 | pass |
| W2 change is targeted | firing / non-firing ≥ 2 | 0.244 / 0.0070 = **34.6** | pass |
| W3′ surrogate gradient differs | max clamped cosine < 0.9999, null exactly 0 | **cosine mean 0.975, min 0.947; null 0 on every call** | **pass** |
| W3 (superseded, reported) | whole-loss cosine < 0.9999 | 1.0000 | fail, as in run 4 |

Verdict **SWITCH_AB_PASS**.

## Consistency

Fifth consistent reading of the weights (targeting 33.5 / 33.4 / 32.4 / 32.5 / 34.6). The surrogate gradient turns by a comparable amount on a different sample (relative L2 median 0.252 against 0.317; cosine mean 0.975 against 0.968), and the dilution by the value term is again measured, 335× at the median (value-term norm 22.5 against the surrogate's 0.070; the surrogate is 0.31 % of the total norm). Advantage sign, reported: 2.1 % of negative-advantage transitions damped (ratio 0.9973), 1.0 % of positive (1.0024).

## What may be written

The forecast responsibility changes the gradient of the policy surrogate loss, substantially, and does so where the forecast changed a decision. Not: that the policy update changed by that amount (global-norm clipping over actor and critic together, and the optimiser, sit in between). Not anything about carbon. The matched pair's launch precondition is met; per the ruling of the same evening, the small-step update trial (reports/EUCRD_SMALL_STEP_PREREG.md) runs first as a cheap direction screen, and the pair waits for its reading and a ruling.
