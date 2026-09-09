# Small-step update trial: LITTLE_CHANGE, direction not established (2026-09-09, 00:08)

Preregistration reports/EUCRD_SMALL_STEP_PREREG.md (cd556239) with Addendum A (c8c3961d, frozen while `ss_on` was training and before any deployment result existed). One PPO iteration (8 000 steps) from the G5-err global module restored strictly (185 tensors), warmup 0 as disclosed, seed 20260910; deterministic deployment of four arms on the six development windows at godeye and shrink75. An independent development experiment: it decides machine time, not whether EU-CRD works.

## 1. Contracts first

Identical across all four arms, both tiers, all six windows: completion 100 %, completion by MI 100 %, on-time share 1.0, forced deadlines 0. No arm buys carbon with work.

## 2. Absolute carbon, pooled over the six windows (kg)

| arm | godeye | vs base | shrink75 | vs base |
|---|---|---|---|---|
| `ss_base` (no update) | 0.001590 | — | 0.002070 | — |
| `ss_on` (responsibility + reweighting) | 0.001620 | +1.5 % | 0.002110 | +1.7 % |
| `ss_off` (forecast channel silent) | 0.001620 | +1.8 % | 0.002110 | +1.7 % |
| `ss_norw` (reweighting off) | 0.001600 | +0.5 % | 0.002180 | +5.0 % |

**Primary comparison** (`ss_on` vs `ss_norw`, the reweighting switch alone): **−3.2 %** at shrink75, +1.0 % at godeye. Both inside the ±5 % margin → **LITTLE_CHANGE**. Per window at shrink75 `ss_on` is lower in four windows, tied in one, higher in one (0.002388 against 0.002253), so the pooled figure is not one window's doing but it is also not uniform.

**Control** (`ss_on` vs `ss_off`, the forecast channel): +0.002 % at shrink75, −0.3 % at godeye — indistinguishable.

**Against the pre-update state**: every trained arm is slightly *worse* than the unchanged checkpoint, at both tiers. One iteration at this budget does not lower carbon; within that, the reweighted arms are less bad than the unweighted one at shrink75.

## 3. The matched-sample check, as required by Addendum A2

| arm | iteration-1 episode return mean | episodes | env steps |
|---|---|---|---|
| `ss_on` | −32.70668588969067 | 12 | 8 000 |
| `ss_norw` | −32.70668588969067 | 12 | 8 000 |
| `ss_off` | −32.69537524971745 | 12 | 8 000 |

The **primary pair is exactly matched** — `ss_on` and `ss_norw` sampled the same first batch to the last digit, which is what the primary comparison needs. `ss_off` did not: selecting `instantaneous_carbon_cf` removes the env's truth-curve gateway call, and the sampled trajectory differs in the fourth decimal. The control comparison therefore carries that caveat; the primary one does not.

## 4. Reading

Direction not established. The screen's own rule applies: a small difference after one iteration does not show long-run ineffectiveness, and a deterministic policy may simply not move its argmax at this budget. Two things are nonetheless worth carrying forward, neither of them a claim about EU-CRD's value:

- The one thing that did move carbon by more than the margin is *turning the reweighting off* at shrink75 (+5.0 % against base, against +1.7 % with it), i.e. the reweighting is not obviously harmful at this budget.
- The forecast channel on top of the reweighting is not separable from noise here (+0.002 %), consistent with the switch comparison's finding that it changes the surrogate gradient but with the update itself clipped and small.

## 5. What this licenses

It licenses a decision about machine time, nothing else. It does not weaken or strengthen any frozen verdict, does not touch the matched pair's registration, and says nothing about the 2020 windows, which stay sealed.
