# Small-step update trial: LITTLE_CHANGE, direction not established (2026-09-09, 00:08)

Preregistration reports/EUCRD_SMALL_STEP_PREREG.md (cd556239) with Addendum A (c8c3961d, frozen while `ss_on` was training and before any deployment result existed). One PPO iteration (8 000 steps) from the G5-err global module restored strictly (185 tensors), warmup 0 as disclosed, seed 20260910; deterministic deployment of four arms on the six development windows at godeye and shrink75. An independent development experiment: it decides machine time, not whether EU-CRD works.

## 1. Contracts first

Identical across all four arms, both tiers, all six windows: completion 100 %, completion by MI 100 %, on-time share 1.0, forced deadlines 0. No arm buys carbon with work.

## 2. Absolute carbon, mean per window over the six windows (kg)

The judge pools with `np.mean`, so every figure below is the **mean carbon per window**, not a six-window total. The relative percentages are unaffected.

| arm | godeye (mean/window) | vs base | shrink75 (mean/window) | vs base |
|---|---|---|---|---|
| `ss_base` (no update) | 0.001590 | — | 0.002070 | — |
| `ss_on` (responsibility + reweighting) | 0.001620 | +1.5 % | 0.002110 | +1.7 % |
| `ss_off` (forecast channel silent) | 0.001620 | +1.8 % | 0.002110 | +1.7 % |
| `ss_norw` (reweighting off) | 0.001600 | +0.5 % | 0.002180 | +5.0 % |

**Primary comparison** (`ss_on` vs `ss_norw`, the reweighting switch alone): **−3.2 %** at shrink75, +1.0 % at godeye. Both inside the ±5 % margin → **LITTLE_CHANGE**. Read as the ruling of 2026-09-09 puts it: a weak favourable observation for having the reweighting on, below this round's screening threshold. Per window at shrink75 `ss_on` is lower in four windows, tied in one, higher in one (0.002388 against 0.002253), so the pooled figure is not one window's doing but it is also not uniform.

**Control** (`ss_on` vs `ss_off`, the forecast channel): +0.002 % at shrink75, −0.3 % at godeye — no distinguishable deployment difference between the new responsibility source and the historical one.

**Against the pre-update state**: every trained arm is slightly *worse* than the unchanged checkpoint, at both tiers. One iteration at this budget does not lower carbon; within that, the reweighted arms are less bad than the unweighted one at shrink75.

## 3. The sampling-summary check, as required by Addendum A2

| arm | iteration-1 episode return mean | episodes | env steps |
|---|---|---|---|
| `ss_on` | −32.70668588969067 | 12 | 8 000 |
| `ss_norw` | −32.70668588969067 | 12 | 8 000 |
| `ss_off` | −32.69537524971745 | 12 | 8 000 |

The primary pair's **sampling summary statistics are identical** — `ss_on` and `ss_norw` agree on the episode-return mean to the last digit, on episode count and on env steps. That is what these logs can show; it is **not** proof that the observation, action and reward sequences are bit-identical, and no batch hash was recorded, so the stronger claim is not made. `ss_off` differs in the fourth decimal: a sampling difference that appeared after the configuration switch, cause not yet verified (the silent source also changes which data path the env takes, but that has not been isolated). The control comparison carries that caveat; the primary one does not.

A note for the responsibility-source ablation registered in the matched pair's §6: it should keep the environment's data path identical across arms and zero the signal only at the learner's entry, so that no arm's sampling can differ for a reason unrelated to the credit.

## 4. Reading

Direction not established. The screen's own rule applies: a small difference after one iteration does not show long-run ineffectiveness, and a deterministic policy may simply not move its argmax at this budget. Two things are nonetheless worth carrying forward, neither of them a claim about EU-CRD's value:

- The one thing that did move carbon by more than the margin is *turning the reweighting off* at shrink75 (+5.0 % against base, against +1.7 % with it), i.e. the reweighting is not obviously harmful at this budget.
- The new responsibility source shows no distinguishable deployment difference from the historical one here (+0.002 %), consistent with the switch comparison's finding that it changes the surrogate gradient while the update itself is clipped and small.

## 5. What this licenses

It licenses a decision about machine time, nothing else. It does not weaken or strengthen any frozen verdict, does not touch the matched pair's registration, and says nothing about the 2020 windows, which stay sealed.
