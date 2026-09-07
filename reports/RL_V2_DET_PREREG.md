# RL_V2 deterministic-deployment continuation (preregistration, 2026-09-07)

Status: FROZEN by the commit that adds this file (hash in STAGE_D_PRIME_DESIGN.md §60). Written after the smoke of `RL_V2_SMOKE_PREREG.md` stopped at gate 1b, before any last-checkpoint reading is taken. User ruling of 2026-09-07 after seeing the init gates.

## 0. What is closed and what is carried over

- The smoke of `RL_V2_SMOKE_PREREG.md` is **STOP_INIT_PRIOR_NOT_CARRIED** and stays that way. Its gate 1b failure is never rewritten as a pass; it is kept and reported as a **diagnostic of the cost of stochastic exploration**: at the frozen gain the sampled policy picks actions whose cover is 0.961–0.984 of the best (its own bar 0.95, met) yet burns 1.112–1.323 × the carbon of the same rule decoded deterministically (bar 1.05, missed) on all four lines, with every contract green. What the gate measured is exploration noise across many near-equal-cover actions, not a broken prior: with deterministic decoding the untrained module reproduces `cover_argmax` on 210/210 decisions per line and its carbon exactly (gate 1, INIT_OK).
- Not done, by ruling: the gain is not raised (that would be tuning after seeing carbon and would also damage exploration), the 1.05 bar is not relaxed (a standard changed after the fact), and the reference is not replaced by each line's own stochastic init (that would turn "does training preserve a strong rule" into "does training beat a noisy start").
- Carried over unchanged: the scene, the twin, the action, the interface, the fixed cover prior with gain 20, the four windows sets, the four trained checkpoints (56 000 steps each, seed 20260907), the reference arms, and the substantive questions.

## 1. The one change

Training keeps stochastic sampling exactly as it was (the checkpoints are not retrained). **Deployment evaluation is deterministic (argmax over the legal candidates, exact ties by the smallest action index).** This is the ordinary "explore while training, act deterministically when deployed" split, and it makes the policy readings commensurable with the zero-parameter references, which are deterministic by construction.

## 2. Frozen inputs

Checkpoints (last, 56 000 steps) and their init counterparts, sha256 over the checkpoint directory, recorded in `stage_a_out/rl_v2/checkpoint_hashes.json` before any reading:

| line | last checkpoint | sha256 | init sha256 |
|---|---|---|---|
| NV | checkpoint_000006 | 010be1a66729f4e2 | 7f30322c540fc149 |
| V | checkpoint_000006 | c9ec0d5f262315e3 | e44006279793cae7 |
| NE | checkpoint_000006 | 497ef3c874c4d460 | bf58183dd8ebe9ff |
| E | checkpoint_000006 | 8d0a47f9d096be62 | 6fdf5035c531bfbd |

Reading windows: the six of the smoke (F_FITS_V2 validation and test: 21850, 1839, 24859, 28745, 41897, 7934), none of them trained on. Tiers: godeye, shrink75, shrink50, shrink25, shrink0, shuffle, anti on the forecast channel; godeye only on the hollow channel. References already computed and contract-green: `cover_argmax` (deterministic, index ties) on every (window, tier), the causal expert, the offline flat planner. Pooled references on these six windows: offline flat 0.024826 kg, causal expert 0.009789 kg, cover_argmax clean 0.009634 kg (capture 1.010), cover_argmax hollow 0.021295 kg (capture 0.235), cover_argmax shrink75 0.011515 kg (+19.5 % over its own clean).

## 3. Readings, taken once

Every line, every tier of its channel, every reading window, deterministic decode, one episode: pooled simulator carbon, capture against the causal expert's headroom, and the contract counters. 96 runs. No selection of anything on any of these numbers.

## 4. Gates (the smoke's questions, decoded deterministically)

1. Contracts green on every reading episode (completion ≥ 0.995, on-time ≥ 0.995, forced 0, stale 0), all four last checkpoints loadable.
2. Prior preserved: V's clean pooled capture ≥ 0.80 × `cover_argmax`'s clean pooled capture.
3. The shrink hurts: V's pooled carbon under shrink75 exceeds its clean pooled carbon by ≥ 5 %, and `cover_argmax`'s own shrink75 loss is reported next to it (it is +19.5 %).
4. EU-CRD keeps more: E's relative shrink75 loss ≤ 0.5 × V's, and E's clean capture ≥ 0.80 × V's clean capture.
5. Not by ignoring the forecast: E's clean pooled carbon below NE's by ≥ 5 %, and E's action distribution differs between godeye and shrink75 (action-marginal KL > 0).
6. EU-CRD internals active (Δr non-zero with non-zero variance, responsibility gate not saturated) from the training logs.

Verdict PASS_DET_SMOKE iff 1–5 hold (6 is reported and required for any claim about EU-CRD's mechanism). Any failure of 2–6 is substantive: it is reported and the line stops for a ruling; nothing is tuned toward a pass. Execution faults (crashes, missing files, wiring) are fixed append-only and the affected runs repeated, as in Addendum A3.

## 5. If it passes

The multi-seed experiment is registered separately and from the outset as "stochastic training, deterministic deployment", with new seeds, its own windows and the 2020 confirmation set still sealed. Nothing in this document authorises it.
