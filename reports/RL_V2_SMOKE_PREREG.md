# RL_V2 health smoke: does training break a correct prior (preregistration, 2026-09-07)

Status: FROZEN by the commit that adds this file (hash in STAGE_D_PRIME_DESIGN.md §58). User ruling of 2026-09-07 after STOP_F2_LEARNER: the behaviour-cloning line is closed (F_FITS_V2 sealed), the RL policy is a fixed cover prior plus a zero-initialised residual, four lines, cover_argmax reported at every tier, one seed of 56k steps first, five paired seeds only after this smoke passes, the 2020 windows sealed, the GPU parked for long runs. This document covers the smoke only; the long-run preregistration is written after its verdict.

## 1. Question the smoke answers (health, no effect claim)

1. Does vanilla PPO break the correct prior on the clean channel?
2. Does the mild shrink (λ = 0.75) make the trained vanilla policy worse?
3. Does EU-CRD keep more of the prior's value under that shrink?
4. Is EU-CRD's robustness not bought by ignoring the forecast?

## 2. Policy (`cover_prior_fixed: true` in the score module, tested)

- Offset logits: logit(j, d, κ) = cover(j, d, κ) + residual(j, d, κ). The cover term is a fixed buffer (gain 1.0, never trained). The residual is the score module's site score plus offset head with their output layers zero-initialised, so the untrained module's deterministic decode equals `cover_argmax` with the index tie rule (smallest action index on exact ties, the order torch.argmax uses). Everything else of the module (trunk, critic) is the frozen v5.2 architecture.
- Init check (hard gate, before training counts): the init checkpoint (before the first SGD step, `save_init_checkpoint`) executed with deterministic decode on the six reading windows must reproduce `cover_argmax` (COVER_TIE=index) on the same twin action for action at every decision and carbon for carbon per window. Any mismatch is STOP_INIT_MISMATCH.

## 3. Lines (one seed 20260907, 56k steps, checkpoints at 0 and every 8000)

| line | forecast channel in training | cover key built from | credit |
|---|---|---|---|
| NV | `forecast_mode: none` | persistence (the present repeated; the key exists with no future information) | vanilla PPO |
| V | clean truth (`perturbed_godeye`, tier godeye) | the arm's own curve | vanilla PPO |
| NE | as NV | as NV | EU-CRD (frozen v5.2 block, `crd.enabled: true`) |
| E | as V | as V | EU-CRD |

Same architecture, prior, action, budget, hyper-parameters and seed; the two named keys are the only differences (whitelisted diff, manifest with hashes).

## 4. Scene, windows, executor

- Certification interface twin (`config_ladder_cert_interface.yml` derivation), dense every-step (DC, κ) executor, `OFFSET_GRID_DENSE=1`, the certified LAG and row conventions (design log §54–§56).
- Training allowlist: the six 2021 development windows and the six F_FITS_V2 training windows (16477, 4240, 9154, 33225, 13223, 49625, 38088, 5463, 11249, 44834, 31713, 35380), cycled.
- Reading windows (health readings, all already read once by F_FITS_V2 and never trained on): the two validation and four test windows of F_FITS_V2 (21850, 1839, 24859, 28745, 41897, 7934).
- The 2020 confirmation windows are not touched.

## 5. Readings (last checkpoint, registered stochastic decode; init checkpoint deterministic for the init check only)

Per line, per reading window, per tier, the pooled simulator carbon and contracts; the tiers are godeye (clean), shrink75 (the ladder's λ = 0.75, provider tier tested equal to `rung_curve`), shrink50, shrink25, shrink0, shuffle, anti; NV / NE only on their own channel. Reference on every (window, tier) in the same pass: `cover_argmax` on that tier's key (index ties), the causal expert (truth) and the offline flat planner from the F_FITS_V2 test pass for the four test windows (recomputed for the two validation windows). Capture is against the causal expert's headroom as in F_FITS_V2.

## 6. Health gates (pass / fail only; every failure is a pipeline finding, fixed as an addendum before any rerun)

1. STOP_INIT_MISMATCH if §2's init check fails on any window.
2. All four lines train to 56k with loadable init and last checkpoints; contracts green on every reading episode (completion ≥ 0.995, on-time ≥ 0.995, forced 0, stale 0).
3. Prior preserved (question 1): V's clean-channel pooled capture on the reading windows ≥ 0.80 of `cover_argmax`'s (both at the last checkpoint; capture ratio, not absolute), and its deterministic-decode action agreement with `cover_argmax` on the reading windows is reported.
4. The shrink hurts (question 2): V's pooled carbon under shrink75 exceeds its clean pooled carbon by ≥ 5 %, and `cover_argmax` under shrink75 also exceeds its clean carbon (the reference's own loss is reported next to the policy's, so the loss can be attributed to the forecast rather than to training).
5. EU-CRD keeps more (question 3): E's pooled loss under shrink75, relative to its clean carbon, is at most 0.5 × V's; and E's clean capture ≥ 0.80 × V's clean capture.
6. Not by ignoring the forecast (question 4): E's clean carbon is below NE's by ≥ 5 %, and E's action distribution differs between godeye and shrink75 (KL of the action marginals > 0, the Stage D probe).
7. EU-CRD internals active (Δr non-zero with non-zero variance, responsibility gate not saturated) from logged statistics.

Smoke verdict: PASS_SMOKE iff gates 1–7 hold; otherwise the failing gate names the finding. The five-seed long run has its own preregistration and is not started by this document.

## 7. Implementation checklist (each with a test)

- Score module `cover_prior_fixed` (buffer gain, zero-initialised site keys and offset head; init decode == cover_argmax index ties): tests/test_offset_mode_module.py.
- Provider tiers shrink75 / 50 / 25 / 0 equal to the ladder's rungs: tests/test_forecast_row_alignment.py.
- `gen_rl_v2.py`: the four blocks by whitelisted diff from the certification interface twin, eval blocks per tier, manifest of hashes (config, jar, windows, crd subtree, model flag); test asserts the diff.
- `rl_v2_smoke.sh`: init check → four trainings (two at a time on the local GPU) → readings → judge; artefacts under `reports/manifests/rl_v2/smoke`.
