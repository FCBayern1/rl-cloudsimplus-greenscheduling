# Stage D long-run preregistration — DRAFT for Codex ruling (2026-09-04)

Steps 2 and 3 of the chain, judged. Everything not stated here is inherited from `STAGE_D_PREREG.md` (Addenda B, G, H, I): the HZ ×2 scene, ledger-aligned reward, four lines N_V / V / N_E / E trained clean and corrupted only at deployment, primary corruption `calibrated_shrink_v1`, negative controls shuffle / anti, the mechanical runner, reader and freeze manifest of the health smoke (PASS_HEALTH, run 2). Identity: accelerated-weather, marginal-carbon mechanism positive control.

## 1. Training budget (R-t)

- Every line: **400,000 env steps** (50 PPO iterations of 8,000), checkpoint every 40,000 steps plus `checkpoint_init`; the verdict checkpoint is the last one (400k). No result is read before all seeds finish; the budget is not changed after any result is seen.
- Rationale for the number, fixed now: the health smoke at 56k left every line near-uniform (entropy ≈97% of uniform, forecast sensitivity ≈0.02–0.04); the F2 pilot's 104k under the legacy reward reached a decisive policy. 400k is 7× the smoke and 4× the pilot. If Codex prefers a different fixed number it is set before launch and never revised.

## 2. Seeds and hardware (R-u)

- **3 paired seeds** (20260904, 20260905, 20260906) as the minimum; **5** (adding 20260907, 20260908) if Isambard is available after a short equivalence smoke (same inputs, action shapes, loadable checkpoints, identical metric fields).
- A seed's four lines run as one block on one hardware class, two lines at a time; hardware never tied to an arm. 5080 budget: ≈13.5 h per seed (≈6.7 h per pair), 3 seeds ≈ 40 h serial.
- Container / CUDA / PyTorch / Ray / jar / config / source hashes frozen in `stage_d_freeze.json` before the first seed starts.

## 3. Windows (R-v)

- Training: the six frozen development offsets 6962 / 15942 / 24014 / 32086 / 40158 / 51156 (`green_episode_offset_allowlist` in the training blocks, as in the smoke).
- **Judgement: the six unread windows 13016 / 21088 / 29160 / 37232 / 45304 / 48230** (`stage_d_windows.json`, pairwise disjoint from every read window under the registered footprint). Implementation before launch, with tests: evaluation blocks that carry these six offsets as their allowlist, selected by `--reset-skip 0…5`; the evaluation runner asserts the allowlist offset reported by the planner-side signature against the registered list. Offsets, turbines, config and trace hashes are frozen before any of these windows is read.
- Certified windows k=26/34/42 are evaluated as well and reported as "certified benchmark evaluation", secondary, never a rescue.

## 4. Readings

Per seed, per line, final checkpoint, stochastic decode, six cells × six judgement windows:

| line | clean | calibrated_shrink_v1 | shuffle | anti |
|---|---|---|---|---|
| N_V | C_NV | — | — | — |
| V | C_V0 | C_V1 | C_V,sh | C_V,an |
| N_E | C_NE | — | — | — |
| E | C_E0 | C_E1 | C_E,sh | C_E,an |

C = pooled carbon intensity (Σcarbon / ΣMI) over the 36-run grid, per seed; seed-level medians and per-seed direction counts. 4 lines × (1+4+1+4) tiers × 36 = 360 evaluations per seed (+ 216 on the certified windows).

## 5. Gates (R-w), unchanged from Addendum B, evaluated per seed then by direction count

1. Vanilla uses the forecast: (C_NV − C_V0)/C_NV ≥ 5%.
2. The error hurts vanilla: (C_V1 − C_V0)/C_V0 ≥ 5% and (C_V1 − C_V0) ≥ 0.5·(C_NV − C_V0).
3. EU-CRD does not buy robustness by ignoring the forecast: (C_NE − C_E0)/C_NE ≥ 5%.
4. EU-CRD removes at least half of vanilla's corruption increment: (C_E1 − C_E0)/C_E0 ≤ 0.5·(C_V1 − C_V0)/C_V0.
5. Absolute: C_E1 < C_V1; C_E0 ≤ 1.05·C_V0; contracts green (completion ≥ 0.995, ontime ≥ 0.995, forced 0, cap 0) on every clean deployment; reward and ledger carbon co-directional init → final; EU-CRD Δr/ΔQ spread > 0 and routing/forecast shares not pinned (logged).

Direction: each of 1–4 holds in ≥ 2/3 seeds (≥ 4/5 with five). G0: every expected row present and contract-green on the clean deployments; a failed run voids its (cell, window) symmetrically; missing rows are INVALID, not a verdict. Negative controls reported, not gated. Verdict names: PASS_STAGE_D, STOP_STAGE_D_STEP2 (gates 1–2 fail: a trained policy ignores or is not hurt by the error), STOP_STAGE_D_STEP3 (gates 1–2 pass, 3–5 fail: EU-CRD does not resist on this scene). Both STOPs are reported as negative results; whether a negative paper is submitted is a submission decision, not a reason to change gates.

## 6. What may and may not change after launch

- May: wiring failures (missing checkpoint, tier not in effect, hash mismatch, missing field) fixed append-only and the affected seed re-run from scratch.
- May not: training length, seeds, hyper-parameters, reward, trace, windows, checkpoint rule, gates, aggregation. Policy collapse, zero forecast sensitivity, inert CRD signals, reward–ledger opposition or contract failure on the clean deployment are substantive and stop the affected line's seed with no re-tuning.

## 7. Implementation checklist before launch (each with a test)

- `gen_stage_d.build_eval(windows="judgement")`: 30 blocks with the judgement allowlist; `stage_d_run.py` phases `train_seed`, `eval_seed`, `verdict_longrun` with the 360-row assertion and the allowlist check; `stage_d_longrun_verdict.py` implementing §5 on a row table (tests: pass, each STOP, INVALID, voiding).
- Freeze manifest per seed; SHA256 list of every output; artefacts under `reports/manifests/stage_d/longrun/`.
