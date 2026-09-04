# Stage D long-run preregistration — FROZEN (Codex rulings R-t…R-w, 2026-09-04)

Status: FROZEN at the commit carrying this text; later changes are addenda only. Codex's rulings and required corrections are in §8 and override any conflicting sentence above it.

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

## 8. Codex rulings (2026-09-04) and corrections that override the draft

- **R-t:** 400,000 steps per line, 50 PPO iterations, verdict on the fixed last checkpoint. No extension to 600k; no selection of an intermediate checkpoint.
- **R-u:** **five paired seeds, fixed** (20260904, 20260905, 20260906, 20260907, 20260908); the "five if resources allow" rule is deleted. A seed's N_V / V / N_E / E complete on the same hardware environment; same GPU model throughout (the local RTX 5080, two lines at a time, seeds serial; ≈67.5 GPU-hours). Hardware is never tied to an arm.
- **R-v:** the six unread windows (offsets 13016 / 21088 / 29160 / 37232 / 45304 / 48230) are the **only** main judgement set; the certified windows k=26/34/42 are a secondary replication run after the main verdict is frozen, never a rescue.
- **R-w:** gates 1–5 kept; direction ≥ 4/5 seeds; added to G0: on every line, reward and ledger carbon move in the same direction from `checkpoint_init` to the final checkpoint (clean tier, judgement windows).
- **Contract rules (replace §5's voiding sentence):** no (cell, window) is ever voided. INVALID_DATA only for a missing file, a tier not in effect (provider tier / hollow flag not matching the block), or a hash mismatch. Any contract failure (completion < 0.995, ontime < 0.995, forced > 0, cap > 0) on a **clean** deployment of any line is the substantive verdict **STOP_STAGE_D_CONTRACT**. Lateness or contract failure on a **corrupted** deployment is part of the forecast harm and is kept in the analysis: for E it fails the robustness gate (gate 4/5), for V it is reported as additional harm while gate 2 must still pass on carbon alone.
- **Before launch, locked:** (1) this file FROZEN; (2) the 360-row-per-seed main runner and reader implemented and tested (`stage_d_longrun.py`, `stage_d_longrun_verdict.py`), plus the 144 init-checkpoint clean rows for the co-direction check; (3) before each seed's evaluation the initialisation weight hashes of the learner RLModule must satisfy N_V = V and N_E = E (the smoke showed N_E ≠ E because Python hash randomisation flips a set/dict iteration order in the EU-CRD module build; `PYTHONHASHSEED=0` makes the build bit-reproducible, verified twice, and is fixed in every training and evaluation environment); (4) `RAY_TMPDIR=/home/joshua/rt` (the /tmp partition is full) and a hard disk gate of ≥ 50 GB free on /home before every phase; (5) per-seed freeze manifest with GPU model and driver, CUDA, PyTorch, Ray versions, PYTHONHASHSEED, jar, configs and source hashes, written before the seed's first process starts; (6) during training only liveness, step counts, disk and NaN are watched; carbon, reward, corruption effects and intermediate-checkpoint behaviour are not read.
- The chain of conclusions so far: forecast error is established to hurt the analytic scheduler; the RL chain is healthy; the long run tests whether vanilla learns and inherits the harm, and whether EU-CRD removes at least half of it.

## Addendum A (2026-09-04, wiring failure on seed 20260904, seed restarted from scratch)

At 07:24, after 40 of 50 iterations (320,000 steps) on both N_V and V, training raised `OSError 28 No space left on device` inside RLlib's `save_checkpoint`: Ray Tune writes each checkpoint to Python's temp directory (default /tmp) before persisting it to `storage_path` on /home, and /tmp had been filled by temp files of an unrelated earlier session (17 GB) plus finished Ray sessions. `RAY_TMPDIR` alone did not cover that path. Fix (append-only): `TMPDIR=/home/joshua/rt/tmp` in every training and evaluation environment, and the disk gate now checks /tmp (≥ 10 GB) as well as /home (≥ 50 GB); both recorded in the per-seed freeze. Per §6 the affected seed's outputs are archived under `logs/stage_d_longrun_INVALID_seed20260904_disk/` and the seed is re-run from scratch; nothing of the failed run is read. The N_V = V init-hash check had passed (a55d8802…) before the failure.
