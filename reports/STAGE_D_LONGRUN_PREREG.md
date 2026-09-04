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

## Addendum C (2026-09-04, two platforms in parallel; wall-clock arrangement only)

The same frozen protocol now runs twice, on two platforms, at the same time. Nothing scientific changes: budget, seeds, windows, reward, lines, corruptions, gates, aggregation and the reader are those of §5 and §8.

- **Workstation (RTX 5080).** Measured while two lines trained: load 2.3 of 8 cores, 26% busy, so the bottleneck is the py4j round trip, not CPU. All four lines of a seed therefore run at once (`STAGE_D_PARALLEL_LINES=4`) instead of two, and the seeds stay serial. Seed 20260904 is restarted from scratch under this arrangement so that all five seeds are identical; its two-line partial run is archived as `logs/stage_d_longrun_SUPERSEDED_2line/`, uninterpreted.
- **Isambard-AI (GH200).** Twenty single-GPU training jobs (five seeds × four lines) plus one dependent evaluation job per seed. A whole four-GPU node is never free on this cluster, while single-GPU jobs start immediately. Per line this hardware is about three times slower (measured: 1397 s per 8000 steps against the workstation's 478 s; an A/B against node-local storage gave 1383 s, so it is CPU, not I/O), but twenty lines run at once. The paired init-hash check (N_V = V, N_E = E) cannot run inside a per-line job, so it runs as a gate at the start of that seed's evaluation job and fails the seed if it does not hold.
- **Which platform is primary — WITHDRAWN by Addendum E, which fixes the workstation as primary.** (Original text: the platform whose five seeds complete first, by wall clock, is the primary run; the other is reported as a hardware replication of the same preregistration. No carbon, reward or gate quantity from either platform is read before that ordering is settled, and the primary is never re-chosen afterwards. Both platforms' verdicts are reported whether they agree or not; a disagreement is itself a result about the protocol's stability, not a reason to prefer one.)
- Both platforms record their own per-seed freeze manifest (commit, GPU model and driver, CUDA, PyTorch, Ray, `PYTHONHASHSEED`, jar and config hashes). The Isambard jar is built on aarch64 so its SHA differs from the workstation's by construction; the equivalence evidence is in `reports/manifests/stage_d/isambard_equivalence_smoke.md`, including that both platforms produce bit-identical initial weights for the same seed.

## Addendum D (2026-09-04, Isambard first submission voided by a wiring failure)

The twenty training jobs submitted at 13:14 cluster time all exited after sixteen seconds. `stage_d_longrun.py` hard-coded the workstation's `RAY_TMPDIR=/home/joshua/rt`, so the freeze step raised `PermissionError` on Isambard, and the sbatch continued past the failure and reported COMPLETED; the five dependent evaluation jobs then failed on the missing checkpoints. No result was produced and none was read. Fixes, append-only and tested: the runner takes `RAY_TMPDIR`, `TMPDIR` and `STAGE_D_DATA_PATH` from the environment when set (test `test_tmpdirs_follow_the_environment`), the disk gate checks the partition holding the outputs and the one holding TMPDIR rather than the workstation's `/home` and `/tmp`, and both sbatch scripts abort on a failed phase instead of exiting zero. The jobs are resubmitted from the corrected commit. The workstation run is unaffected.

## Addendum E (2026-09-04, Codex ruling; supersedes Addendum C's platform rule, recorded before any long-run metric was read)

- **Primary platform is fixed, not raced.** The RTX 5080 workstation is the primary run, continuing the health smoke and the original R-u. The GH200 cluster is a full hardware replication. The "whichever completes first" rule of Addendum C is withdrawn: tying the primary result to scheduling, faults and execution speed is not sound, even though the ordering was to be settled without reading any outcome. Both platforms are reported in full; where they disagree, the GH200 run may not replace the workstation's primary conclusion. The two mechanically refilled cells (below) do not change the primary platform's identity.
- **Deployment-time auditor, current Stage D.** No code change and no change to gate 4. The correlation monitor is designated a phase and sign-error detector and may not be claimed to detect amplitude shrink; the Q-ensemble signal may not be offered as shrink-detection evidence either. Gate 4 rests on EU-CRD's policy behaviour as a whole; if it passes, the robustness is attributed to the training-time credit mechanism and not to a deployment auditor. The auditor signals may be re-measured on the 400k checkpoints as a diagnostic only, never entering a verdict.
- **R-x, the two failed evaluation cells.** Both are on seed 20260904 of the workstation run, both produced no result CSV, and both occurred while a diagnostic evaluation of the auditor was competing for the same eight cores (load 10.2). Their causes differ and are reported as measured rather than folded together: `NV_init / s2_r48_w72_c5_n20 / hollow / k5` failed with `EOFError: Ran out of input` while reading the checkpoint, which is the case the ruling names; `NE_final / s2_r48_w72_c1_n50 / hollow / k4` failed with `torch.distributed.DistNetworkError`, a port-bind collision under process contention. Both are evaluation-infrastructure failures with no substantive content. Refilling is permitted subject to: no valid result CSV exists (confirmed for both); the refill uses the identical checkpoint, config, tier, seed, cell, window and hashes; only the missing cells are re-run and only once; no diagnostic process runs in parallel during the refill; the refill completes and the manifest is frozen before any carbon quantity is read. If the same cells fail again, that platform's seed is recorded INVALID_DATA rather than refilled repeatedly.

## Addendum F (2026-09-05, Codex ruling on platform scope; written before any long-run carbon, reward or gate quantity was read)

> Workstation is the sole platform for the preregistered Stage D Gates 1–5. Isambard is the complete hardware replication platform and the sole platform for the expanded algorithm-comparison table. Every comparison and denominator within a table must come from the same platform. Results may not be substituted across platforms because of failure, speed, or effect direction.

Consequences, spelled out:

- **Gates 1–5** are computed only from the workstation's N_V, V, N_E and E. The Isambard run of those same four lines is a complete hardware replication and is reported alongside, never merged into the verdict.
- **The algorithm-comparison table** (EU-CRD against CCA-PG and the four risk-sensitive objectives) is computed entirely on Isambard, including the N_V, V, N_E and E rows that serve as its references and denominators. Ranking the workstation's EU-CRD against an Isambard baseline is forbidden.
- **If Isambard is incomplete** because jobs hit the 24-hour QoS wall, the affected algorithm or seed is recorded as missing and the comparison table is demoted to incomplete or to supplementary material. Holes are never patched from the workstation.
- **If the two platforms disagree in direction**, the workstation's verdict stands unchanged, both platforms' results are disclosed side by side, neither is chosen for being more favourable, and the Isambard ranking table may not be presented as cross-hardware replication of the main conclusion.
- **Why within-platform pairing is still required**: the bit-identical initial weights across x86-64 and aarch64 establish that wiring and initialisation agree, not that training trajectories agree across architectures.
- **Implementation requirement** for the comparison reader, when written: it must assert that every row it aggregates, including the denominators, carries the same platform tag, and refuse to produce a table otherwise.

**R-x closed.** Both refilled cells are accepted as infrastructure failures that produced no valid result: `NV_init / c5_n20 / hollow / k5` (`EOFError`) and `NE_final / c1_n50 / hollow / k4` (`DistNetworkError`, port collision). Logs retained, only the missing cells re-run, once, with identical checkpoint, config, seed, window and hashes, and with the parallel diagnostics stopped. Seed 20260904 stands at 504/504 and is not recorded INVALID_DATA. What is ratified is the class "infrastructure failure with no valid result", not a general licence to re-run failures.

## Addendum G (2026-09-05, Codex ruling): verdict STOP_STAGE_D_CONTRACT, interim-look disclosure, run terminated

**Verdict.** On the primary platform, seed 20260904, line E (EU-CRD, clean truth-informed forecast) violated the deployment contract on 33 of 36 clean-deployment grids: on-time share 0.80–0.96 and 1–7 deadline-forced starts per grid (mean 3.9). Lines N_V, V and N_E were contract-clean on every grid. Under §8 as amended by Addendum E, a contract failure on a clean deployment of any line is the substantive verdict **STOP_STAGE_D_CONTRACT**. This gate is absolute, not a five-seed vote, so the remaining four seeds cannot restore a PASS. The preregistered Stage D is therefore terminated on this verdict. Gate 2 (calibrated error harms vanilla: +1.1% against the 5% threshold) failed on this one seed; with one formal seed and one pilot that is a strong indication, not a formal structural finding, and it is recorded as such.

**Interim-look disclosure (protocol deviation).** On 2026-09-05 at about 23:20, at the PI's instruction and before the five seeds had completed, the assistant read the final-checkpoint results of seed 20260904 on the workstation: the pooled carbon intensities of all ten (line, tier) combinations, the four gate effects, the gate booleans, the contract fields (completion, on-time share, forced starts) per grid, the defer-action rates, and the training-side CRD statistics (ρ, σ², c_t, ΔQ and Δr spreads); the initialisation rows were read for the co-direction check and the defer rates. After this reading **no code, configuration, threshold, training length, seed, window or run arrangement was changed**; the runs continued unmodified until this addendum. The verdict above follows from the preregistered absolute contract gate, mechanically, and not from a choice to stop on the interim effects. Because nothing was altered on the basis of the look and the contract gate had already triggered, the deviation does not invalidate the result; it is disclosed here so the record is complete.

**Termination.** Checkpoints, logs, result rows and manifests of every run in progress are archived unread beyond what this addendum states (workstation seeds 20260904 complete and 20260905 partial; Isambard main run, CCA-PG and risk-sensitive sets at roughly iteration 7 of 50), and the remaining jobs are stopped: they could only serve post-verdict diagnosis and cannot change the verdict. The CCA-PG and risk-sensitive preregistrations are closed unjudged, with their partial data archived.

**What follows is diagnostic, not a continuation.** A zero-training spatial–temporal decomposition of the analytic lever and a per-action-type audit of the EU-CRD credit signals, each with its definition frozen before it is run, labelled post-verdict mechanism diagnostics; neither alters this verdict.
