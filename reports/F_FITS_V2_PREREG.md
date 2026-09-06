# F_FITS_V2: a candidate-shared scorer on the causal expert's true candidate costs (preregistration, 2026-09-07)

Status: FROZEN by the commit that adds this file (hash in STAGE_D_PRIME_DESIGN.md §55). Approved by the user on 2026-09-07 with the definitions below. Written before any window is drawn, any label produced, or any fit run.

## 0. Standing facts and what this document answers

- The RL-side forecast providers were 12 rows stale on every COMPRESSED SPLINE scene (design log §54; fixed 475c36b8 with alignment tests). The F1–F3 result of F_FITS_PREREG (ALL_FAIL_OPEN) was produced on corpora whose candidate key described windows 13 rows before the labelled ones; it stays on record as a historical result under misaligned corpora and is not evidence about learner generalisation. Stage D's forecast readings ("godeye", "the error hurts V by 1.1 %") are withdrawn as stale-forecast legacy (STAGE_D_PREREG.md Addendum J). The offline exact ladder and the causal expert read the wind files directly and stand.
- Misalignment is established as a major cause of the old fits' failure; whether it was the only cause is what this document answers.
- On the aligned key the zero-parameter rule `cover_argmax` reproduces the causal expert with the truth key (pooled capture 1.009) and keeps 0.820 with the TimeCAP key.

## 1. Row-alignment hard sentinel (runs before any corpus is generated; any failure is STOP, no fallback)

On each twin used (F2: interface twin, perturbed-godeye tier godeye; F3: interface twin, TimeCAP), one instrumented episode per window with the forecast series dumped per step (`FORECAST_ALIGN_DUMP`), checked per (clock, DC, horizon index):

1. perturbed-godeye series index h at observation step t equals the simulator's green at observation step t + h (the wind-file row the simulator burns, `truth_curve`), exactly (float32 tolerance);
2. TimeCAP: the row the provider treats as "now" at clock c is the simulator's current row (its history ends there), checked through the provider's own true-series accessor against `truth_curve`; the forecast issued at step t targets rows t + 1 … t + H;
3. `cand_green_cover[j, d, κ]` recomputed from the dumped series and the executor's committed grid with the key's own formula equals the published key exactly (same present row, same future index h = κ + 1).

Sentinel record: `f_v2/sentinel.json`, per twin and window; STOP_ROW_ALIGNMENT on any mismatch.

## 2. Windows

- Training development set: the six 2021 development windows (already read) plus six new hash-drawn 2021 windows.
- Validation: two new hash-drawn windows.
- Test: four new hash-drawn windows, read once, after the model and the reader are frozen (§6).
- Drawing rule: seeded hash over candidate offsets (tag `f-v2:2021`, seed from the tag), footprint 2922 rows, every drawn window ≥ 2922 rows from every read window (the six development windows, the certification candidates 49625 / 36713 / 30299, the 2021 scene pool) and from each other; accepted in draw order; no window is replaced for weather, headroom or fit results. Record `f_v2/windows.json` with hashes.
- Labels: about 12 × 35 = 420 training decisions, 70 validation, 140 test.

## 3. Labels: the true candidate costs

For every decision of the causal expert (its first sighting of a job, on the certification offset twin), the cost of every legal candidate (site, κ):

- Single new job at the step: the certified version-2 objective increment (49·brown + draw, per-site host profile, occupancy premise) of placing the job at (site, start = t + κ + 1) on the expert's committed grid with the arm's curve, for every legal candidate; not the cover.
- Several new jobs at the step: for each candidate of each job, fix that job's action and re-solve the MILP for the other new jobs to proven optimality; the label cost is the resulting joint objective. If any re-solve is not proven within its limit, the state is excluded and counted (`states_excluded`); nothing is approximated.
- Target set = all candidates whose cost equals the minimum (exact integer ties). Illegal candidates are masked out.
- The expert's own executed action is in the target set by construction (checked; a violation is STOP_LABEL_CONSISTENCY).

## 4. Model and loss (F2 and F3 identical, only the twin differs)

- score(j, d, κ) = cover(j, d, κ) + residual_θ(x(j, d, κ)); residual is a candidate-shared MLP over per-candidate features x: cover, κ/72, site one-hot, the site's current green / short and long future means / utilisation, the job's PEs, MI, seconds to deadline, and the legality bit; output layer zero-initialised. No recurrence.
- Decode: argmax of score over legal candidates; exact score ties broken outside the argmax lexicographically by smaller κ, then smaller site. At zero residual the decoded action equals `cover_argmax` bit for bit (test).
- Loss: candidate-shared set loss, −log Σ_{a ∈ target set} softmax(score)_a over the legal candidates of the decision; mean over decisions.
- Recipe: Adam 1e-3, full-batch over the training decisions, seed 20260907. Model selection on the validation set loss only, over the fixed grid epochs ∈ {50, 100, 200} × weight decay ∈ {0, 1e-4}; never on executed carbon.
- Executed arm `cover_residual`: the frozen scorer applied online from the same feature function on the arm's own twin.

## 5. Gates (test windows, read once)

- Contracts green on every executed test episode (completion ≥ 0.995, on-time ≥ 0.995, forced 0, stale 0).
- Window validity: the offline exact planner's headroom on the window (C_flat − C_truth, both simulator-settled and closed) must satisfy the ladder's L1 rule (rel ≥ 0.15 and abs ≥ 0.05·C_brown_ref). At least 3 of 4 test windows valid, else STOP_TEST_HEADROOM; no window is replaced.
- F2 (hard gate), pooled over the valid test windows: executed capture (C_flat − C_fit) / (C_flat − C_causal_truth) ≥ 0.50 AND on every valid test window capture ≥ capture(cover_argmax, truth key) − 0.02.
- F3 (diagnostic): the same readings against cover_argmax with the TimeCAP key, reported; never overrides the A2 STOP; no claim about resisting the deployed TimeCAP error.
- Reference arms on the test windows, produced in the same pass: offline exact (truth, λ = 0), causal expert, cover_argmax on both twins.

Verdicts: F2 passes → F2_PASS (the RL preregistration may be rewritten as four lines, §7). F2 fails on capture → STOP_F2_LEARNER (the aligned interface carries the value; the learner still does not). Sentinel or label consistency failure → STOP before any fit.

## 6. Freeze order

Sentinel → windows → labels → fit and validation selection → the selected model, the feature function and the reader committed with hashes → reference arms and the fitted arm on the test windows in one pass → verdict. The 2020 confirmation windows are not touched.

## 7. RL (not started)

`RL_CERT_PREREG_DRAFT.md` stays a draft. After F2_PASS it is rewritten with four lines (vanilla without / with forecast, EU-CRD without / with forecast), one architecture with the cover starting point, one action, one budget, paired seeds, `cover_argmax` as a mandatory baseline, every legacy RL number removed, and an explicit rule on whether the fixed cover term is trainable in RL. The GPU stays parked until F2 passes.

## Addendum A (2026-09-07, before any draw: the footprint of the new windows)

The 2021 file has no position at all that is 2922 rows from every read window (fifteen read offsets: the six development windows, the three certification candidates, the 2021 scene pool; capacity computed = 0). The 2022 files for this fleet are stubs (no data) and 2020 is sealed. The rows an episode actually touches are: TimeCAP history 96 rows before the window, the 13-row head, the largest time-zone shift 108, the planner horizon 669 and the candidate horizon 121, about 1007 rows. Windows separated by 1200 rows therefore touch disjoint data; the 2021 capacity at 1200 is 22. The draw uses footprint 1200 with everything else unchanged (seeded hash, accepted in draw order, ≥ 1200 rows from every read window and from each other, never replaced). This is a change of a constant made before the draw, recorded here with the computation; it does not affect the read windows or any gate.
