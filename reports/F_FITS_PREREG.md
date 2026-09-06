# F1–F3 on the causal expert's decisions (preregistration, 2026-09-06)

Status: FROZEN by the commit that adds this file (hash in STAGE_D_PRIME_DESIGN.md §50). Written after gate A of the causal expert passed (pooled capture 0.968, six of six windows ≥ 0.92) and before any fit is run. Order per the user's ruling of 2026-09-06: causal expert and causal error gate → F1–F3 → RL preregistration. This document is conditional on CAUSAL_READ (gate B); if gate B fails, no fit is run.

## 1. Corpus

Labels: the causal expert's (site, κ) decision at each job's first sighting on the six development windows of the certification twin (`causal_truth`, `stage_a_out/causal_v1`), one decision per job, 35 jobs per window. Windows k0–k3 fit, k4–k5 held out; the 2020 confirmation windows are not touched.

Three observation corpora of the same decisions, each produced by replaying the expert's schedule (`schedule_replay`, dense grid) with the observation dump on the certification twin whose interface the fit will see, so labels and observations are exactly aligned:

- F1: the offset twin (the D′ observation: the four per-site summaries, no candidate key). The expert's own runs are this corpus.
- F2: the interface twin with the candidate key `cand_green_cover` computed from the truth curve (`perturbed_godeye`, tier `godeye`).
- F3: the interface twin with the candidate key computed from the TimeCAP forecast (`green_oracle_mode: timecap`, the frozen checkpoint of the scene's interface config).

Same jobs, same schedule, same labels in all three; only the observation differs.

## 2. Fit (frozen recipe of OPTION_ACTION_DESIGN Addendum C4)

The D′ score-based global module in offset mode built from the corpus's own block; per-slot cross-entropy on the labelled slots; recurrent state carried through each window in time order; Adam 1e-3, one optimiser step per window, gradient clip 1.0, no class weighting, default initialisation, seed 20260905, 200 epochs, argmax decode. No early stopping, no hyper-parameter search, no selection on any held-out number. Same model, action mask and budget for F1, F2, F3.

## 3. Gates (held-out windows k4, k5)

1. Executed capture (primary): the fitted module deployed as the `option_bc` arm on its own twin, one episode per held-out window; capture_k = (C_flat,k − C_fit,k) / (C_flat,k − C_causal_truth,k) with C_flat from the frozen ladder (offline λ = 0) and C_causal_truth the expert's simulator carbon. PASS iff pooled capture over k4, k5 ≥ 0.50 and the contract is green on both windows (completion ≥ 0.995, ontime ≥ 0.995, forced 0, stale 0).
2. Classification (reported as in version 1): p_wait = Σ over columns with κ ≥ 2 of the slot's action probability (raw, mask lifted) against the label [κ_expert ≥ 2]; lift ≥ 0.10 and balanced AUC ≥ 0.60 over one decision per job. The expert's earliest executable start is κ = 1, so "wait" means κ ≥ 2. Validity rule A7: fewer than 60 held-out jobs or fewer than 15 of either class → INVALID_CORPUS for this gate, reported and not replaced. From the expert's truth ledgers the held-out windows have 70 jobs with 3 at κ = 1: the classification gate is expected to be INVALID on this corpus and is then descriptive only; the executed capture decides.
3. Supporting readings: exact-action accuracy, site accuracy, offset MAE on the held-out decisions.

Reading rule (SCENE_INTERFACE_DESIGN §4.5): F1 and F2 are the ablation, never used to select. F3 passes gate 1 → the interface the RL will have carries the expert's value; the RL preregistration proceeds. F2 passes and F3 fails → the forecast quality on this scene (TimeCAP), reported. F1 fails while F2 passes → the summaries lose the information. All fail → sample or architecture, reported as open; RL is not started.

## 4. Not done

No RL. No 2020 window. No change to the causal expert, the ladder, or the certification twin after this freeze; a harness defect found while running is fixed append-only and disclosed.
