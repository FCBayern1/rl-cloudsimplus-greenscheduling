# Scene and observation-interface design (v1, 2026-09-05 23:45; revised after the Codex review of v0; for Codex to freeze; nothing run)

Builds on STAGE_D_PRIME_DESIGN §35 and its diagnostic: the (DC, dispatch-offset) action with an every-step grid reproduces the reserving planner exactly on all six development windows; the frozen dyadic grid's downward quantisation, not the action, caused the k4/k5 failure. The option and offset verdicts are not reopened. Every training in §4.1–§4.5 is zero-RL (the supervised fits of §4.5 are training); RL appears only in §4.6.

Changes from v0 (Codex review): the confirmation windows are hash-drawn from a year never used for anything else and kept whatever their headroom; the headroom threshold no longer depends on the data it gates; the blind family covers every offset the oracle can choose; the observation gains one candidate-aligned key, whose semantics net out static draw and already-committed load; the fits of §4.5 use the development set only; seeds are frozen independently of any deadline.

## 1. Data isolation

- Turbines: option B, a never-used set chosen by the frozen hash rule of `stage_d_prime_turbines.py` (tag stage-d-prime-turbines-v1; first five eligible ids in hash order; DC0 ← 2, DC1 ← 2, DC2 ← 1; DC3, DC4 without turbines as in HZ). Eligible = never in any tracked config, audit or report, with complete 2020 (32,224 rows) and 2021 (52,559 rows) files. The choice is written to `stage_a_out/scene_v1_turbines.json` and frozen before any wind value of those turbines is read.
- 2021 = design year: scene certification, TimeCAP error calibration (`calibrated_shrink_hz_v2` on these turbines), margin probe, P0′, the headroom-gated development windows, the supervised fits and their selection.
- 2020 = confirmation year: six judgement windows drawn once by sha256("scene-interface-v1:2020:" + offset) over every legal footprint start, greedy non-overlap, all six kept whatever their headroom; read only by the final RL / EU-CRD judgement of §4.6. If the pooled headroom of the six (measured by B and ST only, at the time of §4.6, after everything else is frozen) is below the threshold of §2.2, the line reports STOP_CONFIRMATION_HEADROOM honestly; no window is replaced.

## 2. Scene certification (2021, zero RL)

### 2.1 Mechanism control, as ruled for any successor scene
Zero-training arms on hash-ordered 2021 windows: the reactive-wait blind B, the reserving godeye planner ST, shuffle and anti of ST, and after calibration the calibrated-shrink arm. Pass iff ST is below B on the pooled sum, shuffle and anti are not below B, and the contract holds (completion ≥ 0.995, on-time ≥ 0.995, forced 0), on the first twelve hash-ordered windows. These twelve are the development pool; the reading of §2.2 is taken on them.

### 2.2 Headroom gate, threshold fixed before any reading
A window has headroom iff (C_B − C_ST) / C_B ≥ 0.15 and (C_B − C_ST) ≥ 0.05 · C_brown, where C_brown is the window's carbon if every job ran on brown power alone (the physical all-brown bound of the fleet, computed from the trace and the fleet's power model; no arm is read for it). Both numbers are frozen here. Development windows = the six hash-earliest windows of the pool that pass the gate; fewer than six → STOP_WINDOW_SPLIT. Rejected windows are archived and never reused. The same gate is the confirmation test of §1 (applied to the pooled six 2020 windows, never per window, never to swap them).

## 3. Action and executor

(DC, dispatch-offset) as in OPTION_ACTION_DESIGN §8 / Addendum C, grid = every step 0..W (W = the wait cap in steps, 72 on the HZ cell; 73 values), no quantisation. Executor, legality mask, ledger, timing truth and the gate-3 contract exactly as Addendum C and the placement repair of §34 (fixed-start reservation, no green read, illegal offsets masked never clipped, route→start ≤ 1 step). No third action.

## 4. Gates, in order; a later gate is not read if an earlier one fails

### 4.1 Expressibility (development windows)
oracle_off (every-step grid, truth curve) against B and ST. Pooled capture ≥ 0.80 and headroom-weighted window robustness Σ_w gap_w · [capture_w ≥ 0.70] / Σ_w gap_w ≥ 0.80, gap_w = C_B,w − C_ST,w. Expected to pass from the §35 diagnostic; kept because the scene is new.

### 4.2 Blind family, frozen, matched to the oracle
fixed_off(κ) for every κ = 0..W (73 arms; site by current visible cost as in C3), reactive_off, persistence_off, climatology_off. All blind arms run first; blind* = lowest pooled carbon, frozen with row hashes before any informed row exists.

### 4.3 Predictive necessity
oracle_off, shuffle_off, anti_off after the freeze; the three conditions of OPTION_ACTION_DESIGN §6 gate 2 at 0.95 on the pooled sum and the headroom-weighted window rule of §4.1. Failure ends the line at this gate: the gain is reservation, not prediction.

### 4.4 Observation interface (one new key)
`cand_green_cover[j, d, κ]`, shape (NB, n·(W+1)), float32: the share of job j's energy over its full runtime, started at t + κ + lag at site d, that the arm's forecast green at d covers **after subtracting the site's static draw and the load already committed on the reservation grid for those steps**, clipped to [0, 1]; zero on padding slots and on illegal (j, d, κ). Committed load is the executor's grid (every route and every held reservation the env has seen), so two jobs cannot claim the same green. The curve is the arm's own: truth for godeye, the TimeCAP output for timecap arms, the perturbed curve for shuffle / anti / calibrated arms, through one function in the env; no arm sees another arm's curve or any answer. Feasibility is the existing legality mask; the deadline margin is derived in the module from the existing per-job time-to-deadline and κ (no new key). The four per-site summaries stay.

Volume: one key of 128 × 365 float32 = 187 kB per step. Before anything else is built on it, a memory-and-throughput smoke: one env-runner with RLlib's rollout batch at the Stage D setting, peak RSS and steps per second recorded against the D′ configuration; if peak RSS exceeds 1.5× the D′ figure or throughput drops below 0.5×, the key is stored as float16 or the candidate set is restricted to the mask's legal entries, and the smoke repeats; the choice is written down before §4.5.

### 4.5 Representation learnability (development set only)
Corpus: oracle_off decisions on the six development windows (k0–k3 fit, k4–k5 held out), one decision per job. Three supervised fits with the frozen recipe of Addendum C4 (Adam 1e-3, one step per window, clip 1.0, no class weighting, default init, seed 20260905, 200 epochs, argmax decode):
- F1: the D′ observation (four summaries, no candidate key);
- F2: the candidate key computed from the truth curve;
- F3: the candidate key computed from the TimeCAP forecast (the interface the RL will have).
Gate on F3: p_delay lift ≥ 0.10 and balanced AUC ≥ 0.60 on the held-out windows, executed capture ≥ 0.50 against blind* and oracle_off on those windows, BC contract clean. F1 and F2 are the ablation, read together with F3 and never used to select anything: F1 fails while F2 passes → the summaries lost the information; F2 passes while F3 fails → the forecast quality on this scene; all fail → sample or architecture (reported as open). The confirmation windows are not touched by any fit.

### 4.6 RL and EU-CRD (separate preregistration, written only after 4.1–4.5 pass)
Vanilla PPO and EU-CRD on the offset action with the F3 interface; the four-line design and the contract of Stage D; perturbation arms shuffle / anti / calibrated_shrink_hz_v2; seeds frozen at five paired seeds if both machines are available for the run, else the minimum of three paired seeds, decided from machine availability when that document is written and never from a deadline; the six 2020 confirmation windows read once; the SMDP credit question of OPTION_ACTION_DESIGN §3.2 resolved in that document.

## 5. Order and cost

1. Turbine choice and data isolation (minutes; frozen file).
2. Scene certification and calibration on 2021 (about one day of zero-RL runs; TimeCAP calibration included).
3. Headroom gate → six development windows (one hour).
4. Expressibility with the every-step grid (one hour).
5. Full blind family (76 arms × 6 windows, about two hours) → freeze → necessity.
6. Interface smoke, then F1 / F2 / F3 (about one hour).
7. Only then the RL preregistration.

Nothing runs before Codex freezes this document.
