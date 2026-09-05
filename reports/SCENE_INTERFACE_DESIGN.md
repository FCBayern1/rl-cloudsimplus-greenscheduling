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

---

## Addendum A (2026-09-06 00:05; closes the four points of the Codex review of v1; where it conflicts with the sections above, the addendum governs)

### A1. The all-brown reference is a scheduling-free formula

§2.2's C_brown is replaced by

    C_brown_ref(window) = Σ_jobs  pes_j · P_dyn_pe · u · runtime_j(sec) / 3600  [Wh]  ×  f_brown_ref  [kg/Wh]

with P_dyn_pe = (214 − 51.4) / 64 W (the fleet's dynamic draw per PE, the constant every planner and the executor already use), u = the block's `cloudlet_cpu_utilization`, runtime_j = mi_j / (vm_pe_mips · u) seconds (the backstop's unit), and f_brown_ref = the arithmetic mean of the block's per-datacentre brown carbon factors (kg per Wh), read from the configuration once and written into the frozen scene file. Static power is excluded (dynamic energy only); no site choice and no arm enters. Implemented as a pure function with a unit test on a hand-computed trace; the value per window is written next to the window record before B or ST is read on it. The headroom gate of §2.2 reads (C_B − C_ST) ≥ 0.05 · C_brown_ref.

### A2. The real forecast error must be load-bearing before any RL

New gate §4.0, read on the six development windows after §2.2 and before §4.1, with the shrink arm recalibrated on the new turbines (`calibrated_shrink_hz_v2`, sister-turbine calibration excluded): C_shrink ≥ 1.05 · C_ST on the pooled sum, and C_shrink,w > C_ST,w on at least four of the six windows. Failure → STOP_ERROR_NOT_LOAD_BEARING: the calibrated error does not harm the analytic scheduler on this scene, so nothing about resisting it can be shown; the line stops before RL. The same arm later serves as a perturbation arm in §4.3 and §4.6, but this is the only gate in which it is a pass condition.

### A3. cand_green_cover: semantics narrowed, same-batch over-claim audited

The claim "two jobs cannot claim the same green" is withdrawn. The key's definition is: the share of job j's energy over its runtime, started at t + κ + lag at site d, covered by the arm's forecast green at d **after subtracting the site's static draw and the load already committed on the reservation grid before this decision** (routes and reservations of earlier steps). Jobs of the same decision batch see the same residual and may claim it jointly.

Audit, computed by the env at every step and exported per episode: for each (d, step) touched by the batch's chosen (d, κ) actions, over-claim = max(0, Σ_batch claimed green energy − residual green energy); exported as `ep_cover_overclaim_wh` (sum), `ep_cover_overclaim_ratio_p95` (per-step ratio of claimed to residual, 95th percentile) and `ep_cover_claimed_wh`. Interface-failure rule, frozen: on the executed behaviour-cloned arm of §4.5, if the realised green share of its jobs' energy (from the simulator's ledger) is below 0.5 × the mean cand_green_cover it acted on, the interface is judged to overstate available green and the verdict is STOP_INTERFACE_OVERCLAIM; this is an interface failure and is never attributed to the policy. The 0.5 is a proposal and is flagged.

### A4. Resource degradation is an ordered rule, not a choice

The smoke of §4.4 proceeds in this order and stops at the first step that passes:
1. dense float32 (the definition);
2. if peak RSS > 1.5 × D′ or throughput < 0.5 × D′: dense float16, admitted only with a test that the float16 key round-trips within 1e−3 of float32 on a saved observation set and that the legality mask is bit-identical;
3. if still over: STOP_RESOURCE_INTERFACE.
A sparse candidate representation is not an option of this document; it would need its own addendum before any use.

### A5. Seeds

§4.6's seeds are frozen now at five paired seeds. Machine availability changes the schedule, never the evidence standard.

---

## Addendum B (2026-09-06 00:20; the unit correction and the coverage weighting Codex required; the document is frozen at the commit that carries this addendum)

### B1. Unit of the brown factor

The block's `brown_carbon_factor` is in kg per kWh (the Java ledger divides Wh by 1000 before multiplying, `DatacenterInstance.updateEnergyMetrics`). A1's formula, corrected:

    E_dynamic_Wh  = Σ_jobs  pes_j · P_dyn_pe · u · runtime_j(sec) / 3600
    C_brown_ref_kg = E_dynamic_Wh / 1000 · f_brown_ref  [kg per kWh]

The unit test locks 1 Wh × 0.5 kg/kWh = 0.0005 kg. Implemented in `g1/compressed_timecap_s2/scene_v1.py` (`dynamic_energy_wh`, `c_brown_ref_kg`, `headroom_ok`), pure, tested.

### B2. Coverage on one footing

- The forecast coverage a policy sees and the audit's claimed coverage are weighted by each job's dynamic energy (pes · P_dyn_pe · u · runtime), never a plain mean over jobs.
- The realised coverage of the executed arm is the realised green share of the same dynamic job energy, from the simulator's ledger; static power never enters the denominator.
- When the forecast coverage of a job is zero the over-claim gate does not fire for it; the raw audit quantities are still recorded.
- The 0.5 threshold of A3 is approved: realised dynamic green coverage below half of the forecast coverage the policy acted on → STOP_INTERFACE_OVERCLAIM.

### B3. Freeze

With this addendum the design is frozen. Step 1 of §5 (turbine choice and data isolation) may start immediately; no other step runs before the previous one is closed and recorded.

---

## Addendum C (2026-09-06 00:30; Codex ruling on step 2a; written before any wind value of candidates 13–24 is read)

### C1. v1 outcome kept

Step 2a on the pool of twelve is STOP_WINDOW_SPLIT (STAGE_D_PRIME_DESIGN §37): five windows pass the headroom gate, six were required. The record stands permanently; it means the window pool was short, not that the forecast lacks value on the new turbines.

### C2. Scene-v2 continuation, same thresholds, same 2021 design set

- Candidates: the next twelve windows of the same hash sequence (positions 13–24 of `draw_windows(52559, 24, "scene-interface-v1:2021:")`), frozen with their offsets and footprints in `stage_a_out/scene_v2_candidates.json` before any of them is read.
- Procedure: in that order, for one candidate at a time, run B (`reactive_wait_planner`) and ST (godeye) with the contract check; apply the two unchanged headroom gates (relative ≥ 0.15, absolute ≥ 0.05 · C_brown_ref = 9.49e−4 kg on this trace); the first candidate that passes becomes the sixth development window and the search stops; candidates after it are not run.
- If none of the twelve passes: final STOP for this scene. No further extension, no threshold change, no switch to a per-window absolute rule.
- The five windows that passed in step 2a (pool k3, k5, k6, k8, k9: offsets 16477, 4240, 9154, 33225, 13223) are kept: they are 2021 design windows chosen by the frozen B/ST gate only and have served no RL, BC or offset selection.
- The mechanism control PASS and the TimeCAP audit v2 (λ ≈ 0.88 per DC) are inherited; the audit's parameters are frozen from now and are not recalibrated on candidates 13–24.
- The 2020 confirmation windows stay sealed.

### C3. Order after the sixth window

On the final six development windows: A2 error gate (calibrated shrink v2) → margin probe → P0′ → every-step offset expressibility (§4.1) → 73 fixed offsets + three blind arms, blind* frozen (§4.2) → predictive necessity (§4.3) → F1 / F2 / F3 (§4.5). The development rows of the six windows are regenerated as one set (`sc2_*` directories) so every gate reads one consistent layout.
