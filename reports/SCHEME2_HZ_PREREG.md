# Scheme 2-HZ preregistration (append-only) — DRAFT pending Codex ruling

Status: DRAFT, 2026-09-03. Nothing below has been run on the discovery windows k=10/18 or on the confirmation windows. This file is frozen at the commit that Codex approves; later changes are addenda only.

## 0. Claim under test

At least one realistic, systematic forecast error (calibrated amplitude shrink) makes a truth-informed carbon-aware planner worse than a strong blind planner, on a fleet where the truth-informed planner is clearly better than that blind. Both halves are required. This is step 1 of the three-step chain (error hurts → vanilla RL inherits → EU-CRD resists); steps 2 and 3 are a separate Stage D prereg written only if this one passes.

## 1. Scenario (fixed; no cell, window, scarcity or fleet change after freezing)

- Fleet: the H fleet on zero-floor host twins `SPEC_ASUS_RS500A_DYN` / `SPEC_ASUS_RS700A_DYN` (1 W technical floor, 162.6 / 324 W dynamic span), hosts exposed as 32-PE VMs, `split_large_cloudlets: false`, `max_cloudlet_pes: 32`, `idle_host_power_down: true` (inherited), brown carbon factor 0.5 uniform, green 0.01. Config: `g1/compressed_timecap_s2/config_s2hz_m2.yml` (primary, divisor 3000) and `config_s2hz_m1.yml` (secondary, divisor 1500).
- Jobs: 32-PE, runtime 48 rows, wait cap 72 rows, deadline arrival+120, traces `traces/s2/<cell>_pes32.csv`, six cells c∈{1,3,5} × n∈{20,50}. Simulator draw per job 65.6 W (0.4 host utilisation from vm_pe_mips 40000 / host 50000).
- Planner environment: `PLANNER_EXPECTED_CAP=640;512;640;512;192`, `PLANNER_STATIC_TOTAL_W=0`, `PLANNER_PERTURB_E=1`. The default 332 W static floor is declared wrong for this fleet (awake hosts draw 1 W) and is disclosed as having been active in every earlier S2/E/F/H planner run.
- Primary scarcity: ×2 (pooled green ≈ load). Secondary: ×1, reported with the same gates, not required for PASS.

## 2. Windows

- Design windows already read: k=3, 4 (pilot_hz, 144 runs, `reports/PILOT_HZ_REPORT.md`), k=2 (H pilot on the legacy fleet; one HZ smoke cell c3_n50). Excluded from every formal table.
- DISCOVERY: k=2 (declared read), k=10, k=18. k=10/18 unread until the formal launch.
- CONFIRMATION: k=26, 34, 42. Sealed; read once, after all discovery gates pass, with no change of arm, cell, corruption, aggregation or threshold.

## 3. Arms and naming

- Blind candidates: `nowait_planner`, `reactive_wait_planner`, `reservation_edf`, `load_smoothing`. Run first on DISCOVERY; candidates must be contract-green on every cell × window; the frozen blind is the one with the lowest pooled discovery carbon, chosen once, never per instance, never re-chosen on CONFIRMATION.
- Clean arm: `perturbed_oracle_planner` tier `godeye`, named "truth-informed planner". Not called oracle, optimum, clairvoyant or EVPI bound (shuffle has beaten it in earlier scenes).
- Primary corruption: `calibrated_shrink_v1` (TIERS_E, audit-derived amplitude shrink, lead-0 exact), unchanged from `SCHEME2_ERROR_REGRET_PREREG.md` Addendum A.
- Negative controls: `shuffle`, `anti` (lead-0 exact).

## 4. Gates (frozen before any k=10/18 carbon is read)

- G0 contract, every run: completion 1.000 by MI, ontime 1.0, forced = stale = unplanned = wrong-DC = never-started = over-cap = 0, cloudlets = trace rows, created/started/finished closure, six arms share weather signature, workload hash and physical config per (cell, window). A failed run voids its (cell, window) for every arm.
- G1 clean load-bearing: truth-informed vs frozen blind, median per-(cell, window) normalised total carbon reduction ≥ 5%, favourable in ≥ 4/6 cells and ≥ 2/3 windows.
- G2 error harm: `calibrated_shrink_v1` vs truth-informed, total carbon worse by ≥ 5% or giving back ≥ 50% of the clean-vs-blind gain; same direction in ≥ 4/6 cells and ≥ 2/3 windows.
- G3 negative controls (strong form, registered now): shuffle and anti each retain ≤ 50% of the clean-vs-blind gain (median over cell × window), reported against both clean and blind.
- G4 confirmation: run once with identical arms, blind, cells, corruption, aggregation and thresholds; PASS requires G0–G3 on CONFIRMATION as well.

Aggregation: retention R = (I_blind − I_arm) / (I_blind − I_clean) on per-run carbon intensity Σcarbon / Σcompleted MI; medians over the 6 × 3 (cell, window) grid; pooled carbon reported alongside.

## 5. Stop rule

Failure on DISCOVERY or CONFIRMATION closes the S2 carbon-axis family as ruled by Codex (no ×1.5, ×0.75, capacity, window or fleet variants; no RL). Pass → Stage D prereg (matched no-forecast, vanilla clean/corrupt, EU-CRD clean/corrupt; 1 seed / 50k health smoke first) before any training.

## 6. Provenance

- Code: commits 6a5c2236 (host twins, toy model, planner static override), pilot_hz phase commit, d5713924 (pilot report). Tests: `ZeroFloorProfileTest` (3), `test_gen_hz.py` (2), `test_toy_lever.py` (6), `test_lever_decomp.py` (4), `test_planner_static_env.py` (3); full Java and Python suites green at 6a5c2236.
- Design evidence: `reports/FORECAST_LEVER_ROOT_CAUSE.md`, `reports/PILOT_H_REPORT.md`, `reports/PILOT_HZ_REPORT.md`, manifest `reports/manifests/PILOT_H_HZ_OUTPUTS.sha256`, toy predictions `reports/manifests/toy_hz_prediction_2026-09-03.md` (written before the pilot ran).
- Runner phases `hz_blind` / `hz_main` / `hz_confirm` to be added after the ruling, mirroring `e_blind` / `e_main`.

## 7. Spiral

This scenario keeps the host floor at 1 W. Turn 2 of the spiral (separate prereg) restores the 51.4 W floor with idle power-down and a packing-aware planner static term, and asks whether the lever survives. Turn 3 adds host heterogeneity back. Each turn is gated by the simulator-free model first.
