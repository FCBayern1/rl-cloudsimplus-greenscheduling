# Scheme 2-HZ preregistration (append-only)

Status: REVISED per Codex conditional approval of 2026-09-03 (R-f), frozen at the commit that carries this text. Nothing below has been run on the discovery windows k=10/18 or on the confirmation windows. Later changes are addenda only.

Scientific identity (Codex item 8): **accelerated-weather, marginal-carbon mechanism positive control.** The 1 W host floor is an artificial isolation device, not a server power model. A PASS shows that the forecast-error → harm → EU-CRD-recovery mechanism exists in a controlled scene; it does not by itself support a real-world energy-saving claim. External validity is the spiral's second turn (real 51.4 W floor), run only after Stage D.

## 0. Claim under test

At least one realistic, systematic forecast error (calibrated amplitude shrink) makes a truth-informed carbon-aware planner worse than a strong blind planner, on a fleet where the truth-informed planner is clearly better than that blind. Both halves are required. This is step 1 of the three-step chain (error hurts → vanilla RL inherits → EU-CRD resists); steps 2 and 3 are a separate Stage D prereg written only if this one passes.

## 1. Scenario (fixed; no cell, window, scarcity or fleet change after freezing)

- Fleet: the H fleet on zero-floor host twins `SPEC_ASUS_RS500A_DYN` / `SPEC_ASUS_RS700A_DYN` (1 W technical floor, 162.6 / 324 W dynamic span), hosts exposed as 32-PE VMs, `split_large_cloudlets: false`, `max_cloudlet_pes: 32`, `idle_host_power_down: true` (inherited), brown carbon factor 0.5 uniform, green 0.01. Config: `g1/compressed_timecap_s2/config_s2hz_m2.yml` (primary, divisor 3000) and `config_s2hz_m1.yml` (secondary, divisor 1500).
- Jobs: 32-PE, runtime 48 rows, wait cap 72 rows, deadline arrival+120, traces `traces/s2/<cell>_pes32.csv`, six cells c∈{1,3,5} × n∈{20,50}.
- Power semantics (Codex item 6), pinned by `ZeroFloorSentinelTest` on a real simulation to 1e-9 W: host floor 1.0 W; one 32-PE job at vm_pe_mips 40000 on 50000-MIPS host cores drives host utilisation 0.4 and draws **65.64 W**; two jobs 130.28 W. The toy model's `P_DYN_W` is this 65.64 W. The idealised 81.3 W (half the 162.6 W span, equal VM and host MIPS) appeared only in the 192-configuration structural sweep of the toy; 132.7 W is the legacy RS500A figure (51.4 W floor + 81.3 W) and belongs to the H fleet, not HZ.
- Planner environment: `PLANNER_EXPECTED_CAP=640;512;640;512;192`, `PLANNER_STATIC_TOTAL_W=0`, `PLANNER_PERTURB_E=1`, `PLANNER_PERTURB_CAL=timecap_error_audit.json`. The default 332 W static floor is declared wrong for this fleet (awake hosts draw 1 W) and is disclosed as having been active in every earlier S2/E/F/H planner run (impact note appended to those reports, verdicts unchanged, per R-e).
- Hidden quantities are not trusted to the shell (Codex item 7): every result row carries `planner_static_total_w` and `planner_expected_cap` reported by the planner itself, the verdict fails a run whose values differ from the registered ones, and the phase manifest (`hz_manifest`) records code commit, jar SHA256, config SHA256, audit-file SHA256, TIERS_E parameters and the full per-job environment.
- Scarcity (Codex item 3): **×2 is the only verdict scene.** ×1 is a secondary replication reported with the same gates; a ×1 result can neither rescue nor overturn the ×2 verdict.

## 2. Windows (Codex item 1)

- k=3, 4: DESIGN_PILOT windows (pilot_hz, 144 runs, `reports/PILOT_HZ_REPORT.md`). Permanently excluded from every formal table.
- k=2: a DISCOVERY window that has already been read (H pilot on the legacy fleet, all six cells; one HZ smoke cell c3_n50 ×1/×2). It participates in DISCOVERY and is labelled "read" in every table.
- k=10, 18: unread DISCOVERY windows until the formal launch.
- k=26, 34, 42: CONFIRMATION, sealed; read once, after all discovery gates pass, with no change of arm, cell, corruption, aggregation or threshold.

## 3. Arms and naming (Codex item 2)

Eight arms run; five enter the main comparison after the blind freeze.

- Blind candidates (4): `nowait_planner`, `reactive_wait_planner`, `reservation_edf`, `load_smoothing`. Run first on DISCOVERY; a candidate must be contract-green on every cell × window; the frozen blind is the one with the lowest pooled discovery carbon, chosen once, never per instance, never re-chosen on CONFIRMATION.
- Clean arm: `perturbed_oracle_planner` tier `godeye`, named "truth-informed planner". Not called oracle, optimum, clairvoyant or EVPI bound (shuffle has beaten it in earlier scenes).
- Primary corruption: `calibrated_shrink_v1` (TIERS_E, audit-derived amplitude shrink, lead-0 exact), unchanged from `SCHEME2_ERROR_REGRET_PREREG.md` Addendum A.
- Negative controls: `shuffle`, `anti` (lead-0 exact).

## 4. Gates (frozen before any k=10/18 carbon is read; Codex item 4)

Notation: per run, carbon intensity I = Σcarbon / Σcompleted MI. Pooled intensity over the DISCOVERY grid (6 cells × 3 windows) I_pool = Σcarbon / ΣMI over all valid runs of an arm. Retention of an arm against the frozen blind:

    R_pool(arm) = (I_blind,pool − I_arm,pool) / (I_blind,pool − I_clean,pool)

Per-run retention R_cw uses the same formula on one (cell, window); when its denominator is not strictly positive it is recorded as undefined and excluded from medians (never written as zero).

- G0 contract, every run of all eight arms: completion 1.000 by MI, ontime 1.0, forced = stale = unplanned = wrong-DC = never-started = over-cap = 0, cloudlets = trace rows, created/started/finished closure, all arms share weather signature, workload hash and physical config per (cell, window), `planner_static_total_w` = 0 and `planner_expected_cap` = registered vector on every planner-family row. A failed run voids its (cell, window) for every arm; voided grids may not be used to satisfy a direction count.
- G1 clean load-bearing: truth-informed vs frozen blind, (I_blind,pool − I_clean,pool) / I_blind,pool ≥ 5% and median per-(cell, window) reduction ≥ 5%; favourable in ≥ 4/6 cells (a cell is favourable when the median over its windows is) and in ≥ 2/3 windows (a window is favourable when the median over its cells is). G1 makes the retention denominator positive.
- G2 error harm: `calibrated_shrink_v1` vs truth-informed: I_shrink,pool ≥ 1.05 × I_clean,pool, or R_pool(shrink) ≤ 0.5; and the same direction (shrink worse than clean) in ≥ 4/6 cells and ≥ 2/3 windows.
- G3 negative controls (strong form): R_pool(shuffle) ≤ 0.5 and R_pool(anti) ≤ 0.5, each also reported against the blind (whether I_ctrl,pool ≥ I_blind,pool).
- G4 confirmation: run once with identical arms, frozen blind, cells, corruption parameters, aggregation and thresholds; PASS requires G0–G3 on CONFIRMATION as well.

Mechanical reader: `g1/compressed_timecap_s2/hz_verdict.py` (tests in `test_hz_verdict.py`), frozen with this text; it emits INVALID_INCOMPLETE_DATA rather than a verdict when any expected run is missing.

## 5. Stop rule

Failure on DISCOVERY or CONFIRMATION closes the S2 carbon-axis family as ruled by Codex (no ×1.5, ×0.75, capacity, window or fleet variants; no RL). Pass → Stage D prereg (matched no-forecast, vanilla clean/corrupt, EU-CRD clean/corrupt; 1 seed / 50k health smoke first) before any training.

## 6. Provenance

- Code: commits 6a5c2236 (host twins, toy model, planner static override), 3086aae2 (pilot_hz phase), d5713924 (pilot report), and the commit carrying this revision (sentinel test, hz phases, verdict reader, planner-reported hidden quantities). Tests: `ZeroFloorProfileTest` (3), `ZeroFloorSentinelTest` (1, real simulation), `test_gen_hz.py` (2), `test_toy_lever.py` (6), `test_lever_decomp.py` (4), `test_planner_static_env.py` (4), `test_hz_verdict.py` (8).
- Design evidence: `reports/FORECAST_LEVER_ROOT_CAUSE.md`, `reports/PILOT_H_REPORT.md`, `reports/PILOT_HZ_REPORT.md`, manifest `reports/manifests/PILOT_H_HZ_OUTPUTS.sha256`, toy predictions `reports/manifests/toy_hz_prediction_2026-09-03.md` (written before the pilot ran).
- Runner phases in `run_stage_a.py`: `hz_manifest` (writes `stage_a_out/hz_manifest_m2.json`: commit, worktree state, jar / config / audit / perturb-module / planner-module SHA256, per-job environment), `hz_blinds` (72 runs), `hz_freeze`, `hz_main` (72 runs, refuses to start without a FROZEN blind), `hz_confirm` (refuses to start without `PASS_HZ_DISCOVERY`). Verdict: `hz_verdict.py discovery|confirmation`.

## 7. Spiral

This scenario keeps the host floor at 1 W. Turn 2 of the spiral (separate prereg) restores the 51.4 W floor with idle power-down and a packing-aware planner static term, and asks whether the lever survives. Turn 3 adds host heterogeneity back. Each turn is gated by the simulator-free model first.

## Addendum A (2026-09-03, after the confirmation run; disclosed, not a change of criteria)

The first build of `hz_verdict.py` returned INVALID_INCOMPLETE_DATA whenever any (cell, window) was not fully contract-green, which contradicts section 4 G0 as written ("a failed run voids its (cell, window) for every arm; voided grids may not be used to satisfy a direction count"): only a missing run or a planner-environment drift is invalid data. The discrepancy surfaced on the CONFIRMATION set, where one run (calibrated_shrink_v1, cell c5_n50, window k=42) finished all 50 jobs but with ontime 0.98 (one job late, forced 0; the other four arms on that grid were 1.0). The reader was corrected to the registered rule (commit carrying this addendum; tests `test_contract_failure_voids_the_grid_and_the_verdict_proceeds`, `test_voided_grids_cannot_satisfy_direction_counts`), and the DISCOVERY verdict is unchanged by the fix (18/18 grids valid). Both readings of CONFIRMATION are reported in `reports/SCHEME2_HZ_RESULTS.md`: the registered voiding rule (17 grids, thresholds still out of 6 cells and 3 windows) and the strict "every run contract-green" reading, which the first build enforced and which the prereg text did not register.
