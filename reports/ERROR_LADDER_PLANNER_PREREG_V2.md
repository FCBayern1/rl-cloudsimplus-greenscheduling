# Error ladder on a certified exact planner, version 2 (draft for freeze, 2026-09-06)

Status: FROZEN by the commit that adds this paragraph (the freeze commit hash is recorded in the design log, STAGE_D_PRIME_DESIGN.md §47). Written after the settlement diagnostic closed every model-vs-simulator term (reports/SETTLEMENT_DIAG_2026_09_06.md §6); the solver feasibility gate of §5 passed before the freeze; no non-truth rung has been solved or read. Version 1 (reports/ERROR_LADDER_PLANNER_PREREG.md, Addenda A–E) is closed with two STOPs that stay on record (reports/manifests/ladder_v3/run1); nothing in this document reopens it. This is not an addendum to version 1.

## 1. Question and claim

Same question as version 1: on a scene where the exact planner's schedule is certified to settle on the simulator to within a tight tolerance, does forecast error move the settled carbon, by how much, and where on a controlled error ladder does the harm become load-bearing? The ladder is the instrument; RL and EU-CRD are not part of this document and may not start before the ladder's reading is frozen.

## 2. Certification twin (fixed)

- The development twin of the scene (six 2021 development windows, `scene_v2_dev.json`; the 2020 confirmation windows stay sealed) with two simulator settings that do not change the scene's physics: every datacenter's cloudlet-processing updates aligned to the 1 s step (`datacenter_scheduling_interval: 1.0`; the scene's default 0 leaves a 0.01 s scheduling constant that made a job started on a busy site finish one second late), and `min_time_between_events` verified equal to 1.0 (the clock grid). Generated on every call by `ladder_run.cert_config(mode)` into `config_ladder_cert_{defer,offset}.yml`; the scene's own configs are untouched.
- Settlement path: the every-step (DC, dispatch-offset) executor, dense grid 0..72 (`OFFSET_GRID_DENSE=1`), replay arm `schedule_replay`; the replay arm refuses any other grid.
- Every replay records its per-step observations; the planner's curve and the replay's green rows must agree row by row.

## 3. Planner model (MODEL_VERSION 2, `ladder_planner.py`)

- Jobs as the simulator presents them at first sighting (row a, PEs, MI, seconds to deadline); runtime r = ⌈MI / (vm_mips · util)⌉ rows; starts s ∈ [a + 2, D − r − 2]; the earliest start a + 2 is the executor's route-now latency (held jobs land exactly).
- Sites from the config: hosts H_d, VMs, host profile (RS500A_DYN 64 PEs / RS700A_DYN 128 PEs), P_job(d, p) = idle + (max − idle) · p · vm_mips / (host_pes · host_mips) in integer mW.
- Placement premise A: occupancy(d, t) = jobs running at t + jobs ending at t ≤ H_d on every row (linear constraint of the planner, fail-fast check of every settlement). Under it every VM id used is ≤ H_d − 1 (lowest-free-VM selection with a VM free one row after its job ends), so concurrent jobs sit on distinct hosts and the model charges one host per job. `placement_hosts` reproduces the simulator's VM and host (35/35 on k0).
- draw(d, t) = Σ_j P_job(d, p_j) x_j (no host variables); brown ≥ draw − G, brown ≥ 0; J = Σ 49·brown + draw (exact for factors 0.5 / 0.01); C_kg = J / 3.6e11. Quantisation: the curve's rounding to integer mW, bounded as in version 1 C3.
- Truth curve straight from the wind files: row(t) = episode offset + tz_d + t + ⌊min_time_between_events⌋ + 12 over the planner's whole horizon; no hold-last, no wrap (STOP past the file); signature recorded; the dump's observation rows must equal it (STOP otherwise).
- Equivalent tightening: on every cell whose candidate jobs draw the same P, the convex envelope of brown = max(0, P·n − G) over integer job counts n, i.e. brown ≥ (P(m+1) − G)(n − m) with m = ⌊G/P⌋. Valid for every integer solution (RHS − true brown = (Pm − G)(n − m − 1) ≤ 0), so no optimum changes; the relaxation becomes near-tight (k0: root-node optimal in 4 s where the plain formulation needed 2707 s and 40,552 nodes). Equivalence is unit-tested on random instances.
- Solver: HiGHS (scipy.optimize.milp), mip_rel_gap 0, one solve at a time, per-cell time limit T_solve (§5); compound OPTIMAL = HiGHS optimal ∧ verifier passes (one start per job, window, capacity, occupancy) ∧ |fun − J_int| < 0.5 ∧ J_int − dual bound < 1 ∧ finite gap ∧ premise. CP-SAT is a cross-check only.

## 4. Rungs, closure, gates (as version 1 unless stated)

- Rungs: truth; shrink λ ∈ {0.75, 0.5, 0.25, 0} around the site's full-year 2021 mean; seeded shuffle; anti. Each rung is optimised on its curve and settled on truth (model) and on the simulator (replay).
- Closure per (window, rung), all required: |C_sim − C_model| / C_model ≤ 3 % AND |draw_sim − draw_model| / draw_model ≤ 0.1 % AND |brown_sim − brown_model| ≤ 0.002 Wh AND every job on the planned site at the planned start AND every counter zero AND curve rows match. The three energy terms are reported separately so no term can hide under the composite (the settlement diagnostic reached 4e-7 on carbon; the brown bound is ten times the largest rounding seen).
- Order: the six truth rungs are solved and closed first; the other rungs are solved only after truth closure on every window; no reading of any non-truth carbon before that.
- Gates L1 (headroom vs λ = 0: rel ≥ 0.15 and abs ≥ 0.05·C_brown_ref) and L2 (a rung is load-bearing iff pooled loss ≥ 5 % of pooled truth carbon and ≥ 80 % of the headroom lies in windows with loss > 0) as in version 1. C_brown_ref = the all-brown carbon of the trace on the certification twin, 0.5 kg/kWh × 35 jobs × 48 s × 65.64 W = 0.015316 kg; 0.1 % of it (1.53e-5 kg) is the quantisation threshold (bound 2.28e-7 kg per window), 5 % (7.66e-4 kg) the absolute headroom gate.
- STOPs: any cell not compound-OPTIMAL within T_solve (no retry, no extension, no solver switch); any closure failure; curve mismatch; curve out of range; premise violation; fewer than 4 valid windows at L1.

## 5. Solver feasibility gate (precondition of the freeze)

Before this document is frozen, the six truth rungs of the certification dumps (`ladder_v4`) must be compound-OPTIMAL under T_solve = 3600 s, k1 included; their replays must close under §4. If k1 is not proven the document is not frozen and the next step is model tightening or a prospectively re-certified smaller workload (never a coarser offset grid), each as a new draft.

Result (reports/manifests/ladder_v4/gate, read before the freeze): all six truth rungs compound-OPTIMAL at the root node, HiGHS wall 3.7–4.4 s each (k1: J 370,233,055, 3.7 s; without the envelope cuts k0 needed 2707 s and 40,552 nodes); truth closure TRUTH_CLOSED on all six windows with carbon relative error ≤ 2.8e-7, draw relative error ≤ 3e-15, brown absolute error ≤ 4.7e-7 Wh, every job on its planned site and start, every counter zero, curve rows matching on every window. T_solve stays 3600 s.

## 6. Freeze

The freeze is the commit that records: this file, the gate result of §5, C_brown_ref, the certification config hashes, the jar hash, the model file hash, the dev offsets. After the freeze only append-only addenda may follow, and no addendum may weaken a tolerance, extend a time limit, switch a solver, or coarsen the grid.

Freeze record (sha256 prefixes at the freeze commit): `config_ladder_cert_defer.yml` 0c68bdc1dd5ae9f1; `config_ladder_cert_offset.yml` f2bd90fe8e4cee2a; gateway jar `cloudsimplus-gateway.jar` fd215aa03f8913e3 (source at the freeze commit); `ladder_planner.py` 9739537be240d593; `ladder_run.py` 0b24b5cad9b380cf; development offsets 16477, 4240, 9154, 33225, 13223, 49625 (2021); scipy 1.17.1 (HiGHS bundled); T_solve 3600 s; mip_rel_gap 0; run directory `stage_a_out/ladder_v4`; archive `reports/manifests/ladder_v4/`.
