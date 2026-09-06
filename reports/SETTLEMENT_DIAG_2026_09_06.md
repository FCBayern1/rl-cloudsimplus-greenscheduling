# Settlement diagnostic: where the k0 model-vs-simulator gap comes from (2026-09-06)

Scope (Codex ruling of 2026-09-06 after the Addendum E stop): the fixed k0 optimal schedule, no re-optimisation, no new tolerance, no scientific verdict. Question: attribute the simulator's extra 0.47 Wh of draw and 0.068 Wh of brown on k0 to mechanisms, per second and per DC, with probes that isolate each mechanism.

Artefacts: `reports/manifests/settle_diag/` (k0_attribution.json, probes/*_align.json, traces, config), code `g1/compressed_timecap_s2/settle_probe.py` (+ tests), the per-step alignment `stage_a_out/settle_diag/k0_align.npz` (model W, simulator W, green W per site and step).

## 1. Result: the gap is fully attributed, three mechanisms plus one harness defect

k0 draw gap 0.4695 Wh (simulator 30.900, model 30.431):

| bucket | draw Wh | share | brown Wh | mechanism |
|---|---|---|---|---|
| A. extra active hosts | +0.2056 (740 host-steps × 1 W) | 44 % | +0.0080 | the simulator runs every job on its own host; the model packs ⌈PEs/64⌉ hosts per site |
| B. extra job-steps | +0.2693 (15 job-steps × 64.64 W) | 57 % | +0.0474 | a job that starts while its site is mid-cycle is sampled for 49 s instead of 48 |
| D. DC2 host model | −0.0054 | −1 % | (in interaction +0.0013) | DC2 is RS700A (128 PEs, 64.6 W per 32-PE job) while the model uses 64-PE/2.02 W-per-PE everywhere |
| C. truth curve tail | 0 | | +0.0117 | the planner's truth curve beyond the dump's 599 rows is hold-last extrapolated; the replay ran to row 638 with real wind |
| total | +0.4695 | 100 % | +0.0684 | matches the ledger (0.7061 − 0.6377) |

Brown attribution uses counterfactual settlement of the same green curve: model + A only, model + B only, simulator with the planner's curve, simulator with its own curve. B carries most of the brown because the optimiser ends jobs exactly where green runs out, so the 49th second of an affected job is a brown second.

## 2. Evidence by level

Simulator side: `dc_current_power_w` per step per DC (the per-step sample sums to the ledger's total_energy_wh to four decimals, so energy = sample × 1 s, no integration-phase issue). Each sample is decomposed as P = hosts × 1 W + jobs × 64.64 W (RS500A_DYN: 1 W floor + 161.6 W × u, u = 32 × 40000 / (64 × 50000) = 0.4; residual < 0.16 W everywhere, 0.04 W per job on DC2).

Level 1, one job (probe L1): identical to the model, 48 samples at the planned rows 20..67, 65.64 W, 1 host.

Level 2, two jobs, same site and start (L2): model 130.28 W (one 64-PE host holds both 32-PE VMs); simulator 131.28 W, i.e. two hosts. VmAllocationPolicyCustom places the 20 VMs of a site across its 10 hosts, and the placement ledger takes the most-free fitting VM, so two concurrent jobs always land on different hosts. On k0 the simulator ran one host per job at every step (max 5 concurrent jobs per site out of 8–10 hosts), so the model under-counts hosts by (jobs − ⌈jobs/2⌉) at every step: 740 host-steps.

Level 3, staggered and fragmented (L3a: starts 20 and 40; L3c: 20, 20, 30): packing as in L2, plus the second effect: the job started while another was running is sampled one row longer (97 job-steps vs 96; 145 vs 144). Two jobs starting together on an idle site (L2, and jobs 0/1 of L3c) do not get the extra row. L3b (third job after the first two end) turned out to be a probe artefact: a planned start more than 73 rows after the first sighting is clipped by the replay arm to κ = 72 (never happened on k0, whose per-job closure passed), so it is reported but not used.

Level 4, k0: rule B "a job whose start step falls while another job on the same site, started strictly earlier, is still running or finishes at that very step gets 49 samples" is exact on all 35 jobs: 15 such jobs, 15 jobs with the extra end sample, the same 15 (k0_attribution.json → rule_B). Rule A "one host per job" holds at every (site, step).

Mechanism of B (from the observation, not yet from the CloudSim source): the cloudlet's finish is processed at the next 1-second scheduling tick after its true end when its execution was interleaved with a datacenter processing cycle already in progress, so the host still reports utilisation at the sample taken at the true end. The actual start rows are exact (the option ledger shows integer starts, route_to_start 0), so B is an end-boundary effect only.

## 3. The harness defect (C) and what it does to the planner

`ladder_run.cmd_solve` builds the truth curve from the dump run's observations and hold-last extrapolates from the dump's last row (599) to the planning horizon (669). The replay ran to row 638 with the real wind. Effect on k0: +0.0117 Wh brown at settlement, and, worse, the planner optimised the tail of the window against a curve that is not the wind. The wind is deterministic (CSV + offset), so the truth curve must be built from the wind files for the full horizon, or the dump run must be forced to cover it. This is a harness fix, not a model question; it was not applied (nothing changed under the frozen preregistration).

## 4. What each mechanism means for a certifier (facts for Codex, no decision taken)

- A is a modelling mismatch of the site's host model with the simulator's placement policy. Either the model charges min(jobs, hosts) hosts (one per job, what the simulator does) or the simulator consolidates (a placement that prefers a VM on an already-active host). Both are exact fixes; neither was applied.
- B is a one-sample end-boundary effect of the simulator (2 % of a 48-s job for affected jobs). It can be modelled (runtime + 1 for jobs starting on a busy site, which is schedule-dependent and non-linear for the planner) or removed on the simulator side (finish processing at the true end); the root cause in CloudSim's processing cycle is not yet located.
- C must be fixed in the harness before any new ladder.
- D is small; a per-site host profile in the model (128-PE hosts on DC2) removes it.
- With A, B, C, D closed on the micro probes first, the 3 % composite tolerance is reachable in principle; nothing here argues for widening it.

## 6. Addendum (same day, under the user's direct authority): A–D closed, source-level B, exact closure

Ruling applied: A model follows the simulator; B source trace first; C and D mandatory; then engineering closure without re-solving.

**B, source level.** Per-step trace of the gateway (SETTLE_TRACE_CSV, new) on probe L3a: the lone cloudlet's progress is never updated between submission and its estimated-finish event (finish 68.0 exact, 48 s). The cloudlet submitted at 40.0 while the other runs is updated at the other's finish (68.0, credited 28 s), then at 69.01, then at 87.99 (credited 18.98 s, 401 MI left), and its finish is stamped 89.0: 49.0 s of execution and 49 utilisation samples. CloudSim Plus 8.5.5 source: `DatacenterSimple.updateHostsProcessing` clamps the next update delay to `minTimeBetweenEvents + 0.01`, `isTimeToUpdateCloudletsProcessing` refuses updates within `minTimeBetweenEvents` (1.0 s in the scene) of the last, and the datacenter's scheduling interval is left at 0 (updates only at estimated finishes), so a 0.01 s sliver plus the `(long)` truncation of partial MI leaves 401 MI that cannot be processed before the next permitted second. Reproduced in pure CloudSim (`CloudletBoundaryTraceTest`): min-time 1.0 and interval 0 give 48 s / 49 s; interval 1.0 gives 48 s / 48 s. Classification: a discrete-update artefact at the event-spacing boundary, not 49 s of work (the MI is complete at 88.0). Fix: the certification twin sets every datacenter's scheduling interval to the 1 s step (`datacenter_scheduling_interval`, new config key, default 0 = legacy so the scene's own configs are bit-identical); `min_time_between_events` stays 1.0 because the clock grid depends on it (a trial with 0.01 s shifted the clock of every observation by 0.99 s and every start by one row, and was discarded).

**C.** `truth_curve` builds the curve from the wind files: row(t) = episode offset + tz + t + ⌊min_time_between_events⌋ + 12, the last term being the 12 rows `GreenEnergyProvider.loadCsvData` drops in COMPRESSED mode before building the time axis, the clock term measured (step 3 at clock 4.0) and confirmed by the per-step trace (`green_row` column). Exact on every step of all six development windows (max 3e-5 W, float32). Signatures on both sides, per-row equality at closure, STOP past the file's end (no wrap, no hold-last).

**D.** Per-site host profile from the config: RS500A_DYN 64 PEs (65,640 mW per 32-PE job), RS700A_DYN 128 PEs (65,600 mW); DC2 and DC3 are RS700A.

**A.** VM i of a site sits on host i mod H (trace: DC0 VMs 0–4 on hosts 0–4, DC1 VMs 20–23 on hosts 0–3, DC2 VMs 36–39 on hosts 0–3); the ledger takes the lowest free VM, and a VM whose job ends at row e is free again from row e + 1 (at row e the finish and the new routing share the clock: k0 job 28, starting at 522 as jobs 26/27 end there, took VM 2, not VM 0; the same-row variant of the rule matched 34/35 jobs, the one-row-later rule 35/35, `k0lag2_placement_check.json`). Theorem: a job takes VM id j only when VMs 0..j−1 are busy or just freed, so every VM id used is ≤ occupancy − 1 with occupancy(d, t) = jobs running at t + jobs ending at t; with occupancy ≤ H all concurrent jobs sit on distinct hosts. The premise "occupancy ≤ H_d" is a constraint of the planner (running and ending jobs both count on a row) and a fail-fast check of every settlement, and the model charges one host per job. Property test: 200 random schedules.

**LAG.** The only residual of the first certification closure: the two jobs the planner started at sighting + 1 (route-now path) executed at sighting + 2 (4 cells of ±65.64 W, draw conserved, brown +0.019 Wh). Held jobs land exactly. The planner's earliest start is now sighting + 2.

**Engineering closure (fixed schedules, no solve), certification twin, model version 2:**

| case | draw model / sim Wh | brown model / sim Wh | carbon rel err | per-step max diff |
|---|---|---|---|---|
| probes L1, L2, L3a, L3b, L3c | equal | equal (same-run curve) | 0 | 0 |
| k0 archived v1 schedule | 30.62667 / 30.62667 | 0.64685 / 0.66569 | 1.48 % (passes 3 %, but the LAG residual is visible) | 65.64 W on 4 cells |
| k0 v1 schedule with the two route-now jobs at sighting + 2 | 30.62667 / 30.62667 | 0.665695 / 0.665694 | 4.3e-7 | 1.2e-5 W |

Every term is closed individually; nothing cancels. Legacy-twin attribution (section 1) is archived as produced with model version 1.

**Solver feasibility gate.** Version 2 has no host variables (draw is linear in x) and the concurrency constraint; the six truth rungs are being solved on the certification dumps (ladder_v4, HiGHS, 3600 s, gap 0). The gate is: k1 proven.

## 5. Not done, by ruling

No re-solve, no re-registration, no new tolerance, no reading of the sealed 2020 windows, no RL. The 26 pre-existing unrelated drl-manager test failures (gateway-dependent PettingZoo tests, logback config, SQT2 preflight, plotting, checkpoint paths) are still listed and still unrelated; the ladder, replay-arm and probe tests pass (16 + 4).
