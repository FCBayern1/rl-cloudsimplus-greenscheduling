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

## 5. Not done, by ruling

No re-solve, no re-registration, no new tolerance, no reading of the sealed 2020 windows, no RL. The 26 pre-existing unrelated drl-manager test failures (gateway-dependent PettingZoo tests, logback config, SQT2 preflight, plotting, checkpoint paths) are still listed and still unrelated; the ladder, replay-arm and probe tests pass (16 + 4).
