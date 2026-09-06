# Forecast-quality ladder on a dominance-safe planner (preregistration draft v0, 2026-09-06 01:20; for the user, then Codex; nothing frozen, nothing run)

Follows the Codex ruling of STAGE_D_PRIME_DESIGN §40. Keeps the scene of SCENE_INTERFACE_DESIGN (turbines 133/78 | 22/81 | 94, the six 2021 development windows 16477, 4240, 9154, 33225, 13223, 49625, the sealed 2020 confirmation windows). Replaces the single calibrated-shrink error arm and the heuristic ST reference by a controlled error ladder measured against an exact, dominating schedule. Zero RL until §6.

## 1. What is being measured

For a window and a curve X (truth, or a degraded forecast), the loss of scheduling on X is

    loss(X) = carbon( schedule(X) settled on truth ) − carbon( schedule(truth) settled on truth ),

where schedule(·) is one exact solver. Reported as a curve over the ladder: forecast quality on the x-axis, loss on the y-axis, per window and pooled. No rung is selected by its loss; the whole curve is the result.

## 2. The dominance-safe planner

### 2.1 Model
Time-indexed assignment, steps of one simulation second (the compressed rows), per window.
- Jobs j: arrival a_j, runtime r_j = ceil(mi_j / (mips · u)) steps, PEs p_j, latest start L_j = deadline_j − r_j − ε (ε = 2, the frozen margin). Decision x[j, d, s] ∈ {0, 1}, s ∈ [a_j + lag, L_j], one per job.
- Capacity: Σ_j Σ_{s ≤ t < s + r_j} p_j · x[j, d, s] ≤ cap_d for every site d and step t (cap = the VM PE count of the fleet: 640, 512, 640, 512, 192).
- Power: load_{d,t} = P_dyn_pe · u · Σ p_j x (W), P_dyn_pe = (214 − 51.4)/64 W, static 0 on the zero-floor fleet. brown_{d,t} ≥ load_{d,t} − G^X_{d,t}, brown ≥ 0; green_{d,t} = load − brown.
- Objective: Σ_{d,t} (f_brown,d · brown_{d,t} + f_green,d · green_{d,t}) / 3600 / 1000 [kg], the block's factors (0.5 and 0.01 kg/kWh). Because f_green < f_brown this is the carbon the simulator's ledger charges under the same power model.
Solver: CP-SAT (ortools 9.15, installed) with integer-scaled costs, time limit 600 s per window and curve, optimality gap recorded; a solution counts only with gap ≤ 1 % (else the rung is INVALID_SOLVER for that window). Size on the HZ cell: 35 jobs × 5 sites × ≤ 73 starts ≈ 12.8k binaries, 5 × ~600 capacity rows.

### 2.2 Settlement
schedule(X) is a list of (job, site, start). It is settled on truth twice: (i) by the model's own objective with G = truth (model settlement); (ii) by the simulator, replayed through the every-step (DC, dispatch-offset) executor with the fixed starts as offsets (simulator settlement, the existing `offset_v1` path with its ledger and route→start check).

### 2.3 Dominance and closure (gate 0, read on truth before any ladder rung)
- Model dominance is by construction: schedule(truth) minimises the model's truth-settled objective, so model-loss(X) ≥ 0 for every X up to the solver gap.
- Closure: on each development window, simulator settlement of schedule(truth) must agree with model settlement within 3 % of the model value, and the simulator must start every job at its scheduled step (route→start ≤ 1 step, forced 0, contract clean). A violation is a model error (power model, capacity, timing) and is fixed in the model, never by changing an arm; the fix is disclosed and gate 0 reruns; a second failure stops the line.
- Simulator dominance check: for every rung X, simulator-loss(X) ≥ −3 % · carbon(truth). A wrong curve beating the truth schedule in the simulator by more than the closure tolerance is a model defect and stops the ladder until closed.

## 3. The error ladder (frozen before any carbon; forecast side only)

- Anchor (realistic): the deployed TimeCAP checkpoint at the deployed lead, producing, per window, the curve the planner sees through the same rolling inference the audit used (`timecap_error_audit.py` path).
- Realistic rungs: the same checkpoint at longer leads, L ∈ {deployed, +24 rows, +48 rows, +96 rows}, chosen now; their forecast quality is measured on the 2021 design year by RMSE against truth (no carbon read) and is the x-coordinate. If a weaker checkpoint is wanted later, it is trained and frozen on validation loss alone, in its own addendum, before it produces any curve.
- Controlled rungs: shrink toward the site's frozen full-year 2021 mean μ_d (the audit's definition), G^λ = μ + λ (G^truth − μ), λ ∈ {1.0, 0.75, 0.5, 0.25, 0}; λ = 1 is the truth rung and λ = 0 the persistence-like flat rung.
- Extreme controls: shuffle and anti of the truth curve (the existing tiers), reported on the same curve, never described as deployed error.
- x-axis for every rung: normalised RMSE of the curve against truth over the window's horizon; y-axis: model loss and simulator loss.

## 4. Gates on the development windows (order fixed; zero RL)

- Gate 0 (§2.3): closure and dominance on truth. STOP_PLANNER_CLOSURE on failure.
- Gate L1, headroom: the no-forecast reference is schedule(flat μ) (λ = 0); headroom_w = carbon(schedule(λ=0)) − carbon(schedule(truth)), simulator-settled. A window with headroom below 15 % of the flat schedule's carbon or below 0.05 · C_brown_ref (the frozen reference of the scene design, 9.49e−4 kg) is INVALID for the ladder; fewer than four valid windows → STOP_LADDER_HEADROOM.
- Gate L2, the load-bearing rung: a rung X is load-bearing iff pooled loss(X) ≥ 5 % of pooled carbon(truth) and Σ_w headroom_w · [loss_w(X) > 0] / Σ_w headroom_w ≥ 0.80 (the headroom-weighted rule of §40 item 6). The report states, for the anchor and for every rung, whether it is load-bearing; the claim available to the thesis is fixed by the weakest load-bearing rung: anchor or a realistic lead → "resists realistic forecast degradation"; a shrink rung λ ≤ 0.5 only → "resists moderate controlled degradation"; shuffle/anti only → "resists controlled severe contamination"; none → the scene cannot support the thesis and the line stops.

## 5. What is not decided here (asked of Codex)

1. The lead set L, the 3 % closure tolerance, the 1 % solver gap, the 600 s limit.
2. Whether the every-step offset executor is the right settlement path for schedule(X) (it fixes starts; the simulator's local placement is the repaired ledger of §34).
3. Whether the RL/EU-CRD preregistration that follows (own document) trains on the offset action with the ladder's rungs as its forecast inputs and judges once on the sealed 2020 windows.

## 6. Cost and order

Planner + closure (CP-SAT model, settlement through the executor, tests): about one day. Ladder curves on six windows × (4 realistic + 5 controlled + 2 extreme) rungs: solver time bounded by 11 × 6 × 600 s worst case, usually far less; simulator settlement 66 replays, about an hour. No carbon run before Codex freezes this document.

---

## Addendum A (2026-09-06 01:50; closes the three points of the Codex review of v0; where it conflicts with §1–§6 the addendum governs; the document is frozen at the commit that carries it)

### A1. The power model is the simulator's zero-floor host model, not a per-PE constant

§2.1's "P_dyn_pe = (214 − 51.4)/64 W per PE, static 0" is withdrawn. The simulator charges each host through the RS500A_DYN profile (`HostProfile.SPEC_ASUS_RS500A_DYN`, CloudSim's utilisation-linear host power model): a host with no running work draws 0 W (idle power-down); a host with running work draws

    P_host = 1 W + 161.6 W × u_host,   u_host = Σ_{VMs on the host} PEs_used × 40000 / (64 × 50000),

with 64 PEs per host at 50,000 MIPS and VM PEs at 40,000 MIPS (10 hosts = 640 PEs on DC0, matching the fleet capacity). One 32-PE job on an otherwise idle host therefore draws 1 + 161.6 × 0.4 = 65.64 W (the earlier micro-measurement), not 32 × 2.54 = 81.3 W; the v0 model overstated job power by about 24 % and would have failed its own 3 % closure. (The profile's own source comment quotes 81.8 W for that job, i.e. utilisation counted as PEs rather than MIPS; the measurement and the comment disagree, and the closure gate of §2.3 on the truth schedule is what settles which utilisation the simulator actually charges before any rung is read.)

Model in the planner: load power at site d and step t = Σ_running jobs p_j × 2.02 W (= 40000/50000 × 161.6/64 W per PE) + 1 W × H_{d,t}, where H_{d,t} is the number of active hosts. H is an approximation term: the planner sets H_{d,t} = ceil(Σ_running p_j / 64), the packing that the repaired most-free-fitting placement approaches but does not guarantee. This term is stated as an approximation, is bounded by the 3 % model–simulator closure gate of §2.3, and the document does not claim "the same power model as the simulator" beyond that gate. Static draw of the fleet is 0 by the idle power-down, as before.

### A2. Optimality: the truth schedule must be proven optimal

- Truth rung: CP-SAT must return OPTIMAL (gap 0) within the 600 s limit on every required window; otherwise STOP_SOLVER_TRUTH_UNRESOLVED for that window, and a required window that is unresolved is never dropped or replaced (the line stops until the model is made solvable, disclosed).
- Error rungs (shrink, shuffle, anti): a gap ≤ 1 % is accepted and recorded; the settled loss then carries a ±gap uncertainty band in the report.
- The 600 s limit, the 3 % closure tolerance and the every-step offset executor as the simulator settlement path are approved and frozen.

### A3. Forecast issuance: the exact offline ladder is a controlled-error mechanism experiment

An offline solver needs one curve covering the whole schedule horizon. The deployed TimeCAP issues a fresh 144-step forecast at every decision time; stitching forecasts issued at future times into one horizon curve would hand the offline solver information that a causal policy never has, and one checkpoint's 144 steps cannot even cover wait cap 72 + runtime 48 = 120 steps at any longer lead. The lead set {deployed, +24, +48, +96} is withdrawn.

Frozen:
- The exact ladder of §3 contains only truth (λ = 1), the fixed shrink rungs λ ∈ {0.75, 0.5, 0.25, 0} around the frozen 2021 site means, and the extreme controls shuffle and anti of the truth curve. It is a controlled-error mechanism experiment and is described as such; its x-axis is the rung's normalised RMSE against truth.
- The deployed TimeCAP remains the natural error anchor only through the causal rolling heuristic planner (the ST-style arm of the scene design, forecasts issued at each decision time), reported next to the ladder as a descriptive reference; it is not part of the dominance proof and is not presented as the same information condition as an offline curve. A longer-horizon checkpoint, if ever trained, is frozen on validation loss in its own addendum before it produces any curve.
- The per-window offline solver knows every arrival in the window. Its schedules certify whether an error is load-bearing; they are not an online-achievable policy and are not used as leak-free behaviour-cloning labels.

### A4. Claim ladder, restated for the exact rungs (see Addendum B for the optimality and closure rules that now apply to every rung)

Weakest load-bearing rung under L2 → the claim: λ = 0.75 → "resists mild controlled degradation"; λ ≤ 0.5 → "resists moderate controlled degradation"; only shuffle / anti → "resists controlled severe contamination"; none → the scene does not support the thesis. Whether the deployed forecast's natural error is itself load-bearing is reported from the causal anchor separately, in the narrowed wording of STAGE_D_PRIME_DESIGN §39, and is not merged with the exact ladder's claim.

---

## Addendum B (2026-09-06 02:10; closes the two remaining points of the Codex review of Addendum A and locks two implementation constants; the document is frozen at the commit that carries it, after the final mechanical check)

### B1. Every rung must be proven optimal; no uncertainty band

A solver gap under a wrong curve X is a gap in X's objective, not in the truth-settled loss, so it cannot be propagated as a ±band on the reported loss. A2's "gap ≤ 1 % for error rungs" and its uncertainty band are withdrawn. All seven rungs of the exact ladder (truth, shrink λ ∈ {0.75, 0.5, 0.25, 0}, shuffle, anti) must return OPTIMAL within the 600 s limit on every required window; any required rung that does not is STOP_SOLVER_RUNG_UNRESOLVED for the line, and no window is dropped or replaced. The ladder is "exact" only under this rule.

### B2. Closure is checked on every rung's schedule, not only on the truth schedule

The active-host term H = ceil(Σ PEs / 64) is a packing approximation, and different curves pack differently, so closure on the truth schedule says nothing about a shrink or shuffle schedule. For every rung X and every required window, schedule(X) is settled on truth twice and must satisfy

    | C_sim(schedule_X; truth) − C_model(schedule_X; truth) | / C_model(schedule_X; truth) ≤ 3 %,

together with per-job start alignment (route→start ≤ 1 step for every job), forced = 0, completion ≥ 0.995 and on-time ≥ 0.995. Any rung that fails is STOP_PLANNER_CLOSURE_RUNG and the ladder does not enter gate L2; the failure is a model defect, fixed in the model and disclosed, never by changing the rung. Gate 0 (§2.3) is thereby the truth rung's instance of this rule, not a substitute for it.

### B3. Integer scaling of the CP-SAT objective, with a precomputable quantisation bound

Powers are integers in milliwatts: a job's dynamic draw is p_j × 2020 mW (2.02 W per PE), the active-host floor 1000 mW per host, and the green curve G_{d,t} is rounded to the nearest milliwatt. Carbon per (site, step) is (f_b · brown_{d,t} + f_g · green_{d,t}) with brown, green in mW over a one-second step; the objective coefficient is scaled by 10^13 and rounded to an integer: c_b = round(f_b / 3.6 · 10^13 / 10^9) = round(1388.9) = 1389 for f_b = 0.5 kg/kWh (per mW·s), c_g = round(27.8) = 28 for f_g = 0.01 (per mW·s; f_g coefficient relative rounding 0.8 %, applied only to the green part, which is at most a few per cent of the brown term at these factors). Quantisation bound per window, computable before any solve:

    Δ ≤ Σ_{d,t} [ 0.5 mW × (c_b + c_g) + |round(c_b) − c_b| × brown_max + |round(c_g) − c_g| × green_max ] / 10^13 kg,

with brown_max, green_max ≤ the site's capacity draw (640 PEs × 2.02 W + 10 W) in mW. On the HZ cell (5 sites, ≤ 600 steps, f_b = 0.5, f_g = 0.01) this evaluates to below 2 × 10^−6 kg, against 0.1 % × C_brown_ref = 1.9 × 10^−5 kg: the requirement Δ ≤ 0.1 % × C_brown_ref holds with a factor of about ten to spare. The bound is printed by the implementation and stored with each solve; a window whose bound exceeds the requirement is INVALID_QUANTISATION.

### B4. Locked implementation facts

- The 65.64 W draw of one 32-PE job (utilisation counted in MIPS) is the simulator's; the source comment quoting 81.8 W in `HostProfile.SPEC_ASUS_RS500A_DYN` is corrected during implementation (comment only, no behaviour change).
- The offline ladder is seven rungs × six windows = 42 solves and 42 simulator replays; the TimeCAP causal anchor is reported separately and is not counted in the ladder.
- 600 s solver limit, 3 % closure tolerance, the every-step offset executor as the settlement path, the TimeCAP / offline-curve separation, and the certification-only status of the offline solver: as in Addendum A.
- No carbon run before the freeze.

---

## Addendum C (2026-09-06 02:30; corrects the quantisation arithmetic of B3 found by the Codex mechanical check; no experimental design changes; the document is frozen at the commit that carries it)

### C1. B3's bound was wrong

With c_b = 1389 and c_g = 28 the coefficient rounding alone (0.11 and 0.22 per mW·s against a site draw of up to 1.30 × 10^6 mW over 600 steps on 5 sites) bounds the objective error at about 6.8 × 10^−5 kg (tight) or 1.0 × 10^−4 kg by B3's own formula, against the registered threshold 0.1 % × C_brown_ref = 1.897 × 10^−5 kg. The "below 2 × 10^−6 kg" statement of B3 was an arithmetic error and is withdrawn.

### C2. Exact integer objective

Because f_brown / f_green = 0.5 / 0.01 = 50 exactly, the objective is the exact integer

    J_int = Σ_{d,t} ( 50 × brown_mW[d,t] + 1 × green_mW[d,t] ),      C_kg = J_int / 3.6 × 10^11,

with no coefficient rounding at all. A preflight assertion checks that every datacentre of the block carries brown_carbon_factor = 0.5 and green_carbon_factor = 0.01 and stops the run otherwise (a different pair of factors needs its own exact integer ratio and an addendum).

### C3. The only remaining quantisation, with its bound

Rounding the green curve to the nearest milliwatt is the only quantisation left. Worst case per window:

    Δ ≤ 5 sites × 600 steps × 0.5 mW × (0.5 − 0.01) kg/kWh / 3.6 × 10^9 = 2.04 × 10^−7 kg,

about 93 times below the 1.897 × 10^−5 kg threshold. The implementation prints this bound with the actual number of steps per window and stores it with each solve; INVALID_QUANTISATION applies as in B3.

### C4. The three reference numbers, stated once

C_brown_ref = 0.01897 kg (the trace's dynamic energy 37.94 Wh at 0.5 kg/kWh); 0.1 % × C_brown_ref = 1.897 × 10^−5 kg (the quantisation threshold); 5 % × C_brown_ref = 9.49 × 10^−4 kg (the absolute headroom gate). §4 gate L1 refers to the last of these; its wording "the frozen reference of the scene design, 9.49e−4 kg" is corrected to "the absolute headroom gate 0.05 · C_brown_ref = 9.49e−4 kg".

### C5. Per-rung closure, per-job ledger conditions added

B2's closure requires, in addition to the 3 % carbon agreement, start alignment and contract, that for every job of every rung's replay: the datacentre it ran on equals the planned datacentre; it was dispatched, started and finished exactly once; and the ledger counters wrong_dc, unplanned_start, dispatched_never_started, running_pes_over_cap and deadline_forced are all zero. Any non-zero value is STOP_PLANNER_CLOSURE_RUNG.

### C6. Unchanged

All seven rungs OPTIMAL; per-rung replay closure; TimeCAP causal anchor separate from the offline ladder; 42 solves and 42 replays; 600 s; 3 %; every-step offset settlement; certification-only status of the solver; no carbon run before the freeze.
