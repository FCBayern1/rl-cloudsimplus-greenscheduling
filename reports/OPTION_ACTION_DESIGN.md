# Option action design and four-gate preregistration (v1, 2026-09-05)

Status: design and preregistration only. Nothing in this document is implemented. Once committed the criteria in §6 are frozen; later changes go into append-only addenda at the end of this file, never into the frozen sections. The fallback design in §8 is frozen at the same time and cannot be edited after any option result is read.

Basis: STAGE_D_PRIME_DESIGN.md §3 (E1 decomposition: the HZ ×2 value is temporal), §27 (development smoke A: step-wise DEFER failed the timing-selectivity gate, lift −0.022, balanced AUC 0.308), §28 (Codex ruling of 2026-09-05: option is a candidate action, not an approval to train; the four gates must be written and committed before implementation).

## 1. What the smoke showed and what an option is meant to change

Under step-wise DEFER a policy expresses "wait for green" as a chain of per-step decisions, one per sighting, on a job that re-enters the batch every step. Smoke A showed the D′ policy waits about 60 % of the time regardless of state and slightly more the longer a job has been present. The per-step credit for a wait is the defer charge of that step; the carbon consequence lands hundreds of steps later on a different action (the eventual route). The option moves the whole wait under one decision: the policy chooses once, an executor with a fixed causal stopping rule carries the wait, and the entire consequence is booked to the decision that created it.

The ruling's central risk is stated up front. The stopping rule of HOLD_FOR_GREEN is itself a sensible causal policy (the E1 blind arm `reactive_wait` is exactly "wait until the meter carries the job"). If a policy with no forecast access gets close to the oracle by choosing HOLD everywhere, the gain belongs to the executor, not to prediction, and the design is rejected by gate 2 (§6) in favour of §8.

## 2. Action primitive

Global action per batch slot j that holds a real cloudlet:

| index a | meaning |
|---|---|
| 0 ≤ a < n | ROUTE_NOW(d = a): route to DC d now, exactly today's route |
| n ≤ a < 2n | HOLD_FOR_GREEN(d = a − n): commit the job to DC d and hand it to the executor |

Action space MultiDiscrete([2n] × NB), config key `global_action_mode: option_v1` (default unchanged: the current `dc + bare DEFER` space, every frozen run byte-identical). The bare DEFER index disappears under option_v1. Padding slots (pes ≤ 0) accept any index and are ignored, as today.

### 2.1 HOLD_FOR_GREEN(d) semantics

Creation (step t_c): the cloudlet leaves the global routing queue into a hold ledger keyed by cloudlet id with (d, t_c, latest_start). From this moment it is never presented in a batch again and no policy, RL or heuristic, sees it as a slot. It stays visible only in aggregate through the existing v31 global-deferred count and MI (the ledger contributes to `getGlobalDeferredCount` / `getGlobalDeferredMi`).

Termination (checked at every step boundary before the batch is composed; held jobs are processed in ascending latest_start, tightest first, and the residual is re-evaluated after each release so one step cannot oversubscribe a site's green):

- T1 green. residual_green(d, t) ≥ draw(j), where residual_green(d, t) = max(0, G_d(t) − P_static,d − P_dyn,d(t)); G_d(t) is the realised green power at d at the current clock; P_static,d the configured static draw (0 on the zero-floor scene, PLANNER_STATIC_TOTAL_W otherwise); P_dyn,d(t) the dynamic draw of PEs busy at d plus the draw of cloudlets already routed to d and not yet started; draw(j) = pes_j × dyn_per_pe × u. This is the planner's `_reactive_choice` formula restricted to the committed site, computed from the simulator's own state. A Java pure function and the Python planner formula are tested for equality on a shared fixture.
- T2 margin. The deadline-safe rule of the DEFER mask: the job starts at the last step at which `defer_allowed_from(...)` (margin 2 steps, frozen in STAGE_D_PRIME_DESIGN §13) still permits a start, whether or not T1 holds. The Java latest-start backstop stays as the last net and must never fire (forced = 0 is a gate).
- No other exit. A held job cannot be recalled, re-targeted, or re-decided.

On termination the executor routes the cloudlet to d through the same route call as ROUTE_NOW and records (id, d, t_c, latest_start, t_s, reason ∈ {green, margin}, k = t_s − t_c).

### 2.2 What the executor may and may not read

It reads the current clock, the realised green power at d, the occupancy at d, and the job's own deadline arithmetic. It reads no forecast, no curve, no future truth, and it never compares candidate start times. It therefore cannot encode a best start; the only timing knowledge it carries is "the meter covers the job now" and "the deadline margin has arrived".

### 2.3 Shared executor

Every arm acts through the same action index and the same Java code path: the RL modules, the planner arms of §5, the adversarial always-hold arm, and the behaviour-cloned module of gate 4. No arm has a private executor.

### 2.4 Deadline mask under options

The existing observation key `batch_cloudlet_defer_allowed` keeps its rule and now governs HOLD: the score-based module masks the n HOLD columns of a slot whose value is 0 (the same −1e9 mechanism as §3 of the D′ design, with the audit switch retained). For arms that cannot be masked at the logit level, the env re-routes a disallowed HOLD(d) as ROUTE_NOW(d) with the same target DC, counts it in `ep_mask_route_count`, and records the id (per-id closure as today). Padding slots are excluded.

## 3. Reward and credit attribution

### 3.1 Reward stream (Java, per slot)

- ROUTE_NOW(d) at step t: unchanged per-action route reward at t.
- HOLD_FOR_GREEN(d) at creation step t_c: the first-defer base charge, exactly as the first explicit DEFER is charged today (`firstDeferBaseCharge`).
- Termination at t_s: r_term(j) = urgency settlement −w·[U(t_s) − U(t_c)] (the telescoping total is path-independent, so booking it once is identical to today's per-sighting sum) + the per-action route reward at d evaluated at t_s (the same function ROUTE_NOW would call at that moment).

A held job therefore pays what a step-wise defer chain pays today and earns what its eventual route earns; only the number of decisions changes.

### 3.2 Relocation to the initial decision

Java exports per step the list of terminations (cloudlet id, slot at creation, t_c, k, r_term). A training-side connector (`option_reward_relocation`, inserted before the advantage estimation) rewrites the batch: reward at t_c += γ^k · r_term, reward at t_s −= r_term, with γ the PPO discount (0.99, frozen). Because γ^{t_c} · γ^k = γ^{t_s}, the discounted return from the episode start is unchanged to floating-point precision; value targets and the P0′ metric `global_reward_discounted_sum` are invariant. What changes is the step, and hence the action, that carries the credit in the advantage. The relocated reward also joins the creation slot's per-slot channel, so EU-CRD's counterfactual ΔQ, Δr and responsibility are computed on the transition that made the decision. Nothing else in EU-CRD changes; the guard stays at η = 0.5.

Options still open at the drain boundary terminate by the shared drain (as reservations do today) and are relocated normally. An option cut by an episode truncation is booked at the truncation step with k to that step and counted in `ep_opt_truncated`, which must be 0 on every judgement window (windows end after the drain).

### 3.3 Ledger and closure fields

Per episode: `ep_opt_created`, `ep_opt_term_green`, `ep_opt_term_margin`, `ep_opt_truncated`, `ep_opt_dc_mismatch` (start DC ≠ creation DC, must be 0), `ep_opt_double_start` (must be 0), `ep_opt_stale` (ids still held after the drain, must be 0), `ep_opt_ledger_sha` (sha256 of the sorted rows), plus the existing `ep_mask_route_count`, `ep_mask_routed_ids_sha`, forced count, completion and on-time. Evaluation writes the full ledger as a CSV next to the decision dump.

## 4. Observation

Unchanged from D′: obs_v31 timing features on, obs_v32 off, the forecast channel as configured per arm (godeye truth, TimeCAP, perturbed, or none). Held jobs appear only in the aggregate v31 backlog keys. No new observation key is added for options; adding one later would require an addendum.

## 5. Arms (all through the shared executor)

| arm | forecast access | decision at first sighting |
|---|---|---|
| oracle_opt | truth curve | run the godeye planner's `_plan(r, p, latest)`; planned start ≤ t + lag → ROUTE_NOW(d*), else HOLD(d*) with d* the planned site |
| shuffle_opt, anti_opt | perturbed curve | same rule as oracle_opt on the perturbed curve (negative controls) |
| persistence_opt | flat future at the current level | same rule as oracle_opt on the flat curve |
| climatology_opt | mean curve | same rule on the climatology curve |
| reactive_opt | none | if `_reactive_choice` finds a site whose residual green covers the job now → ROUTE_NOW(d), else HOLD(d) with d the site of lowest current cost that is feasible |
| nowait_opt | none | ROUTE_NOW(d) by the shared spatial cost model, never HOLD (the S line) |
| always_hold | none | HOLD(d) with d the greenest site now, on every slot the mask allows (adversarial contract arm) |
| reference B, ST | E1 arms | `reactive_wait` (blind, step-wise) and the reserving godeye planner (truth, step-wise), unchanged, as the yardstick of gate 1 |

blind* = the blind option arm (persistence_opt, climatology_opt, reactive_opt, nowait_opt) with the lowest summed carbon on the development windows. It is chosen once, on the development windows, before gate 2 is read.

## 6. The four gates: order, data, frozen criteria

Data: the six development windows of the D′ corpus (2021, k0–k5, `stage_a_out/dprime_corpus`), development only. Judgement windows are not touched by any probe. Contract on every arm and window: completion 1.0, on-time ≥ 0.995, forced 0. All probes are zero-training except gate 4's supervised fit. Order is fixed; a later gate is not read if an earlier one fails.

### Gate 3 first: execution closure and contract (one window smoke, then re-checked on every rollout of gates 1–2)

Run all arms of §5 on window k0. Pass iff, per id: created at most once, started exactly once, start DC equals creation DC, no double start, no stale hold after the drain, no route rejected for capacity, forced 0; and per episode: completion 1.0, on-time ≥ 0.995 for every arm including always_hold; relocation closure: the discounted return of the raw and relocated reward streams agree within 1e−6 and per-id relocated mass equals per-id termination reward; the ledger sha reproduces from the CSV. A failure here is an instrument failure: it is repaired once, disclosed as a development-phase instrument repair, and the smoke is rerun; a second failure stops the line.

### Gate 1: expressibility

capture = (C_B − C_oracle_opt) / (C_B − C_ST), with C the summed clean carbon over the six windows, B = `reactive_wait`, ST = the reserving godeye planner (E1 numbers: C_B 4.110e−11, C_ST 2.494e−11 on the E1 grid, re-run under the same config for this gate). Pass iff capture ≥ 0.80 on the sum and ≥ 0.70 on at least five of six windows individually. Fail → §8.

### Gate 2: predictive necessity

Pass iff all three hold, each on the six windows paired by window:
1. C_oracle_opt ≤ 0.95 × C_blind* on the sum, and oracle_opt below blind* on at least five of six windows;
2. C_oracle_opt < C_shuffle_opt and C_oracle_opt < C_anti_opt on the sum and on at least five of six windows;
3. neither negative control beats the blind: min(C_shuffle_opt, C_anti_opt) > 0.95 × C_blind*. If a wrong forecast beats the best blind by the gate-2 margin, the gain is the executor's and the gate fails regardless of 1 and 2.

Fail → §8.

### Gate 4: small-sample learnability

Corpus: oracle_opt's decisions on windows k0–k3 (train) and k4–k5 (held out): one sample per job, at its first sighting, with the observation the D′ module sees (obs_v31 on, forecast channel as in the RL config); label = HOLD vs ROUTE_NOW and the target DC. Minimum: at least 60 held-out jobs with at least 15 of each class, otherwise the gate is INVALID (not failed) and the train/held-out pools are extended with further development windows before any reading.

Fit: the D′ score-based global module architecture trained by cross-entropy on the labelled slots, seed 20260905, epochs fixed at 200, no early stopping on held-out data, no hyper-parameter search.

Pass iff both hold on the held-out windows:
1. classification: HOLD-vs-ROUTE lift ≥ 0.10 and balanced AUC ≥ 0.60 (raw, pre-mask, recurrent in time order, one decision per job);
2. executed value: the fitted module acting through the shared executor on k4–k5 captures at least half of the oracle's gain over the blind, (C_blind* − C_BC) / (C_blind* − C_oracle_opt) ≥ 0.50, with the contract clean. DC agreement with oracle_opt is reported descriptively.

Fail → STOP, report; no RL run follows from a learnability failure and no preregistered RL route exists for that case.

### Consequence table

| outcome | consequence |
|---|---|
| gate 3 fails twice | line stops |
| gate 1 or gate 2 fails | the option design is rejected; the frozen fallback of §8 is started as written; the option is not modified |
| gate 4 fails | stop; report to the user and Codex; no RL |
| all four pass | write the RL preregistration (seeds, windows, re-certification of §9) as a separate document; still no training until that document is ruled |

## 7. Frozen numbers

γ 0.99; mask margin 2 steps; static draw as configured (0 on the zero-floor scene); capture 0.80 (sum) and 0.70 (five of six); necessity 5 % and the negative-control rule; closure zeros and 1e−6; lift 0.10, balanced AUC 0.60; executed capture 0.50; minimum 60 held-out jobs with 15 per class; BC seed 20260905, 200 epochs; window set k0–k5 with k0–k3 / k4–k5 split.

## 8. Preregistered fallback: discrete (DC, start-offset)

Started only if gate 1 or gate 2 fails; frozen now.

Action per slot: (d, κ) with κ ∈ {0, 1, 2, 4, 8, 16, 32, 64} steps (MultiDiscrete([n × 8] × NB)); the job starts at d at t + κ_eff with κ_eff = min(κ, latest_start − t). The executor is a reservation ledger with a fixed start: it reads no green at all, so the ruling's risk does not arise; the relocation of §3.2 applies unchanged with k = κ_eff. Deadline mask: offsets beyond latest_start − t are masked; an unmaskable arm's illegal offset is clipped to κ_eff.

Arms: oracle_off (the godeye planner's planned start quantised down to the grid), shuffle_off, anti_off, persistence_off, climatology_off, nowait_off (κ = 0), always_max (κ = 64, adversarial). blind* is chosen as in §5.

Gates: the same four, in the same order, with the same numbers. Gate 1's yardstick stays B and ST. If the fallback also fails gate 1 or 2, the action-space line stops and the report says so.

## 9. Parked by the ruling

The 2020 judgement set, the never-used turbine set and `calibrated_shrink_hz_v2` stay parked (STAGE_D_PRIME_DESIGN §24–§25). Any RL preregistration after the four gates has to re-certify the scene and re-calibrate the error model before it can be ruled; nothing is inherited.

## 10. Implementation order (after this document is committed and ruled; not started)

1. Java: hold ledger, termination rule as a pure function next to `PerActionRewardMath`, tie order, ledger export, closure counters; unit tests including the Java–Python residual-green equality fixture.
2. Env: `global_action_mode: option_v1`, HOLD-column mask, env-side re-route of disallowed HOLD, info passthrough of terminations and ledger fields; tests.
3. Learner: `option_reward_relocation` connector with a closure test (discounted return invariant, per-id mass).
4. Evaluation: the arms of §5, the ledger CSV, gate-3 checks in the verdict script; tests on synthetic ledgers.
5. Gate 3 smoke → gates 1 and 2 → gate 4 (BC trainer, held-out scorer, executed arm) → one verdict script with pure judge functions and tests for every gate.

Estimated cost: gates 1–3 about ten arms × six windows of zero-training rollouts, roughly one hour locally; gate 4 a supervised fit of minutes plus two executed windows.

## 11. Known hazards recorded now

- The residual-green formula exists twice (Java executor, Python planner); the equality fixture is the only thing keeping them one rule.
- Runtime and draw must use the cpu-utilisation-aware formulas of the backstop and the mask (the u = 0.5 stretch found in the SQT2 audit), never the nominal length/mips.
- Held jobs are invisible per job; if the RL policy needs the held backlog per site, that is an observation change and an addendum.
- The margin was frozen from a probe at development load where route→start delay never exceeded 1 s; gate 3's contract check on always_hold is what catches a larger delay under a full hold backlog.

---

## Addendum A (2026-09-05, after the Codex review of v1; no option result has been read)

Sections above stay as written; where this addendum conflicts with them, the addendum governs.

### A1. §3.2 suspended; the four gates read raw rewards

The claim in §3.2 that value targets are invariant under relocation is wrong for the states strictly between creation and termination (t_c < t ≤ t_s): their return-to-go included r_term before relocation and does not after it, so their value targets and the GAE change. Only the discounted return from the episode start and from states at or before t_c is preserved. The relocation connector is therefore not part of the four-gate phase; gates 1–4 use analytic arms and a supervised fit and need no PPO credit assignment. Gate 3 reads the raw per-step reward ledger, the option lifecycle and the physical contract; its "relocation closure" clause is withdrawn. The SMDP / credit-attribution design (relocation, or an option-level critic) is written as a separate RL preregistration only after all four gates pass, and is ruled before any training.

### A2. Discount and λ corrected; a P0′ instrument discrepancy disclosed

The global policy trains with γ = 0.999 and GAE λ = 0.98 (`config_stage_d_dprime.yml`, `global_model` block); the local policy with 0.99 / 0.95. Every "γ 0.99" in §3 and §7 refers to the global policy and is corrected to 0.999. Disclosure found while checking this: `evaluate.py` computes `global_reward_discounted_sum` with the top-level `gamma` key (0.99), not the global policy's 0.999, so the formal P0′ (STAGE_D_PRIME_DESIGN §21) was judged at 0.99. The reader is corrected to the `global_model` value with a test, and P0′ is re-read at 0.999 as a development-phase instrument repair, recorded in STAGE_D_PRIME_DESIGN with both readings; the 0.99 rows stay archived.

### A3. Capacity commitment of HOLD_FOR_GREEN

1. Fallback reservation at creation. HOLD(d) books, on d's reservation grid (active executions plus held fallback reservations, the planner's `occ` semantics kept in Java), the latest feasible start s_f ≤ latest_start_j for (pes_j, runtime_j). If no such start exists, HOLD(d) is illegal for that slot. Legality is exposed as a new observation key `batch_cloudlet_hold_allowed` of shape (NB, n): 1 iff the deadline rule of §2.4 allows waiting and a fallback reservation exists at d against the ledger at observation time. The module masks the HOLD(d) column with it (the per-slot deadline key stays for compatibility). Creations within one step are applied in slot order against the ledger; a HOLD whose reservation no longer fits when its turn comes is converted to ROUTE_NOW(d) and counted in `ep_opt_hold_refused`, which must be 0 for every analytic arm in gate 3 and is reported for the behaviour-cloned arm.
2. T1 needs capacity as well as green: residual_green(d, t) ≥ draw(j) and free PEs at d ≥ pes_j, both evaluated after the releases already made earlier in the same step (local step accumulators for released draw and PEs; the simulator's next-step state is not waited for).
3. T2 is the fallback reservation: the job starts at s_f, which is ≤ latest_start_j by construction. A T1 release frees its reservation.
4. Timing truth. The ledger records both t_route (the executor's route call) and t_s, the simulator's execution-start event of the cloudlet (the start-time map already used by the margin probe). k, every timing statistic and the option's route→start delay use t_s; the per-option delay is a gate-3 reading and its maximum must stay within the frozen margin (≤ 1 step).
5. Observation additions (§4 amended; these are the only new keys): per DC held count, held PEs and the tightest held margin (min over held jobs at d of latest_start − t, scaled by the v31 deadline scale), all (n,), plus the (NB, n) hold mask above. The ledger that shapes legal actions is thereby observable; nothing else about the held jobs is exposed.

### A4. Gate 4 executed capture

The threshold (C_blind* − C_BC) / (C_blind* − C_oracle_opt) ≥ 0.50 is approved and frozen.

### A5. Fallback offset grid (§8 amended)

The set {0, 1, 2, 4, 8, 16, 32, 64} is withdrawn: the HZ scene's wait cap is W = 72 steps and the grid must reach it. Rule without a free parameter: K(W) = {0} ∪ {2^q : 2^q < W} ∪ {W}; for W = 72 that is {0, 1, 2, 4, 8, 16, 32, 64, 72}. The oracle quantises its planned start down to the largest legal offset not later than it. An offset is legal iff t + κ ≤ latest_start_j and a reservation for (pes_j, runtime_j) fits at (d, t + κ). Illegal offsets are masked for the RL module through a (NB, n × |K|) key; nothing is clipped, so no two actions alias. Analytic arms choose only legal offsets because they read the ledger; an illegal choice from any arm is counted in `ep_off_illegal`, which gate 3 requires to be 0.

### A6. Gate 1 denominator validity

capture is computed only when C_B − C_ST > ε with ε = 0.10 × C_B (E1's pooled gap was 39 %). If the pooled denominator fails this, gate 1 is INVALID_DENOMINATOR: stop and report, no ratio. A window whose own denominator fails is dropped from the individual count, which then requires 0.70 on all but one of the remaining windows; fewer than four valid windows is INVALID_DENOMINATOR as well. ε = 0.10 is a proposal beyond the ruling and is flagged.

### A7. Gate 4 corpus insufficiency

If the frozen corpus (k0–k3 train, k4–k5 held out) fails the minimum of §6 gate 4, the gate is INVALID_CORPUS and the line stops; the corpus is not extended on the spot. An extension requires its own window-selection rule with a hash, written as an addendum and frozen before anything is read.

### A8. What may be implemented now

(see also Addendum B below on where the executor's parts live)

After this addendum is committed: the Java hold ledger with fallback reservations, the option and offset action modes in the env with their masks, the analytic arms of §5 and §8, and the four gates with their verdict script and tests. Not now: the learner connector of §3.2, any RL training, any change to the 2020 / turbine / `calibrated_shrink_hz_v2` items.

---

## Addendum B (2026-09-05, before implementation; where the executor's parts live)

A3.1 says the reservation grid is "kept in Java". Checked against the code before implementing: the simulator does not expose future occupancy (remaining runtimes of executing cloudlets, local waiting queues) through any existing interface, while the Python planner already carries a tested occupancy grid (`occ`, `_hold`, `_release`, `_feasible_all`) built from committed dispatches and the same runtime formula the backstop and the mask use. Duplicating that grid in Java would create a second model of the same thing with its own drift. The split is therefore:

- Java (truth and primitives): `holdCloudlet` removes a cloudlet from the routing queue into a held map that is never batched again and counts in the deferred aggregates; `releaseHeld(id, dc)` routes it through the ordinary route call at the current clock, books the urgency settlement and the per-action route reward exactly as a route would, and returns the reward; the creation charge is the first-defer base charge on the hold step; execution-start times per held id come from the simulator's own start events (the map the margin probe already builds). Gate 3's timing truth (t_s, route→start delay, forced count) is read from here only.
- Python env layer (one executor for every arm): the option ledger, the reservation grid with the planner's semantics and constants (cap from the VM configuration, runtime = mi / (mips · u), per-PE dynamic draw, static draw from `PLANNER_STATIC_TOTAL_W`), the fallback reservation at creation, the T1 / T2 rule with same-step accumulators, the hold mask (NB, n), the per-DC held observation keys, and the env-side re-route of illegal HOLDs. Every arm, RL or analytic, acts through `env.step`, so the executor is shared by construction.

The grid is a model of the simulator, as it is for the planner. What keeps it honest is gate 3: per-option route→start delay from the simulator's start events, forced = 0, `ep_opt_hold_refused` = 0 on analytic arms, completion and on-time on every arm including always_hold. The residual-green formula exists once, in the Python executor, and the planner's `_reactive_choice` is tested for equality against it.

Commit status, stated as it was: this addendum was written before the executor was implemented but was not committed until Addendum C (the design-log entry §29 said "appended" without the `git add`); the option gates were therefore judged with Addendum B in the working tree only. Its content did not change between writing and commit.

---

## Addendum C (2026-09-05, Codex ruling on the option result; before any fallback code)

The option result stands as archived (STAGE_D_PRIME_DESIGN §32): gate 3 PASS, gate 1 capture 0.313 pooled, no window ≥ 0.70. It is not revisited, not modified, not rerun. The preregistered fallback of §8 / A5 starts under the following amendments, all frozen before implementation.

### C1. Executor reuse, with its own tests

The fallback executor reuses the Java hold / release primitives and the Python reservation grid. Three independent tests are required before any fallback row is produced: (i) termination is a function of t_creation + κ only (green, occupancy and deadline enter legality at creation, never the release); (ii) given a fixed action sequence, replacing the entire green curve leaves every release step and site bitwise unchanged (an executor property, tested on the executor with scripted actions; the arms' choices may and should change with the curve); (iii) every arm, analytic or fitted, reaches the executor through the one env translation.

### C2. κ is a dispatch offset

κ is the interval from creation to the route call, not to execution start. The ledger records the route step and the simulator's execution start; gate 3 keeps route→start ≤ 1 step. The action is named (DC, dispatch-offset) in code and reports. Legality of (d, κ) at step t: the start t + κ + lag is not later than the job's latest start, and a reservation for (pes, runtime) fits at d from that start on the current grid.

### C3. The no-forecast family and the order of reading

Frozen blind arms, each through the same rule "κ illegal → the analytic arm itself takes the largest legal offset not above it; the executor never clips":
- fixed_off(κ) for every κ ∈ K(72) = {0, 1, 2, 4, 8, 16, 32, 64, 72}, nine arms; site by the current visible cost (the persistence view: the meter as it reads now, priced at the chosen start on the grid), feasibility respected; fixed_off(0) is the no-wait arm and fixed_off(72) the latest-legal-offset arm, so neither gets a duplicate;
- reactive_off: dispatch now at the site whose meter covers the job, else the largest legal offset with the site by current visible cost;
- persistence_off and climatology_off: the reserving planner with the flat and the mean curve, its planned start quantised down to the largest legal offset.

Procedure: the blind arms run on the six development windows first; blind* is frozen as the arm with the lowest pooled carbon, written to `stage_a_out/offset_blind_star.json` with the rows' hashes; only then are oracle_off, shuffle_off and anti_off produced and gates 1 and 2 read. Gate 3 is checked on every row of every arm, blind or not.

### C4. Gate 4 in offset semantics

p_delay = Σ_{d, κ>0} P(d, κ); label = [κ_oracle > 0]; lift ≥ 0.10 and balanced AUC ≥ 0.60 on this dispatch-now / delay split; executed capture ≥ 0.50 unchanged. Exact-action accuracy, site accuracy and offset MAE are supporting readings only. Fit hyper-parameters frozen now: Adam, learning rate 1e-3, one optimiser step per training window (the whole window's labelled slots, recurrent state carried, detached between steps), gradient-norm clip 1.0, no class weighting, the module's default initialisation with the block's `score_encoder_init_gain`, seed 20260905, 200 epochs, no early stopping, argmax decode for the executed arm.

### C5. Procedural disclosures

- Addendum B was written before implementation but committed only with this addendum (see the note under Addendum B).
- A6's denominator validity (ε = 0.10 · C_B) was a proposal that the option gate-1 judge already applied; Codex ratifies it for the fallback. The option gate-1 conclusion does not depend on it (pooled 0.313; every valid window far below 0.70).
- shrink_opt (the sister-turbine-calibrated arm) was never registered in §5; its rows are descriptive and enter no gate, option or fallback.

### C6. Order and consequence

Gate 3 smoke (k0, blind arms and fixed_off(72)) → six-window blind rows → blind* frozen → oracle / shuffle / anti rows → gate 3 on all rows → gate 1 → gate 2 → gate 4. A fallback failure at gate 1 or gate 2 ends the action-space direction; no third action is added.

---

## Addendum D (2026-09-05 22:0x, written while the fallback chain runs and before any gate-4 number is read)

### D1. What gate 4 can and cannot say

The RL observation carries, per site, four forecast summaries (short mean, short trend, long mean, peak timing; `forecast_mode: full`, obs_v32 off). The oracle arms read the raw curve. A gate-4 failure therefore admits two readings that the preregistration did not separate: the four summaries do not carry what a (DC, dispatch-offset) choice needs (which offset covers the whole runtime with green, whether a better window follows), or 140-odd samples over 45 classes are too few for this architecture whatever the observation. Gate 4's verdict text is fixed now: a failure is reported as "the current forecast representation does not support this action under the frozen fit", never as "the forecast carries no information" (the raw-curve value is established: STAGE_D_PRIME_DESIGN §3, the 2-HZ record, and a fallback gate 2 pass if it comes).

### D2. Representation diagnostic, preregistered, read only after gate 4

One additional supervised fit, same corpus, same split, same hyper-parameters, same seed and architecture as gate 4, with one change: the observation gains, per (slot, site, κ), the oracle-representation feature future_covered_energy = the green energy the job would draw at site d starting at t + κ + lag over its runtime under the arm's curve view, normalised by the job's total draw (a (NB, n·|K|) key; zero on padding). It is a diagnostic of the interface, not a gate: it changes no threshold and never enters the RL preregistration by itself.

Reading rule: gate 4 fails and the diagnostic passes → the representation is the bottleneck, and the next design item is an observation change (a candidate-offset feature of this kind) written as its own addendum before any RL. Both fail → sample size or architecture, and the report says the learnability question is open. Gate 4 passes → the diagnostic is reported descriptively only.

### D3. Interpretation of the fallback gates, fixed before reading

- oracle_off well below every fixed_off / reactive blind, shuffle and anti worse than blind*: the timing in the forecast is worth something beyond the licence to reserve.
- some fixed_off(κ) close to oracle_off: the gain is reservation, spreading and de-peaking, not per-job prediction; the line ends at gate 2 as ruled.
- oracle_off expresses ST (gate 1) but the fit cannot learn it (gate 4): see D1/D2.
