# Stage D′ design basis — Codex ruling of 2026-09-05 on the root-cause analysis, with verified implementation notes

Status: DESIGN APPROVED, retraining NOT approved as proposed. This file records the ruling verbatim in substance, the corrections it makes to `CODEX_PROMPT_2026_09_05_ROOT_CAUSE_AND_PLAN.md`, and what the codebase actually offers for each item (checked this session). The Stage D′ preregistration is written only after steps 1–5 of §4 complete.

## 1. Root causes as ruled

- **R1 (accepted, reworded):** the policy lacks an explicit, per-job, approximately Markov-sufficient timing state. Not "mathematically unobservable": a GTrXL can partially infer timing from repeated observations. Evidence E2 stands (18 obs keys, no deadline / wait age / deferred state; forecast = 3-row and 144-row scalars).
- **R2 (corrected):** the SLA is not absent from training. The Lagrangian channel is on with `sla_mode: deadline_miss` and `sla_deadline_miss_target: 0.1`, so a 6.5% miss rate (E's mean on-time 0.935) is *feasible* for training while the contract allows about 0.5%. This is a contract–objective misalignment, not a missing term. Verified: `SimulationSettings.java:569–571`, `lagrangian_callback.py` dual update, frozen E block `sla_target: 0.62` (the old completion target).
- **R3 (partly accepted):** zero slack is unsafe once start latency and queueing exist, but enlarging the Java backstop slack fires `forced` earlier, and the contract requires `forced = 0`; it cannot be the main fix.
- **R4 (stays a hypothesis):** the existing `normalize_rho_cap` bounds only the upward amplification, not the recovery-phase suppression (w ≈ 0.06); a one-sided cap may not close the described ratchet.
- **E1 wording:** "in the pooled decomposition, no positive gain was observed once active deferral was forbidden"; spatial and temporal effects may interact and per-window shares are not uniformly negative (−0.50 / +0.60 / −0.53). Not "100% temporal".

## 2. The five rulings

| # | ruling | what the code offers (verified) |
|---|---|---|
| Q1 | `obs_v31_features: true`, `obs_v32_job_forecast: false`. All four lines get raw per-job timing state (deadline, wait age, deferred flag, global deferred backlog); no best-start hint. | Flag exists; V3.1 campaign used it in 26 blocks. Keys added: `batch_cloudlet_time_to_deadline`, `deadline_present`, `wait_age`, `is_deferred`, `defer_count`, `global_deferred_count/mi`. |
| Q2 | Timing enters via the **existing Lagrangian SLA channel**, not `global_reward_gamma` (that is the green-waste weight): `sla_mode: ontime_mi`, `sla_target: 0.995`. No fixed defer cost: it would systematically oppose the waiting that is valuable here. | `MultiDatacenterSimulationCore.java:2805–2808`: `ontime_mi` mode computes c_ep = max(0, sla_target − ontime_mi_share); pure config change. Note the frozen `sla_target` is 0.62 and must be set to 0.995. |
| Q3 | Do **not** freeze `normalize_rho_cap = 1.5` now. Run M5 first, separately for DEFER and ROUTE: raw ρ, normalised w, upper-tail amplification, lower-tail suppression, advantage sign and magnitude, ΔQ, Δr, c_t, τ. If only upper-tail amplification appears, 1.5 is a candidate frozen after development; if "amplified in, erased out" appears, a one-sided cap is insufficient and a symmetric guard or shrinkage toward w = 1 is needed. Design fixed before the new preregistration. | Cap exists (`normalize_rho_cap`, default inf); a symmetric guard would be new code in `_compute_responsibilities`. |
| Q4 | Reject the "[0.3×, 3×] of ST's defer rate" gate. Replace with a **timing-selectivity gate** on a frozen diagnostic corpus containing non-empty ST-route and ST-defer samples: P_V(defer | ST-defer) − P_V(defer | ST-route) ≥ 0.10, balanced AUC ≥ 0.60, clean contract all green, and no collapse of the overall defer rate. | Needs a corpus builder: run ST on the corpus windows, record its per-slot decisions, replay the same states through V and score V's DEFER probability. New diagnostic code. |
| Q5 | Do **not** withdraw the reservation "not yet proven structurally unlearnable". E1 shows the value comes from timing and that V did not realise it; the action space can still express waiting through repeated DEFER. Stage D stopped on E's contract failure; gate 2 is a strong single-seed negative indication only. | Recorded in Addendum G wording. |

## 3. Two additions to the plan

- **Deadline-safe DEFER action mask, shared by every algorithm.** At a slot's last safe start point the DEFER choice is masked so the policy must pick a DC itself; the Java backstop remains only as a last safety net. An adversarial always-defer policy must be routed legally by the mask: on-time ≥ 0.995 and Java `forced = 0`. Verified state of the code: the global `action_mask` is **slot-level** (`get_global_action_mask`, shape (128,)), applied by the GTrXL module as slot validity; there is no per-choice mask. Implementation needs (a) Java: per-slot `defer_allowed` from the latest-start rule with a safety margin, exported in the global observation; (b) env: obs key + mask; (c) module: −∞ on the DEFER logit of slots with `defer_allowed = 0`. The backstop slack stays as the last net.
- **New seeds and new judgement windows.** The current changes were designed from seed 20260904 and windows already read. Old windows and seeds serve development and smoke only; Stage D′ freezes five new paired seeds and new unread judgement windows (window preflight rerun). Scene, corruptions and gate 1–5 formulas may stay. CCA-PG and the risk set must be retrained under the new observation, SLA and mask semantics if D′ passes; no old baseline result is reused.

## 4. Order of execution (ruled)

1. **M5** action-conditioned credit audit on the archived E / N_E checkpoints (§5 below).
2. **M1** `obs_v31_features` + contract-aligned SLA (`ontime_mi`, 0.995) + deadline-safe DEFER mask.
3. **P0′** extended reward truth table, comparing the actual PPO **discounted return**, not only undiscounted reward and terminal carbon, on behaviours differing only in timing: best-green-window-and-on-time > start-now-on-brown > defer-until-late.
4. One development smoke on the old windows and old seeds.
5. Timing-selectivity gate and contract gate pass → freeze the new Stage D′ preregistration (new seeds, new windows).
6. Only then the five new seeds; baselines after the main line passes.

## 5. M5 audit — definition (frozen before running)

Inputs: archived checkpoints `E_s20260904` and `NE_s20260904`, `checkpoint_init` and `checkpoint_000000 … 000009` (every 40k). For each: restore the algorithm, sample fresh episodes on the training scene with the checkpoint's own policy, build a local learner from the frozen config with the checkpoint's module weights, disable the reweighting warm-up for the audit, and run the EU-CRD term computation exactly as the learner does (`_compute_crd_terms`), capturing per transition: ρ_routing raw and after normalisation, w, advantage before and after reweighting, ΔQ, Δr, c_t, τ, and the DEFER share of the valid slots at that transition. Split transitions into DEFER-dominated (share ≥ 0.5) and ROUTE-dominated, report the distributions and the four quantities Q3 names: upper-tail amplification (w > 1), lower-tail suppression (w < 0.2), advantage sign by class, and whether the DEFER-class w differs from the ROUTE-class w in a consistent direction across checkpoints after warm-up (checkpoints ≥ 000005). The same on N_E as the control. Caveat recorded in advance: EMA scale state and the blender's τ are learner-side memory not stored in the checkpoint (to be verified in code); the audit re-warms them on a burn-in of five batches before recording and reports first-batch values separately.

## 6. Provenance

Root-cause prompt commit a3703679 (its footer originally mis-stated df2ce234; corrected). This design file supersedes M1–M7 of that prompt where they conflict.

## 7. Implementation notes verified after the ruling (2026-09-05)

- **Per-job timing state is already produced by Java on every step.** `GlobalObservationState` carries `batchCloudletWaitAge`, `TimeToDeadline`, `DeadlinePresent`, `IsDeferred`, `DeferCount`, `Ids`, `globalDeferredCount/Mi`; the Python env reads them (`hierarchical_multidc_env.py:2019–2026`) and exposes them only when `obs_v31_features` is true. M1's observation change is therefore a config flag, no Java change.
- **The score-based global module applies no action mask at all.** `GTrXLScoreBasedGlobalRLModule._forward_pass` concatenates five DC scores and one `defer_head` logit per slot (`:1405–1418`) and reshapes to the 768 action logits without any masking; the slot-level `action_mask` in the observation is consumed only by the older per-head modules. The deadline-safe DEFER mask is therefore: (a) env computes `batch_cloudlet_defer_allowed` (128,) = 1 iff `time_to_deadline − mi/(vm_pe_mips·u) − margin > 0` (the backstop's own runtime formula, `_v32_vm_mips` and `cloudlet_cpu_utilization` already in the env) for valid, deadline-carrying slots, else 0 (padding slots 0, no-deadline slots 1); (b) the module sets the defer logit to −1e9 where `defer_allowed = 0` before the reshape. Two small, testable changes; shared by every algorithm through the module base class.
- **Java backstop stays as the last net** with `defer_deadline_slack_sec` unchanged; the mask's margin is the parameter to freeze (proposal: one decision step plus the observed p95 queueing delay from the development smoke, recorded in the D′ preregistration).
- **Contract-aligned SLA is a config change**: `sla_mode: ontime_mi`, `sla_target: 0.995` (Java `:2805–2808` already implements the on-time cost; the frozen `sla_target` 0.62 is the old completion target and must be replaced).
- **Timing-selectivity corpus (Q4)** needs a per-decision dump that does not exist yet: the planner has no decision log and evaluate.py has no per-step action dump. A small addition to evaluate.py (per-step CSV of slot id, cloudlet id, time-to-deadline, action) serves both ST (corpus labels) and V (defer probability replay), gated by an env var so frozen runs are unaffected.
- **M5 audit** (running): the Ray restore path works end to end (algorithm, env runner, local learner, checkpoint module weights); learner-side EMA and warm-up state are in-memory only (`_crd_share_scale_ema`, `_crd_reweight_calls`), so the audit burns in and reports first-batch and warmed values separately, as §5 states.

## 8. Implementation status (2026-09-05, before any D′ run)

| item | state | where |
|---|---|---|
| M5 audit | script verified end to end on E ckpt 000009 (loss-mask and slot-mask aligned); full sweep E/N_E × init + 10 checkpoints, burn-in 5, running | `stage_d_credit_audit.py`, `stage_d_credit_audit_summary.py`, results `drl-manager/results/stage_d_credit_audit/` |
| M1 timing state | config flag `obs_v31_features: true` (Java already produces the arrays) | `config_stage_d_dprime.yml` |
| M1 deadline-safe DEFER mask | env computes `batch_cloudlet_defer_allowed` one step ahead of the backstop rule; score-based module masks the DEFER column; env re-routes a disallowed DEFER from a heuristic or adversarial arm to the greenest DC with room and counts it (`info["mask_route_count"]`); all off unless `defer_deadline_mask: true` | env, `rlmodule_gtrxl_models.py`, 11 tests |
| M1 contract-aligned SLA | `sla_mode: ontime_mi`, `sla_target: 0.995` in the overlay | `gen_stage_d.py DPRIME_OVERLAY` |
| D′ config | generated from the frozen HZ block + ledger-aligned reward + overlay; diff against the long-run config limited to the six overlay keys by test; SHA 003fd846… (margin still the placeholder 0.0) | `config_stage_d_dprime.yml`, `stage_d_manifest_dprime.json` |
| P0′ | `hz_p0` replays the extra `godeye_nodefer` arm under `P0_VARIANT=dprime`; `p0_verdict.py dprime` judges discounted-return order clean > nodefer > always_defer, clean < nodefer on carbon, always_defer routed legally (on-time ≥ 0.995, forced = 0), plus the legacy P0 gates; 5 tests | `run_stage_a.py`, `p0_verdict.py` |
| evaluate.py | `global_reward_discounted_sum` (γ from config, 0.99) in both loops; `EVAL_DECISION_DUMP` per-slot decision CSV (+ optional raw-obs npz) for the Q4 corpus | `evaluate.py`, 3 tests |
| Q4 timing-selectivity gate | corpus builder and replay scorer not yet written (needs the dump above) | — |
| D′ preregistration | not written; waits for M5, P0′, the development smoke and the margin | — |

Nothing above has been run on the scene except the M5 audit; no frozen run is affected (every change is behind a flag that the frozen configs do not set).

## 9. M5 result (2026-09-05 02:52; 22 checkpoints, E and N_E, burn-in 5 batches, 4000 steps per batch, fresh rollouts on the training scene)

Full tables: `drl-manager/results/stage_d_credit_audit/summary.json` (copied to `reports/manifests/stage_d/credit_audit/`). DEFER = timesteps where ≥ 50% of the valid slots chose DEFER; w = the multiplier the mean-preserving reweighting actually applied.

**E (forecast).** The drift into deferring is late: the DEFER share of timesteps is 0.3–2.3% up to checkpoint 6, then 11.3% (7), 15.5% (8), 41.1% (9), i.e. after the reweighting warm-up (450 calls ≈ checkpoint 4–5). On the three checkpoints where deferring grows:

| ckpt | defer % | w(DEFER) | w(ROUTE) | P(w<0.2) DEFER / ROUTE | P(w>1) DEFER / ROUTE | adv>0 DEFER / ROUTE | mean abs adv DEFER / ROUTE |
|---|---|---|---|---|---|---|---|
| 7 | 11.3 | 0.946 | 1.007 | 0.085 / 0.075 | 0.768 / 0.858 | 0.50 / 0.54 | 0.82 / 0.73 |
| 8 | 15.5 | 0.967 | 1.006 | 0.069 / 0.079 | 0.782 / 0.849 | 0.52 / 0.53 | 0.83 / 0.77 |
| 9 | 41.1 | 0.959 | 1.030 | 0.141 / 0.102 | 0.794 / 0.870 | 0.44 / 0.52 | 0.81 / 0.65 |

DEFER transitions receive **less** weight than ROUTE transitions (w difference −0.04 to −0.07, negative on every checkpoint from 7 on), are amplified less often, and at the last checkpoint are erased (w < 0.2) 14.1% of the time against 10.2% — while carrying a larger and more often negative advantage. That is the "erased on the way out" half of the ratchet: the corrective signal against deferring is the one being suppressed. The "amplified on the way in" half is **not observed** at any checkpoint: DEFER never gets a systematically larger w than ROUTE (checkpoints 0–6 have w differences of ±0.03 on DEFER samples too small to read).

**N_E (control, no forecast).** DEFER share stays 0.2–0.6% at every checkpoint; the DEFER class holds 40–120 transitions, so its statistics are noisy. The pattern above is absent: w(DEFER) − w(ROUTE) is mixed in sign (mean +0.02 after warm-up) and DEFER transitions are erased *less* often than ROUTE (0.00–0.09 vs 0.09–0.12).

**Reading for Q3.** What the audit confirms is one-sided but the opposite side from the existing guard: the defect is in the **lower tail** (ρ near its floor → w ≈ 0.06–0.2 on the transitions that should correct over-deferral), not in upper-tail amplification. A `normalize_rho_cap` alone therefore does not address the observed mechanism. The candidate guard for the D′ preregistration is a floor on w or shrinkage toward 1 (w′ = 1 + λ(w − 1), λ < 1) applied symmetrically, with the value fixed in development before freezing. What starts the drift at checkpoints 6→7 is not shown by this audit; the flat defer axis (R2) is sufficient to start it, and the reweighting then fails to stop it.

**Caveats, stated in advance and confirmed.** Fresh rollouts on the training scene (defer share 41% at checkpoint 9 here against 95% in the judgement evaluation, which used other cells, windows and windows' contract), learner-side EMA and τ re-warmed rather than restored, and DEFER samples too small before checkpoint 7 on E and everywhere on N_E to carry a sign.

## 10. Codex ruling on M5 (2026-09-05): symmetric guard adopted, one-sided cap rejected

**Wording of the M5 finding, as ruled:** M5 confirms that during the drift phase DEFER-dominated transitions receive lower responsibility weights and carry larger, more often negative advantages; no systematic amplification is observed in the entry phase. This supports the reading that corrective credit is suppressed, but because of policy-induced state-distribution shift and the re-warmed EMA/τ it is not causal proof that the reweighting caused the collapse.

**Guard (frozen now, no sweep, 0.25/0.75 not to be tried):** w′ = 1 + η (w − 1), η = 0.5, applied after the ρ/mean(ρ) normalisation and before the advantage multiply. Config key `crd.responsibility.responsibility_shrink_strength` (not "lambda", to avoid the SLA multiplier), default 1.0 so every historical configuration is bit-identical; η = 0 must reduce to un-reweighted PPO. Both `w_raw` and `w_guarded` are recorded. Properties: mean 1 preserved, responsibility ordering preserved, w = 0.06 → 0.53, 0.2 → 0.6, 2 → 1.5.

**Cross-statistics still owed before the guard is called a fix** (this needs per-transition records, which the archived JSONs do not hold; the drift checkpoints are re-sampled with raw records saved): E[w | DEFER, A<0] vs E[w | DEFER, A>0]; P(w<0.2 | DEFER, A<0 / A≥0); the share of DEFER negative-advantage mass retained after reweighting; the same recomputed counterfactually with η = 0.5. If the negative corrective mass clearly recovers, write "supports the ratchet reading"; otherwise the guard is reported as a general regulariser, not as a root-cause fix.

**Order:** cross-statistics → η guard with bit-identical regression tests → mask margin fixed mechanically as ceil(max route→exec-start delay) + 1 step from a saturated-dispatch probe on a fixed development load (never from carbon or training) → P0′ on the actual discounted return (best window on time > start now on brown > late; always-defer routed by the mask with on-time ≥ 0.995 and forced = 0) → Q4 frozen corpus and scorer → ONE development smoke on an old seed and old windows, which must simultaneously give: four lines contract-clean, no forced, V timing-selectivity lift ≥ 0.10 and balanced AUC ≥ 0.60, no defer collapse, no large-scale lower-tail erasure under the guard, reward co-directional with physical carbon. Any failure stops; no second guard strength. Only then the D′ preregistration with new seeds and new unread windows.

## 11. Implementation status after the M5 ruling (2026-09-05 morning)

| item | state |
|---|---|
| Symmetric guard η = 0.5 | implemented (`shrink_weights`, key `responsibility_shrink_strength`, default 1.0 bit-identical; `crd_w_raw` / `crd_w_guarded` in the batch); 4 tests |
| Cross-statistics | `stage_d_credit_cross.py` written and tested; the drift checkpoints (E, N_E × 7/8/9) are being re-sampled with `--save-raw` because the archived JSONs hold only class summaries |
| Mask margin | Java exports `ep_route_to_start_max/p95/n` (JUnit-tested, jar reinstall queued behind the re-sample); `hz_margin_probe` (nowait_planner on the D′ config, six training windows) + `margin_probe.py` set the margin as ceil(max/timestep)+1 steps; the value then replaces the 0.0 placeholder in `DPRIME_OVERLAY` and the config is regenerated |
| P0′ | tooling ready (`hz_p0` with `P0_VARIANT=dprime`, `p0_verdict.py dprime`); runs after the margin is frozen |
| Q4 corpus and scorer | `hz_corpus` (ST under the D′ config, decision + observation dumps) and `timing_selectivity.py` (lift ≥ 0.10, balanced AUC ≥ 0.60) written and tested; the corpus is built after the margin is frozen. Caveat recorded: the dump labels ST's intended action before the env-side mask re-route, and the module is scored per observation with a fresh recurrent state |
| Development smoke | not started; runs only after P0′ passes |

Nothing has been run on the scene beyond the M5 audit and its re-sample.

## 12. M5 cross-statistics (2026-09-05 09:34; drift checkpoints re-sampled with per-transition records, burn-in 5, 4000 steps per batch; `drl-manager/results/stage_d_credit_audit_raw/cross_statistics.json`)

Does the low weight fall on the DEFER transitions whose advantage is negative, i.e. on the corrective signal? E line, DEFER-dominated class (n = 2280 / 3125 / 7635 at checkpoints 7 / 8 / 9):

| ckpt | E[w \| DEFER, A<0] | E[w \| DEFER, A≥0] | P(w<0.2 \| A<0) | P(w<0.2 \| A≥0) | negative mass retained, raw | retained with η = 0.5 |
|---|---|---|---|---|---|---|
| 7 | 0.936 | 0.956 | 0.092 | 0.077 | 0.863 | 0.932 |
| 8 | 0.953 | 0.980 | 0.082 | 0.058 | 0.908 | 0.954 |
| 9 | 0.941 | 0.973 | 0.097 | 0.070 | 0.901 | 0.951 |

On every drift checkpoint the negative-advantage DEFER transitions carry a lower weight than the positive ones and are erased more often; 9–14% of the corrective DEFER mass is lost to the reweighting, against 0–5% of the ROUTE class's (ROUTE at checkpoint 9 even amplifies its negative advantages: 1.036 vs 1.018). The η = 0.5 guard recovers about half of the loss (+0.069 / +0.046 / +0.050; two of three above the 0.05 mark set in the script beforehand, the third at 0.046). N_E control: the DEFER class holds 52–101 transitions and its statistics are erratic (P(w<0.2 | A<0) = 0.000 / 0.178 / 0.000); nothing is claimed from it.

**Wording, per the ruling.** The cross-statistics support the reading that corrective credit against over-deferral is suppressed: the erasure concentrates on negative DEFER advantages consistently across the drift checkpoints, and the frozen guard restores about half of the lost corrective mass. The magnitude is modest (about a tenth of the corrective mass), so the guard is reported as a mechanism-supported regulariser of the identified defect, not as a demonstrated root-cause fix; policy-induced state-distribution shift and the re-warmed EMA/τ remain the stated limits, and this is not causal proof that the reweighting caused the collapse.

## 13. Mask margin frozen mechanically (2026-09-05 09:37)

Saturated-dispatch probe: `nowait_planner` on the D′ config over the six development training windows (35 jobs each, gateway jar 58413f681b0bc8f7 with the route→exec-start export). Route→exec-start delay: max = p95 = 1.0 s on every window (one simulation step: a job routed at clock t starts at t + 1), forced = 0, on-time 1.000. Margin = ceil(1.0 / 1.0) + 1 = **2 steps = 2.0 s**, written into `DPRIME_OVERLAY` and the regenerated `config_stage_d_dprime.yml` (manifest SHA recorded there). Disclosure: on this development load the 35 jobs fit the fleet at once, so no queueing delay was observed; the margin covers the dispatch latency, not contention. It is frozen by the rule regardless and is not to be revisited on carbon or training results; if the development smoke shows forced > 0 under the mask, that is a STOP, not a margin retune.

## 14. P0′ first run (2026-09-05 09:42; config c7d3d1e2…, margin 2 steps): STOP on one ledger field, every timing gate passed

Pooled over the six development windows (discounted return γ = 0.99; carbon kg):

| arm | discounted return | carbon |
|---|---|---|
| clean (ST, best window on time) | +9.31 | 0.0113 |
| blind (reactive wait) | −7.02 | 0.0167 |
| nodefer (S, start now) | −15.34 | 0.0181 |
| shrink | −28.06 | 0.0224 |
| always_defer (mask-routed) | −23.27 | 0.0296 |

Gates: discounted order clean > nodefer > always_defer pooled and by window majority — pass; clean beats nodefer on carbon — pass; **always_defer routed legally by the mask on all six windows (on-time 1.000, forced 0)** — pass; legacy P0 order, clip, cap, defer-no-arbitrage, shrink-worse-both — pass. **Verdict STOP_P0_PRIME on `contract_green` alone**: blind window 4 has `planner_n_unplanned_start = 3` with 32 of 35 planner dispatches. Cause: the env-side mask re-routed three of the blind planner's DEFERs at the last safe step, and the planner's consistency ledger counts a job it did not dispatch as an unplanned start. That ledger field was built to catch planner bugs; under D′ the mask is meant to override every arm's DEFER there, so the field now also counts the mask's own interventions.

**Rule amended before the rerun (disclosed):** the env exports its re-route count as `ep_mask_route_count`; the P0′ contract keeps completion ≥ 0.995, on-time ≥ 0.995, forced = 0 and every other zero-field, and requires `planner_n_unplanned_start ≤ ep_mask_route_count` (unplanned starts may be mask re-routes and nothing else). Run-1 rows are archived as `p0_dprime_run1_*` with their verdict; the rerun regenerates all thirty rows under the final D′ config (which now also carries the η = 0.5 guard in `crd.responsibility`, config SHA in the manifest). No threshold, arm, window or reward changed.

## 15. P0′ runs 2–4 (2026-09-05 09:44–09:56): PASS_P0_PRIME on run 4, with two disclosed instrument fixes

- **Run 2** (config e74f9980…, guard in place): same single STOP on the ledger field, because the env's re-route counter had not reached the result rows (evaluate.py passed `ep_*` keys only from the Java stats). Fix: env-side `ep_*` counters are passed through too. Rows archived as `p0_dprime_run2_*`.
- **Run 3**: PASS, **voided by me before acceptance**: `ep_mask_route_count` read 76,340 on a 35-job window. The env-side re-route was rewriting DEFERs on padding slots (no job; Java ignores them), so the counter was meaningless and the amended contract vacuous. Fix: padding slots are never re-routed nor counted (test added). Rows archived as `p0_dprime_run3_*`.
- **Run 4 (valid)**: PASS_P0_PRIME. Every gate true; `contract_bad` empty. Counters now read as they should: always_defer is mask-routed exactly 35 times per window (each job once) with on-time 1.000 and forced 0 on all six windows; blind window 4 has 3 unplanned starts = 3 mask re-routes; no other row is touched by the mask. Pooled: always_defer: R_disc -23.27, carbon 0.0296 | blind: R_disc -7.02, carbon 0.0167 | clean: R_disc +9.31, carbon 0.0113 | nodefer: R_disc -15.34, carbon 0.0181 | shrink: R_disc -28.06, carbon 0.0224.

No threshold, arm, window, reward or margin changed across the four runs; only the instrument (the counter's export and its padding exclusion). The development smoke (§4 step 6) starts from run 4.

## 16. Codex ruling on the progress report (2026-09-05 ~11:00): the running smoke is "development smoke A"; five points must close before any D′ preregistration

1. **Guard gate (Q1).** P(w_guarded < 0.2) ≤ 0.05 is near-tautological under η = 0.5 and stays only as a wiring sentinel. Substantive gate on E's last checkpoint: at least 100 valid DEFER-dominated transitions with advantage < 0; retained negative corrective mass R_guarded ≥ 0.90; if R_raw < 0.95 then R_guarded − R_raw ≥ 0.5 (1 − R_raw) − ε; and a bitwise check w_guarded = 1 + 0.5 (w_raw − 1).
2. **Selectivity corpus and scoring (Q2).** The 41:1 decision-point corpus is descriptive only (appendix). The main gate uses a **job-paired corpus**: per job at most one deterministic ST-defer sample and one ST-route sample; only states where DEFER and ROUTE were both legal; samples where ROUTE was forced by the deadline mask excluded; metrics job-equal-weighted. Scoring must **carry the recurrent state in the original time order** (per-observation reset is not a deployment gate). Two probabilities reported: the raw DEFER preference **before** the mask (the learning-selectivity gate) and the deployed probability **after** the mask (a safety diagnostic); passing only after the mask means the safety layer works, not that the policy learned timing.
3. **Margin (Q3).** The saturated probe must run on the six D′ cells × six development windows, reading no forecast effect. Margin = ceil(max route→start delay / timestep) + 1. If the maximum stays 1 s, margin 2 stands and smoke A may be read; if larger, smoke A is recorded as such, the margin is corrected once by the formula, and P0′ and the smoke are rerun; no further tuning.
4. **New seeds and windows (Q4).** Formal seeds frozen: 20260909–20260913. New judgement windows: exclude every read window and its full footprint; read no green, carbon or policy result; for each legal offset compute sha256("stage-d-prime-judgement-v1:" + offset); sort by hash and greedily pick six non-overlapping windows; fewer than six → STOP_WINDOW_SPLIT. Old training windows may keep serving training; old judgement windows serve development and smoke only.
5. **If the selectivity gate fails (Q5)** while the other health gates pass: stop D′; write the (DC, start-offset) / option action design first; zero-training expressibility, action–execution closure, contract safety and small-sample learnability probes; no training before those pass.

**P0′ validity addition.** `planner_n_unplanned_start ≤ ep_mask_route_count` is a count relation only. Before a formal P0′: `unplanned_start_cloudlet_ids ⊆ mask_routed_cloudlet_ids`, with at least the sha256 of the sorted id sets stored. Run 1 is to be called a development-phase interface-contract repair (it exposed and prompted the contract revision), not an instrument fix; run 4 is the first valid P0′ under the revised contract once the per-id closure is added.

The running smoke continues to completion as development smoke A; its selectivity score is recomputed with the recurrent, pre-mask, job-paired scorer without retraining.

## 17. Window rule outcome (2026-09-05 11:30): STOP_WINDOW_SPLIT under the ruled exclusions; numbers for the ruling

`stage_d_prime_windows.py` implements Q4 exactly (grid of PRE-spaced legal offsets, sha256("stage-d-prime-judgement-v1:"+offset) order, greedy non-overlap, six needed). The Stage D eval footprint is 2922 rows (tz 108 + max deadline 2518 + runtime 48 + horizon 144 + spline 4 + safety 100) on a 52,559-row turbine file.

| excluded set | intervals | largest free gap | legal candidates | chosen | status |
|---|---|---|---|---|---|
| read (15 k) + Stage D judgement (6) + Stage D training (6) | 27 | 1040 | 0 | 0 | STOP_WINDOW_SPLIT |
| read + Stage D judgement | 21 | 2119 | 0 | 0 | STOP_WINDOW_SPLIT |
| read only | 15 | 7259 | 2296 | 6 | OK |

So no fresh window of the Stage D footprint fits once the six Stage D judgement windows are excluded, regardless of the training windows: the binding term is the 2518-row maximum deadline of the c*_n50 cells inside the footprint. Not decided here; options for the ruling: (a) accept that the six Stage D judgement windows are "read" only for the E line's contract failure and not as carbon evidence for D′ — they remain excluded under the letter of Q4, so this option is a rule change; (b) shrink the footprint by tying the eval window to each cell's own maximum deadline instead of the global 2518 (the n20 cells are far shorter), which changes the preflight's definition; (c) move D′ to a second turbine series (the SDWPF set has others) with its own preflight, which changes the scene's wind data. Nothing is chosen; the rule as written returns STOP.

## 18. Instruments after the §16 ruling (2026-09-05 11:50)

- **Selectivity scorer v2** (`timing_selectivity.py`): job-paired (first sighting where DEFER was legal as the ST-defer sample; the routing sighting only if DEFER was legal there, otherwise the route counts as mask-forced and the job is dropped), GTrXL memory carried in time order per window, RAW probability with the mask key removed as the gate, DEPLOYED probability as the diagnostic, all-sightings appendix. Wiring verified on the health-smoke V checkpoint (no D′ keys, so RAW = DEPLOYED there) against corpus window k0: 35 jobs → 23 pairs, 7 excluded because ST itself routed only when DEFER was already illegal, 5 never deferred; raw lift +0.011, AUC 0.69 (a 56k policy trained without timing state, reported only as a wiring check).
- **Guard gate** (`stage_d_credit_cross.guard_gate`): n(DEFER, A<0) ≥ 100, R_guarded ≥ 0.90, R_guarded − R_raw ≥ 0.5(1 − R_raw) − 0.01 when R_raw < 0.95, bitwise |w_guarded − (1 + 0.5(w_raw − 1))| ≤ 1e-6 on the applied weight the audit now records.
- **Per-id closure**: env records the cloudlet ids the mask re-routes (`ep_mask_routed_ids`, sha, unknown count); the planner exports `planner_unplanned_start_ids`; `contract_ok_dprime` requires the unplanned set ⊆ the mask-routed set with zero unknown ids. P0′ run 5 with these columns is the formal one under the revised contract; run 4 stands as the first valid count-level PASS.
- **Six-cell probe** (§16 Q3) running on `config_stage_d_eval_dprime_dev.yml` (allowlist = the six training offsets), 36 runs.
- **Windows**: §17 STOP under the ruled exclusions; awaiting the ruling.

## 20. Window ruling (2026-09-05 ~12:00): formal judgement moves to the 2020 files of the same five turbines — a cross-year confirmation

Not adopted: relaxing "unread", reusing the six Stage D judgement windows, or per-cell footprints. Adopted: the same turbines, the same DC mapping, scarcity definition and action structure, with the formal judgement windows drawn from the turbines' 2020 series (32,225 rows each; six full footprints need 17,532 rows). Reasons as ruled: reusing old windows loses independent confirmation; per-cell weather confounds load scale with weather; changing turbines changes the spatial structure and would require re-certifying the HZ scene; changing the year keeps everything but adds a cross-year generalisation test.

Execution rule, frozen:
1. Collect every 2020 offset ever used in the repository for these five turbines and write their full-footprint intervals to `read_2020_intervals.json`, SHA frozen.
2. On the remaining legal offsets compute sha256("stage-d-prime-judgement-v1:2020:" + offset).
3. Greedily take six non-overlapping 2,922-row windows in hash order.
4. Read no green value, carbon or policy result of any candidate window.
5. `wind_csv_year`, the forecast provider's year and the audit's year must all be 2020, checked fail-fast at construction.
6. Fewer than six after the exclusions → formal STOP_WINDOW_SPLIT; no fallback to 2021 and no reduction of the window count.
7. Training keeps the frozen 2021 training windows; the judgement uses the new 2020 windows. This is written as a cross-year confirmation, not a same-year random split.

Order: the three background tasks finish first (smoke A as development only; the six-cell probe decides whether the margin stays 2; P0′ run 5 must pass the per-id set closure to be the formal P0′). Only when margin, P0′ and the development smoke all pass are the 2020 windows generated and frozen. No analytic planner is run on 2020 beforehand to pick "valuable" weather: 2021 certified the scene; 2020 is a one-shot external confirmation.

**§19 correction (12:10).** The 36th grid (cell c1_n50, development window k5) cannot be run at all: that window's offset 51,156 plus the 2,922-row eval footprint reaches row 54,078 of a 52,559-row file, and the provider raises `IndexError` when the series ends (the run's log shows exactly that; the second attempt failed the same way). So the six-cell probe is complete on the 35 grids that exist on this data; the margin statement stands on those. This also means the sixth development training window is only usable by the short-deadline cells, a fact the 2020 window rule must not repeat (§20 item 3 already requires full 2,922-row footprints inside the file).

## 21. P0′ run 5 — the formal P0′ under the revised contract (2026-09-05 11:47–12:0x)

Same config (e74f9980…), same arms and windows; rows now carry `ep_mask_routed_ids` (+ sha, unknown count) and `planner_unplanned_start_ids`. **PASS_P0_PRIME**, `contract_bad` empty, every gate true. Per-id closure on the only grid with unplanned starts: blind, window 4 — unplanned ids {18, 19, 20} = mask-routed ids {18, 19, 20}, unknown 0. always_defer: mask-routed exactly once per job on every window, on-time 1.000, forced 0. Rows and verdict archived under `reports/manifests/stage_d/dprime/p0_prime/run5/`. Runs 1–4 remain archived as development-phase interface-contract repair history.

## 22. Disclosure found while wiring the 2020 rule (2026-09-05 12:20)

`g1/compressed_timecap_s2/timecap_error_audit.json`, the audit that defines `calibrated_shrink_v1`, records `dc_turbines {0: [12, 36], 1: [95, 91], 2: [96]}` and `year: 2020`. The HZ scene's turbines are 123/10, 51/53, 112 (2021). The primary corruption used throughout Stage D was therefore calibrated on a different turbine set and year than the scene it was applied to. This predates D′ (it came with the HZ preregistration) and is reported here, not repaired: the shrink parameters are amplitude/lag statistics of the forecaster's error, and whether they transfer across turbines is a question for the ruling, not something to fix silently. The same file satisfies §20 item 5 ("audit year 2020") only nominally.

Year wiring verified for §20 item 5: the simulator reads `wind_csv_year` (SimulationSettings, default 2021); the forecast turbine map reads `wind_csv_year` too and already fails fast when a TimeCAP `csv_year` disagrees with it. The D′ 2020 judgement blocks will set `wind_csv_year: 2020` and the generator will assert it on every block and on the audit.

Prior 2020 usage of the HZ turbines: the planner gate's 2020 confirmation (`G1_PLANNER_GATE_VERDICT_2020.md`, `g1/config_C_2020.yml`) used turbines 12/36, 95/91, 96 — not the HZ set; TB12 calibration used turbines 100/101. The repository scan for `read_2020_intervals.json` is run and frozen before selection (§20 item 1).

## 23. 2020 read set and a dry fit check (2026-09-05 12:30; nothing selected or frozen)

Repository scan (tracked yml/json/md naming a 2020 wind year): the only 2020 offsets ever named are the planner gate's confirmation windows 2574 / 11554 / 13117 / 19171 / 22301 (`PLANNER_GATE_PREREG.md`), which ran on the sister turbines 12/36, 95/91, 96 — the HZ turbines 123/10, 51/53, 112 were never used on 2020. Two readings of §20 item 1:

| read set | excluded intervals | six 2,922-row windows fit? |
|---|---|---|
| conservative: exclude the sister-turbine gate windows as well | 5 | **no** (STOP_WINDOW_SPLIT) |
| strict, as worded ("offsets used for these five turbines"): empty | 0 | yes (6 × 2,926 = 17,556 of 32,225 rows) |

The frozen file `read_2020_intervals.json` currently holds the conservative set (SHA 69bc36d5…). Not decided here: the strict reading follows the ruling's own logic (changing turbines changes the spatial structure, so the gate's 2020 windows are not these turbines' windows, and no green value of these turbines' 2020 series has ever been read); the conservative reading fails by the rule's item 6. Awaiting the ruling before any selection is written.

## 24. Codex ruling on §22–§23 (2026-09-05 ~12:45)

1. **2020 read set: conservative.** "Unread" is not decided by turbine identity: sister turbines share the weather process, and results on those periods have already been seen, so the period is known indirectly; changing turbines does not make the same period blind again. The five planner-gate 2020 windows stay excluded; the 2020 series then holds four legal windows, and under the frozen six-window rule the 2020 plan is recorded **STOP_WINDOW_SPLIT**. No four-window variant, no footprint shrink, no reuse. Two clean continuations for the formal D′ judgement set, in priority order: (a) another complete valid year of the same HZ turbines; (b) failing that, a deterministically chosen turbine set never used for design or judgement, which must pass the zero-training HZ scene gate, the TimeCAP error calibration and the wiring gate afresh (no inheritance of the HZ certification). This STOP concerns the formal judgement set only; development smoke A continues.
2. **`calibrated_shrink_v1` is sister-turbine-calibrated**, with per-DC, per-lead amplitude and correlation structure (some in kW) that varies with turbine set, year and power scale; on the HZ turbines it is at most a transferability hypothesis. Stage D is not re-judged retroactively but must disclose that its primary corruption was calibrated on sister turbines. The formal D′ must recalibrate on the finally chosen turbines, using development years/windows only and never the judgement windows, under a new name (`calibrated_shrink_hz_v2`) with frozen error parameters, checkpoint, data range and audit SHA, re-verifying at least per-DC amplitude shrink, lead-wise bias, residual scale and cross-DC covariance. If the audit cannot be completed on the final scene, the corruption may only be called a synthetic shrink stress, never "real TimeCAP error". Smoke A, which uses the old v1, can only answer whether the guard and the training chain are healthy; it is not error-resistance evidence for D′. Once the final scene and the new error model are frozen: if the training scene is unchanged, only the deployment evaluation is redone; if the turbine set changes, the affected training is rerun.
3. The six-cell probe is written as **35/36**: the missing grid is a data boundary; "all six windows pass" is not to be written. The 35 grids at one-second start latency keep the two-step margin as a development setting; the formal runs rest on the per-id contract.

## 25. Continuation inventory after §24 (2026-09-05 13:00; nothing chosen)

**Option (a), another complete year of the HZ turbines: not available.** The five HZ turbines have 2020 (32,224 rows), 2021 (52,559 rows) and 2022 (2 rows, empty) only.

**Option (b), a never-used turbine set: available.** Structured scan (every tracked YAML/JSON parsed for `turbine_ids` / `dc_turbines`, every report scanned for `Turbine_<id>_<year>`): 28 real turbines have been used in some experiment config, audit or report — 1, 3, 10, 12, 15, 30, 36, 47, 51, 52, 53, 54, 60, 71, 90, 91, 95, 96, 100, 101, 105, 112, 113, 114, 115, 118, 123, 130 — plus the synthetic stretched series (7xxx/8xxx/9xxx). **106 real turbines have never been used** and all have complete 2021 files (and 2020 files of 32,224 rows). Inventory saved as `stage_a_out/turbine_usage_inventory.json`.

What (b) entails, per §24: a deterministic choice (hash-ordered, reading no wind value) of five never-used turbines mapped onto the HZ structure (two, two, one across DCs 0–2 with the same time-zone offsets); the zero-training HZ scene gate rerun on them (discovery + one-shot confirmation, including the capacity vector and the ×2 scarcity calibration, which depend on the turbines' power scale); `calibrated_shrink_hz_v2` audited on their development windows only; the wiring gate; then new training on the new turbines (the training scene changes) before any D′ judgement. The judgement windows for that scene follow the frozen hash rule on the new turbines' 2021 series with an empty read set (never-used turbines, never-used periods), or on 2020 if the ruling prefers cross-year — to be ruled. The selection rule is implemented (`stage_d_prime_turbines.py`) but not executed.

## 26. Development smoke A, part 1: contract and behaviour readings (2026-09-05 12:58; seed 20260903, certified windows k = 26/34/42, 56k steps per line, D′ config e74f9980…, 252 rows, zero failed)

Read as development only (§16, §24): the corruption here is the sister-turbine `calibrated_shrink_v1`, so nothing below is error-resistance evidence.

| line | tier | on-time | completion | forced | defer rate | env mask re-routes | carbon (kg, mean of 18) |
|---|---|---|---|---|---|---|---|
| N_V | hollow | 1.000 | 1.000 | 0 | 0.580 | 0 | 0.00760 |
| V | godeye | 1.000 | 1.000 | 0 | 0.504 | 0 | 0.00731 |
| V | calibrated_shrink_v1 | 1.000 | 1.000 | 0 | 0.503 | 0 | 0.00724 |
| V | shuffle | 1.000 | 1.000 | 0 | 0.515 | 0 | 0.00736 |
| V | anti | 1.000 | 1.000 | 0 | 0.507 | 0 | 0.00738 |
| N_E | hollow | 1.000 | 1.000 | 0 | 0.418 | 0 | 0.00704 |
| E | godeye | 1.000 | 1.000 | 0 | 0.449 | 0 | 0.00679 |
| E | calibrated_shrink_v1 | 1.000 | 1.000 | 0 | 0.439 | 0 | 0.00676 |
| E | shuffle | 1.000 | 1.000 | 0 | 0.462 | 0 | 0.00686 |
| E | anti | 1.000 | 1.000 | 0 | 0.460 | 0 | 0.00682 |

Contract clean on all 72 clean deployments, forced = 0 everywhere, and the env-side re-route never fired on an RL row (the logit-level mask removed the DEFER choice first, as designed). No collapse in either direction: defer rates sit between 0.42 and 0.58 across the four lines, against 0.038 / 0.956 in Stage D. The health runner's own verdict read FIX_AND_RERUN only because it ran before the chain's probe step (`probe_missing`); the chain and the post-smoke script re-run it after the probes. The guard gate, the recurrent pre-mask selectivity and the six-criteria verdict follow in part 2.

## 27. Development smoke A, part 2: guard gate, selectivity, six-criteria verdict = STOP_DPRIME_SMOKE (2026-09-05 13:20; artefacts and hashes in `reports/manifests/stage_d/dprime/smoke_a/`)

**Verdict: STOP_DPRIME_SMOKE.** Six of seven gates pass; the timing-selectivity gate fails. Per §16 item 5 the consequence is fixed: stop D′, write the (DC, start-offset) / option action design first, and run its four zero-training probes before any training.

| gate | reading | pass |
|---|---|---|
| contract clean, all lines | 72/72 clean rows on-time 1.000, completion 1.000 | yes |
| forced = 0 | 0 on every row | yes |
| defer not collapsed | V 0.504 (band 0.02–0.90) | yes |
| guard wiring sentinel | applied weight w′: P(w′ < 0.2) = 0.000, min 0.528 (= 1 + 0.5 (0.056 − 1)) | yes (after the column fix below) |
| guard, no mass erasure (E last, DEFER) | n(A<0) 585 ≥ 100; R_raw 0.994, R_guarded 0.997; bitwise error 6e−8 | yes |
| reward–carbon co-direction | V reward −146.5 → −111.0, carbon 0.00909 → 0.00731; E −140.0 → −100.5, 0.00876 → 0.00679 | yes |
| timing selectivity (V last, raw, recurrent, job-paired) | lift −0.022, balanced AUC 0.308 (need ≥ 0.10 / ≥ 0.60) | **no** |

**Selectivity detail.** Corpus: 210 jobs on the six development windows; 129 paired (44 excluded because their route was mask-forced, 37 never deferred by ST). Per window the balanced AUC runs 0.14–0.46 and the lift −0.043 to −0.005, so the failure is uniform, not one window. The deployed (post-mask) probabilities are identical to the raw ones on the paired states, as they must be: both members of a pair are DEFER-legal by construction, so the mask never binds there (a sanity check that passed). The all-sightings appendix reads lift +0.046, AUC 0.641 (8693 : 210), which is the queue-time exposure of the 41 : 1 corpus and is why it is not the gate. Reading: V's DEFER preference is about 0.60 at a job's first legal wait moment and 0.62 at the moment ST starts it, i.e. the policy leans slightly more towards waiting the longer a job has been present, the reverse of the planner's timing. V waits about 60 % of the time regardless of state. This is the E1 picture (§3) seen from the policy side: the value is in timing and the policy has not represented it.

**Guard reading.** On this smoke E never drifted (defer rate 0.45, stable), so there was no ratchet for the guard to correct: E[w | DEFER, A<0] 1.045 vs E[w | DEFER, A≥0] 0.971, the opposite sign from the Stage D drift checkpoints (§12). The guard is wired and harmless here; the smoke says nothing about whether it is needed.

**Caveat, recorded but not acted on.** 56k steps per line is a development budget. Whether selectivity would emerge at 400k steps was not tested, and testing it now would be tuning on a result; the pre-registered rule (§16 item 5) applies as written.

**Instrument disclosure (development phase).** The first judge run also failed `guard_wiring_sentinel`: the judge read the audit's `lower_tail_suppression`, which is computed on the raw weight (DEFER 0.066 > 0.05), while §16 item 1 defines the sentinel on the applied weight w′ (near-tautological under η = 0.5). `stage_d_credit_audit.summarize` now also emits `lower_tail_suppression_guarded` and `w_guarded_min` when the applied weight was captured; the judge reads the guarded field and fails when it is absent. The E audit JSON was re-summarised from the saved per-transition file (same n and mean w; the original is kept as `audit_E_last.prefix.json`, the first verdict as `dprime_smoke_verdict.prefix.json`). The verdict is STOP either way; 11 judge and audit tests pass.

**What is now moot and what is not.** The 2020 / never-used-turbine continuation (§24–§25) is moot for D′ training. The re-certification requirement and `calibrated_shrink_hz_v2` carry over to whatever successor scene the action design is tested on. Nothing has been selected.

## 28. Codex ruling on smoke A (2026-09-05 ~14:00) and the option design document

1. STOP_DPRIME_SMOKE upheld; the current step-wise DEFER architecture is not extended to 400k (that would be budget added after reading a result). The STOP means "the step-wise DEFER action failed the development gate for entering the long run", not "PPO can never learn timing".
2. The sentinel column fix is accepted: the written definition was the applied weight, the raw reading is kept, the verdict did not change.
3. Option approved as a **candidate action only**, not as an approval to train: ROUTE_NOW(d) and HOLD_FOR_GREEN(d); a hold is a single persistent commitment that terminates only on realised green or at the safety margin; no per-step re-decision once inside; creation, target DC, termination time and reason are logged and the cumulative consequence is attributed to the initial decision; no future truth and no implied best start; every RL arm and baseline shares one executor. Key risk named: HOLD_FOR_GREEN may write the smart causal rule into the environment so that a no-forecast policy also gains; if a matched no-forecast option comes close to the oracle, the design is rejected in favour of a discrete (DC, start-offset) action, and a "waiting licence" must not be mistaken for forecast value again.
4. The four probes (semantics, criteria, order) go into an independent design document, committed before implementation. Suggested freeze: expressibility, oracle-driven option captures ≥ 80 % of the ST advantage; predictive necessity, oracle-option total carbon ≥ 5 % below the strongest fully matched no-forecast option, with shuffle/anti negative controls; execution closure and contract, per id one creation, one start, consistent DC, no duplicates, stale reservations, over-capacity or forced, completion and on-time both 1; small-sample learnability, the existing job-paired held-out gate (lift ≥ 0.10, balanced AUC ≥ 0.60) plus the carbon actually captured by executing the behaviour-cloned actions. Any failure → no RL; if the option fails necessity or expressibility, the preregistered (DC, start-offset) fallback starts, the option is not modified on the spot.
5. The 2020 windows, the new turbine set and `calibrated_shrink_hz_v2` are all parked until the candidate action passes the four gates.

**Written in response:** `reports/OPTION_ACTION_DESIGN.md` v1 (162 lines, sha256 70bc5342…), frozen on commit. Contents: action primitive and executor semantics (§2), reward stream and SMDP relocation of the option's consequence to the creation step with the discounted return invariant (§3), arms including the matched blind option arms and the adversarial always_hold (§5), the four gates in the order 3 → 1 → 2 → 4 with the frozen numbers (§6–§7), the preregistered (DC, start-offset) fallback with κ ∈ {0, 1, 2, 4, 8, 16, 32, 64} (§8), implementation order (§10) and hazards (§11). Two numbers in it are my proposals beyond the ruling and are flagged for Codex: the executed-BC capture threshold 0.50 in gate 4, and the fallback's offset grid. Nothing is implemented.

## 29. Codex review of the option design v1 (2026-09-05 ~15:00) and Addendum A

Direction upheld, v1 not released for implementation until an addendum closes two structural points. Verified before writing it: (i) §3.2's invariance claim is wrong for states strictly between creation and termination (their return-to-go loses r_term), so relocation changes value targets and GAE there; (ii) the global policy's discount is γ = 0.999 with λ = 0.98 (`global_model` block), not 0.99; (iii) the HZ wait cap is 72 steps (`wait_cap_rows` in the trace cell name), so the offset grid must reach 72. Rulings: BC executed capture 0.50 approved; grid replaced by K(W) = {0} ∪ {2^q < W} ∪ {W}; the relocation connector is out of the four-gate phase (gate 3 reads raw rewards, lifecycle and contract; SMDP design comes after the gates as its own RL preregistration); HOLD must carry a fallback capacity reservation no later than latest-start or be masked, T1 needs immediate capacity, same-step releases use local accumulators, t_s is the execution-start event, and per-DC held count / PEs / tightest margin are observed; gate 1's denominator needs C_B − C_ST > ε; gate 4's corpus shortfall is INVALID_CORPUS with no on-the-spot extension.

Addendum A appended to `reports/OPTION_ACTION_DESIGN.md` with all of the above (A1–A8). One further disclosure from the check: `evaluate.py` reads the top-level `gamma` (0.99) for `global_reward_discounted_sum`, so the formal P0′ of §21 was judged at 0.99 rather than the global policy's 0.999. The reader is corrected to the `global_model` value and P0′ is re-read at 0.999 as a development-phase instrument repair; the 0.99 rows stay archived and both readings are recorded below when the re-read finishes. Implementation may start after the addendum commit: Java ledger with fallback reservations, env option and offset modes, analytic arms, four gates. No learner connector, no RL.

**P0′ run 6 (2026-09-05 20:19–20:26, same config e74f9980…, same five arms and six windows, only the metric's discount corrected to the global policy's 0.999): PASS_P0_PRIME**, every gate as in run 5. Carbon totals are identical to run 5 (deterministic replays); the pooled discounted returns move with the discount and keep the ruled order: clean −2.02 > blind −85.9 > nodefer −109.5 > always_defer −266.0, shrink −175.0 < clean (run 5 at 0.99: +9.31 > −7.02 > −15.34 > −23.27, shrink −28.06). A first launch of run 6 returned cached run-5 rows (the runner skips existing CSVs); those were removed and the run repeated from scratch. Rows and verdict in `reports/manifests/stage_d/dprime/p0_prime/run6/`; the run-5 rows stay in `run5/`. The formal P0′ is now run 6.

## 30. Option executor implemented; gate 3 smoke PASS (2026-09-05 20:47)

Implemented and committed after Addendum A (design doc Addendum B records the split): Java hold ledger (`holdCloudlet` / `takeHeld` / `releaseHeld` with the route reward booked at the decision clock / execution-start times per id / `ep_opt_*` counters; HOLD action index n + 1 + d), the Python `OptionExecutor` (reservation grid with the planner's constants, fallback reservation at the latest feasible start, hold legality per (slot, site), T1 green + capacity with same-step accumulators, T2 fallback, ledger), env mode `global_action_mode: option_v1` (2n actions, keys `batch_cloudlet_hold_allowed` (NB, n) and `dc_held_count / pes / tightest_margin`, releases before the Java step, one translation rule `plan_option_actions` for every arm, ledger CSV per episode), the score-based module's option mode (pairwise HOLD(d) columns masked by the legality key), the analytic option arms of §5 (planner mixin: a wait becomes HOLD at the reserved or cheapest feasible site, mask-repaired; `always_hold`), `config_stage_d_dprime_option.yml` (same six train windows as D′), runner phases `hz_opt` / `hz_opt_corpus`, judges `option_gates.py` (3 → 1 → 2 with the frozen numbers) and `option_bc.py` (gate 4 fit and classification score). Tests: 8 executor, 5 translation, 5 arms, 4 module, 5 judge, 3 Java ledger, 1 decision dump, all passing. Two interface repairs on the way, both instrument-level: Py4J passes Python ints as Integer (Long cast failed in `releaseHeld`), and a Long-keyed Java map cannot be looked up from Python (start times now returned as a list aligned with the ids).

**Gate 3 smoke on window k0, nine arms: PASS_GATE3_SMOKE.** Every arm: completion 1.000, on-time 1.000, forced 0, no refused or masked hold, every created option released (all by the green rule on this window except one or two margin releases on the perturbed and climatology arms), every held cloudlet found in the simulator's start events, route→start delay 0 steps, ledger row counts equal the counters. Artefacts and the jar hash in `reports/manifests/stage_d/dprime/option/gate3_smoke/`. The carbon numbers of this one development window are not gate readings and are not tabulated here; gates 1 and 2 are read only on the six-window run launched next (`hz_opt` → `option_gates.py`), whose references B and ST are the P0′ run-6 rows on the same windows and block.

## 31. Six-window run 1: STOP_GATE3 on an instrument hole; repaired once and rerun from scratch (2026-09-05 20:53–21:0x)

Gate 3 over the 54 rows failed on two: climatology_opt k1 (completion 0.857, five options still held at the episode end) and shrink_opt k1 (0.971, one). Cause, verified in the Java core: `hasUnfinishedCloudlets()` counted the unrouted queue and the datacentre brokers but not the hold ledger, so an episode whose last jobs were held ended as "all finished" while their fallback starts (s_f 557–620) lay beyond the step at which the simulator stopped (550). The step-wise arms never meet this because a deferred cloudlet stays in the global queue. Repair: held cloudlets count as unfinished (one condition in `hasUnfinishedCloudlets`). This is the one instrument repair gate 3 allows (§6): disclosed here, jar rebuilt, every option row and the gate-4 corpus deleted and regenerated under the repaired jar; nothing else changed. Gates 1 and 2 were not read from run 1 (the judge stops at gate 3 by construction).

## 32. Six-window run 2 (repaired jar 16df1990…): gate 3 PASS, gate 1 FAIL → STOP_GATE1_FAIL_FALLBACK_OFFSET (2026-09-05 21:03)

Gate 3: 54/54 rows clean (contract, forced 0, no refused or masked hold, every option released and started, route→start 0 steps, ledgers closed). Gate 1, expressibility of the oracle-driven option against the step-wise references (B = `reactive_wait_planner`, ST = reserving godeye planner, P0′ run-6 rows, same windows and block):

| window | C_B | C_ST | C_oracle_opt | capture | options (green / margin) |
|---|---|---|---|---|---|
| k0 | 0.002634 | 0.001476 | 0.002114 | 0.449 | 30 (30 / 0) |
| k1 | 0.005079 | 0.003191 | 0.004481 | 0.317 | 32 (31 / 1) |
| k2 | 0.003876 | 0.003870 | 0.003608 | invalid denominator (gap 0.2 % of C_B) | 32 (30 / 2) |
| k3 | 0.002718 | 0.000894 | 0.002188 | 0.290 | 30 (30 / 0) |
| k4 | 0.001946 | 0.001600 | 0.001980 | −0.098 | 26 (23 / 3) |
| k5 | 0.000422 | 0.000309 | 0.000634 | −1.883 | 23 (23 / 0) |
| pooled | 0.016674 | 0.011339 | 0.015006 | **0.313** (need 0.80) | |

Zero of five valid windows reach 0.70. Verdict by the frozen rule (§6, A6): gate 1 FAIL; gates 2 and 4 are not read; the option design is rejected as written and the preregistered discrete (DC, start-offset) fallback of §8 / A5 starts, with the option left unmodified.

Reading, descriptive only: with the oracle choosing site and wait-or-not once, and the executor releasing at the first moment the meter covers the job, the option captures under a third of the reserving planner's advantage pooled and is worse than the blind reactive waiter on the two windows with the smallest gaps. The one thing the option cannot express is "wait past the first green moment for a better one", and that is the timing the reserving planner uses. The fallback's fixed start offset expresses exactly that and reads no green at all, so the ruling's executor risk does not arise there. Artefacts: `reports/manifests/stage_d/dprime/option/six_window_run2/` (54 rows, ledgers, the gate-4 corpus decisions, jar hash).

## 33. Codex ruling on the option result → Addendum C; fallback implemented and launched (2026-09-05 21:4x)

Ruling (design doc Addendum C): fallback (DC, dispatch-offset) confirmed; executor reuse allowed with three independent tests; κ is the creation→route-call offset; the blind family is fixed_off(κ) for the nine grid values plus reactive_off, persistence_off, climatology_off, run first and frozen to a pooled-carbon blind* before any informed row exists; gate 4 in offset semantics (p_delay = Σ_{κ>0} P, label κ_oracle > 0) with the fit's hyper-parameters written down; two procedural disclosures (Addendum B uncommitted at the option judgement; A6's ε used before ratification, now ratified for the fallback); shrink_opt descriptive only; a fallback failure at gate 1 or 2 ends the action-space direction.

Implemented and committed before any fallback row: executor fixed-offset mode (`offset_grid`, `offset_allowed`, `create_fixed`, `releases_fixed`; curve-invariance test), env `global_action_mode: offset_v1` (n·|K| = 45 choices, mask (NB, 45), `plan_offset_actions`, nothing clipped), module offset mode (site score + offset head), arms (`*_planner_off` quantised down to the grid, `fixed_off` with the base commitment undone before its own booking), `config_stage_d_dprime_offset.yml` (same six windows), runner phases, `offset_gates.py` (freeze + 3 → 1 → 2 + gate 4), `option_bc.py --offset`. Tests: 5 executor, 4 translation, 6 arms, 3 module, 2 gates, 1 dump, 1 BC. Tracked tree clean at launch (commit 1e29227a); config sha 25446146…, jar 16df1990… (unchanged since §31). Chain `$JOB_TMP/off_chain.sh` runs the C6 order once; readings follow in §34.

## 34. Fallback run 1: gate 3 stopped on one row; forensic replay; placement-ledger repair (2026-09-05 22:10–23:0x)

Run 1 (jar 16df1990…): gate-3 smoke PASS; 72 blind rows clean; blind* frozen before any informed row (persistence_off, pooled 0.020480; then fixed_off_72 0.020924, fixed_off_0 0.020926, fixed_off_1 0.020926, fixed_off_2 0.021295, fixed_off_4 0.022028, reactive_off 0.022073, fixed_off_8 0.023094, fixed_off_16 0.025380, fixed_off_64 0.026719, fixed_off_32 0.026928, climatology_off 0.027672); informed rows generated; judge stopped at gate 3 on one of 108 rows: persistence_off k5, cloudlet 10 (32 PEs, κ = 16) released to DC 0 at step 175, execution start 224.03, route→start 49 steps. Gates 1, 2, 4 not read.

Codex ruling (§CODEX_PROMPT_2026_09_05_OFFSET_GATE3): not yet classifiable; my two pieces of evidence were insufficient (`dc_available_pes` is the allocation counter the code itself calls unreliable, and an empty DC queue does not exclude scheduler-waiting or in-flight submissions on other VMs); the planner's drift 160 PE had to be explained too; one forensic replay of the failed cell authorised, not counted as a repair; the fallback has its own single gate-3 repair only if the forensic proves the selector violated its own most-free-fitting semantics, with the repair confined to the placement ledger; a repair forces the whole chain and the B/ST references to be regenerated.

Forensic (evaluator-only per-VM dispatch snapshot, `PLACEMENT_SNAPSHOT_FILE`, behaviour-neutral; replay bitwise identical, 49.03 steps again): at clock 175 cloudlet 11 was dispatched to VM 4 (idle). At clock 176 cloudlet 10 was dispatched to **VM 4 again**: its scheduler showed exec 0 and waiting 0, the selector's free map said 32, and 15 other VMs were created, suitable and idle (exec 0, waiting 0, in flight 0), but VM 4 carried cloudlet 11 **in flight** (submitted the step before, in neither scheduler list yet). The selector therefore violated its declared "most-free fitting VM" rule by omitting cross-step in-flight submissions from the committed count; cloudlet 10 then queued behind 11 on the space-shared VM for a full runtime. The drift 160 PE is the planner's sentinel comparing its grid with `cap − dc_available_pes`, the never-recovering allocation counter (five finished 32-PE jobs), and says nothing about occupancy. Snapshot and rows in `reports/manifests/stage_d/dprime/option/forensic_k5/`.

Repair, within the placement ledger only (`PlacementLedger.java`, pure; 5 JUnit tests): committed PEs = exec + waiting + in-flight (submitted to the VM by this broker, not yet listed, not finished); selector = most-free fitting VM, lowest id on ties, −1 when none fits (unchanged SpaceShared queueing then). The four ruled properties are the tests: idle VM beats an in-flight VM; a previous-step submission counts; same-step dispatches spread; no idle VM → queued. Full Java test suite run on the rebuild. Consequence per the ruling: this is the fallback's one gate-3 repair; the 108 run-1 rows are archived as pre-fix; the whole chain reruns from the gate-3 smoke with a fresh blind* freeze, and the step-wise references B and ST (P0′ run 6) are regenerated under the repaired jar as P0′ run 7 with an A/B against run 6 before any downstream use; the option line is not rerun and its rows are marked as belonging to the old placement jar.

## 35. Post-repair rerun (jar b0b44d1e…): references unchanged, fallback gate 3 PASS, gate 1 FAIL on the window criterion → STOP_GATE1_FAIL_ACTION_SPACE_LINE_ENDS (2026-09-05 22:30–22:44)

P0′ run 7 under the repaired jar: PASS_P0_PRIME and every one of the 30 rows bitwise equal to run 6 (the placement defect never fired in the step-wise reference runs), so B and ST are unchanged; archived as `p0_prime/run7/`. Fallback chain from scratch: gate-3 smoke PASS; 72 blind rows; blind* frozen before any informed row = persistence_off (pooled 0.020400; fixed_off_72 0.020924, fixed_off_0 0.020926, fixed_off_1 0.020926, …); 18 informed rows; gate 3 on all 90 rows PASS (no violation of any kind).

Gate 1, expressibility of oracle_off against B = reactive_wait and ST = the reserving godeye planner:

| window | C_B | C_ST | C_oracle_off | gap (C_B − C_ST) / C_B | capture |
|---|---|---|---|---|---|
| k0 | 0.002634 | 0.001476 | 0.001150 | 44.0 % | 1.281 |
| k1 | 0.005079 | 0.003191 | 0.002836 | 37.2 % | 1.189 |
| k2 | 0.003876 | 0.003870 | 0.003736 | 0.2 % | invalid denominator |
| k3 | 0.002718 | 0.000894 | 0.000933 | 67.1 % | 0.978 |
| k4 | 0.001946 | 0.001600 | 0.001894 | 17.8 % | 0.150 |
| k5 | 0.000422 | 0.000309 | 0.000377 | 26.7 % | 0.397 |
| pooled | 0.016674 | 0.011339 | **0.010926** | 32.0 % | **1.077** |

Pooled capture 1.077 clears 0.80 (the quantised offset oracle is below the step-wise reserving planner on the pooled sum), but the window criterion (A6: 0.70 on all but one of the valid windows, i.e. four of five) reads three of five: k4 and k5, the two windows with the smallest absolute gaps (0.000346 and 0.000113 kg), capture 0.15 and 0.40. **Verdict by the frozen rule: gate 1 FAIL → STOP_GATE1_FAIL_ACTION_SPACE_LINE_ENDS** (C6). Gate 2 was not read and no blind-versus-oracle number was computed; gate 4 was not read (its corpus was generated, unused). Rows, ledgers, freeze, verdict and corpus in `reports/manifests/stage_d/dprime/offset/run2/`.

Recorded without acting on it: the window criterion was written by me in Addendum A6 as "all but one of the remaining windows" and ratified in Addendum C; it treats a window with a 0.0001 kg gap the same as one with a 0.0018 kg gap. Whether that is the right criterion is a question that can only be answered for a future preregistration; applying a different one to these rows would be a rule change after reading them.

**Codex ruling (2026-09-05 ~23:00) and closure of the action-space line.** The repair and the rerun are accepted (forensic proved the cross-step in-flight omission; scope, tests and whole-chain rerun as ruled; P0′ run 6/7 bitwise equal). The formal verdict stands: STOP_GATE1_FAIL_ACTION_SPACE_LINE_ENDS; pooled 1.077 does not override the frozen window criterion. The scientific statement is narrowed: *the frozen discrete dispatch-offset action did not meet the cross-window robustness certification, so the action line is terminated as preregistered*; it is not shown that the action cannot express forecast-driven scheduling (pooled capture 108 %, three main windows 0.98–1.28, the two failing windows carry about 8.6 % of the total avoidable carbon). The window criterion's weakness (equal votes for headroom 0.000113 and 0.001824 kg) is archived for future preregistrations (pooled effect plus headroom-weighted robustness, or a minimum absolute headroom), never applied back. Gates 2 and 4 stay unread, so it is not known whether a fixed reservation alone suffices, whether forecast timing has value over the strongest blind, or whether the four forecast summaries can support an offset decision; this gate-1 result used the full-truth analytic oracle and says nothing about the summaries. Next: a new prospective scene-and-interface design (unread windows with substantive headroom in every window; observations aligned with candidate actions, future_covered_energy(job, dc, offset), runtime coverage, deadline margin, all computed from the current forecast and identically for shuffle/anti; the same gate order expressibility → necessity → representation learnability → RL/EU-CRD); machines idle until it is approved; no reading of the existing gate 2/4 material; no further change to the offset action.

**Diagnostic disclosed, read only, not a judgement (2026-09-05 22:55–23:1x).** Before this ruling arrived, a read-only replay of oracle_off on the six windows with a dense every-step grid (`OFFSET_GRID_DENSE=1`, a diagnostic switch, off in every gate run; outputs in the job scratch directory, not in `stage_a_out`) was started to test the cause of the k4/k5 failure. Its motivation: the offset oracle's chosen κ equals the step-wise planner's actual wait rounded DOWN to the dyadic grid on every job of every window (histograms identical), a mean loss of 6–9 steps and up to 31, i.e. a systematic early start that on narrow-green windows moves part of the run onto brown. The dense-grid result is recorded below when it finishes and is used only to inform the next design's grid, never to rejudge these rows.

Dense-grid diagnostic result (2026-09-05 23:12; six windows, same block, same jar, `OFFSET_GRID_DENSE=1`, scratch outputs archived under `reports/manifests/stage_d/dprime/offset/diag_dense_grid/`):

| window | C_ST | dyadic oracle_off | dense oracle_off | dense capture |
|---|---|---|---|---|
| k0 | 0.001476 | 0.001150 | 0.001476 | 1.000 |
| k1 | 0.003191 | 0.002836 | 0.003191 | 1.000 |
| k2 | 0.003870 | 0.003736 | 0.003870 | (invalid denominator) |
| k3 | 0.000894 | 0.000933 | 0.000894 | 1.000 |
| k4 | 0.001600 | 0.001894 | 0.001600 | 1.000 |
| k5 | 0.000309 | 0.000377 | 0.000309 | 1.000 |

With an every-step grid the offset oracle reproduces the step-wise reserving planner's carbon to the last digit on all six windows, with route→start 0 everywhere. The k4/k5 failure of the frozen run was therefore entirely the dyadic grid's downward quantisation (a wait of 33–63 steps became 32), not a limit of the (DC, dispatch-offset) action; the dyadic grid's wins on k0/k1 (1.28/1.19) show the reserving planner is itself not optimal on wide windows. This is a diagnostic reading: the frozen verdict is unchanged, and the next design's grid rule is chosen before any of its rows exist.

## 36. Scene + interface design frozen; step 1 (turbines and data isolation) done (2026-09-06 00:40)

`reports/SCENE_INTERFACE_DESIGN.md` v1 with Addenda A and B is frozen at commit 0bbd6f7a (sha256 a01b46d4…): Codex approved the direction, the four v1 points (headroom-selected confirmation windows, data-dependent threshold, blind family narrower than the oracle, feature volume) closed in Addendum A, and the unit correction (brown factor in kg per kWh) and energy-weighted coverage closed in Addendum B; nothing else may change without an addendum.

Step 1, by the frozen rules, reading no wind value:
- Eligibility disclosure: the structured inventory (`turbine_usage_inventory.json`) had missed the legacy singular `turbine_id:` key, which the gateway still honours; a literal scan of every tracked yml/yaml/json/md/py outside the wind dataset and its preprocessing (`scene_v1.used_in_tracked_files`) excludes six more ids (57 and 124 as legacy defaults in configs; 2, 5, 9, 46 mentioned by tests, scripts or an old README), leaving 100 candidates. The first hash-ordered draw had taken 57 as the fifth turbine; that draw is discarded and recorded here.
- Turbines (hash rule stage-d-prime-turbines-v1, first five of 100): DC0 ← 133, 78; DC1 ← 22, 81; DC2 ← 94; DC3, DC4 without turbines. Files and their row counts and hashes are in `stage_a_out/scene_v1_isolation.json` (2021: 52,559 rows each, design year; 2020: 32,224 rows each, confirmation year).
- Confirmation windows (2020, sha256("scene-interface-v1:2020:" + offset), greedy non-overlap, all six kept whatever their headroom): 24398, 10829, 7479, 20843, 523, 14997. They are read only by §4.6 of the design.

Next: step 2, scene certification and calibration on 2021 (mechanism control on the first twelve hash-ordered 2021 windows; TimeCAP error calibration on these turbines; margin probe; P0′), zero RL.

## 37. Step 2a: mechanism control PASS; headroom gate leaves five of twelve → STOP_WINDOW_SPLIT (2026-09-06 00:04)

Certification arms on the twelve hash-ordered 2021 pool windows (offsets 259, 27364, 39729, 16477, 43574, 4240, 9154, 23604, 33225, 13223, 19663, 46630), defer-mode twin config 7135c3f5…, jar b0b44d1e…, 48 rows, contract clean on all: pooled carbon B (reactive_wait) 0.039255, ST (godeye) 0.027651 (−29.6 %), shuffle 0.059917, anti 0.070697 → mechanism control PASS (ST below B; both controls above B). TimeCAP error audit v2 on these turbines, 2021: λ_pooled 0.881 / 0.892 / 0.876 per DC, regression to the mean confirmed, no systematic false peaks, no spatial ranking error (`timecap_error_audit_hz_v2.json`).

Headroom gate (§2.2 + A1/B1: relative ≥ 0.15 and absolute ≥ 0.05 · C_brown_ref; C_brown_ref = 37.94 Wh of dynamic job energy × 0.5 kg/kWh = 0.01897 kg, absolute gate 9.49e−4 kg):

| pool k | offset | C_B | C_ST | gap rel | gap abs (kg) | pass |
|---|---|---|---|---|---|---|
| 0 | 259 | 0.001733 | 0.001430 | 17.5 % | 0.000303 | no (abs) |
| 1 | 27364 | 0.000312 | 0.000310 | 0.5 % | 0.000001 | no |
| 2 | 39729 | 0.001326 | 0.000771 | 41.8 % | 0.000554 | no (abs) |
| 3 | 16477 | 0.004512 | 0.001406 | 68.8 % | 0.003106 | yes |
| 4 | 43574 | 0.008796 | 0.008933 | −1.6 % | −0.000137 | no |
| 5 | 4240 | 0.003603 | 0.002391 | 33.6 % | 0.001212 | yes |
| 6 | 9154 | 0.004483 | 0.001899 | 57.6 % | 0.002584 | yes |
| 7 | 23604 | 0.001078 | 0.000346 | 67.9 % | 0.000732 | no (abs) |
| 8 | 33225 | 0.002855 | 0.001879 | 34.2 % | 0.000975 | yes |
| 9 | 13223 | 0.002229 | 0.000942 | 57.7 % | 0.001287 | yes |
| 10 | 19663 | 0.004664 | 0.003857 | 17.3 % | 0.000807 | no (abs) |
| 11 | 46630 | 0.003665 | 0.003485 | 4.9 % | 0.000180 | no |

Five windows pass; the rule needs six from the first twelve → **STOP_WINDOW_SPLIT** by the frozen rule. Observation, recorded and not acted on: the absolute gate (5 % of the all-brown reference) equals 25–90 % of a typical window's blind carbon on this scene, so it rejects windows with 42 % and 68 % relative gaps (k2, k7); the relative gate alone would pass eight of twelve. The design fixes the pool at the first twelve hash-ordered windows and fixes both thresholds, so neither can be changed for this pool without an addendum; the steps 2b/2c (shrink gate, margin probe, P0′) were not run. Rows, judge, manifest and audit in `reports/manifests/scene_v1/cert_pool12/`.

## 38. Codex ruling (scene-v2 continuation) and an invalid first search run, disclosed (2026-09-06 00:15–00:40)

Ruling (design doc Addendum C): v1 STOP kept permanently ("pool too short", not "no forecast value"); continuation on the next candidates of the same hash sequence with unchanged thresholds, first pass = sixth window, the five passing windows kept, mechanism PASS and audit v2 inherited and frozen; none passes → final STOP of this scene; 2020 sealed. Disclosed at the freeze: the 2021 file holds at most seventeen disjoint footprints, so after the pool of twelve only three candidates exist (offsets 49625, 36713, 30299); the search covers those three.

Search run 1 is **invalid** and is archived as such (`reports/manifests/scene_v1/v2_search_invalid_run1/`): the harness ran the candidates with the pool config and reset index 100–102, and the simulator takes its window from the block's allowlist by reset index modulo its length (`episode_offset_rows`: allow[k % 12] → pool windows 4, 5, 6), while the planner was given the candidate offset. The truth-curve planner therefore planned on a window the simulator was not running; its carbon collapsed to 13–87 % above the blind (which reads only the live meter), and the run reported STOP_SCENE_FINAL. Caught from the pattern (ST far above B on all three, B's carbon on "candidate 13" equal to pool window 4's) before any consequence was drawn. Fix: every run on windows outside the pool now uses its own config whose allowlist is exactly the window list (candidates → `config_scene_v2_defer.yml`; the development set → `config_scene_dev_{defer,offset}.yml`, block `svdev_*`), so reset index k and the planner offset always name the same window; the pool-12 certification (k = 0..11 on the pool allowlist) was aligned and stands. Search run 2 follows below.

## 39. Search run 2: sixth window found; A2 error gate STOP_ERROR_NOT_LOAD_BEARING (2026-09-06 00:22–00:24)

Aligned search: candidate 13 (offset 49625) passes at the first try (C_B 0.005110, C_ST 0.003961, gap 22.5 %, 0.00115 kg ≥ 9.49e−4), so the development set is 16477, 4240, 9154, 33225, 13223, 49625 (five from the pool, one from the continuation); candidates 14–15 were not run. Rows regenerated on the dev-set config for godeye and the calibrated shrink v2 arm (`sc2_*`, 12 rows, contract clean).

A2 (Addendum A2 of the scene design): C_shrink ≥ 1.05 · C_ST pooled and shrink above ST on ≥ 4 of 6 windows.

| dev k | offset | C_ST | C_shrink_v2 | ratio | green used ST / shrink (Wh) |
|---|---|---|---|---|---|
| 0 | 16477 | 0.001406 | 0.001327 | 0.943 | 28.55 / 28.80 |
| 1 | 4240 | 0.002391 | 0.003336 | 1.395 | 26.71 / 24.74 |
| 2 | 9154 | 0.001899 | 0.002478 | 1.305 | 27.58 / 26.38 |
| 3 | 33225 | 0.001879 | 0.001458 | 0.776 | 27.70 / 28.58 |
| 4 | 13223 | 0.000942 | 0.000887 | 0.942 | 29.63 / 29.76 |
| 5 | 49625 | 0.003961 | 0.003896 | 0.983 | 23.51 / 23.66 |
| pooled | | 0.012479 | 0.013380 | **1.072** | |

Pooled ratio 1.072 clears 1.05, but the shrunk forecast is above truth on only two of six windows; on the other four it is equal or better (k3: 22 % lower). **Verdict by the frozen rule: STOP_ERROR_NOT_LOAD_BEARING.** The calibrated real error of the deployed TimeCAP checkpoint on these turbines (λ ≈ 0.88, a mild pull toward the mean, no false peaks) does not consistently harm the analytic scheduler on this scene; a forecast pulled toward the mean even helps the reserving planner on half the windows, which says the truth-curve planner is not the optimum on them (consistent with §35's dyadic wins on k0/k1). Nothing about resisting this error can be shown here, so by A2 the line stops before RL; the margin probe, P0′ and gates 4.1–4.5 were not run; the 2020 confirmation windows were never read. Artefacts in `reports/manifests/scene_v1/v2_search_run2/`.

**Wording as ruled (Codex, 2026-09-06 ~01:00):** *under the frozen ST heuristic planner, the shrink forecast calibrated to the TimeCAP error raised pooled carbon by 7.2 % but did not cause consistent harm on most development windows, so it failed the robust-load-bearing gate.* The phrase "the real error is too mild" is withdrawn: the pooled harm exceeded 5 %; ST is not the terminal carbon optimum; and on the four windows where shrink did better the error may have corrected the heuristic's own defects rather than being small.

## 40. Codex ruling: scene stops, no third turbine set, next = forecast-quality ladder on a dominance-safe planner (2026-09-06 ~01:00)

1. STOP_ERROR_NOT_LOAD_BEARING confirmed with the narrowed wording above.
2. No further turbine draws (route a): drawing until the deployed checkpoint happens to fail badly would be outcome-driven selection; the five turbines keep their proven forecast value.
3. No post-hoc choice of a worse checkpoint (route b) unless the checkpoint or lead is chosen on forecast validation metrics alone, never on scheduling carbon.
4. Approved: a **real anchor plus a controlled error ladder** — the deployed TimeCAP as the natural anchor (allowed to come out not load-bearing); a pre-frozen weaker checkpoint or longer lead chosen only by forecast validation; a fixed shrink ladder λ ∈ {1.0, 0.75, 0.5, 0.25, 0}; shuffle / anti as extreme negative controls, never presented as deployed error; no rung chosen by its carbon effect: the whole forecast-quality → scheduling-loss curve is reported. Claim strength follows from where harm starts: only shuffle / anti harmful → "EU-CRD resists controlled severe contamination"; moderate, realistic rungs harmful → "resists realistic forecast degradation".
5. Before any new experiment the ST-not-optimal problem is closed: the same exact or provably dominating scheduler on the truth curve (settled on truth) and on each wrong curve (settled on truth), so a wrong forecast can never beat the truth schedule; verified by simulation replay first.
6. Future registrations may use the headroom-weighted robust gate (pooled loss threshold plus ≥ 80 % of the valid headroom in windows where the error is harmful); never applied back to this round.
7. Kept: the five turbines, the six 2021 development windows (16477, 4240, 9154, 33225, 13223, 49625), the sealed 2020 confirmation windows. Next: write the "error-quality ladder + dominance-safe planner" preregistration only; no carbon run before it is frozen.

Fact checked before drafting: the repository has no CP-SAT scheduler code (nothing imports ortools), but ortools 9.15 is installed in the venv, as is `scipy.optimize.milp` (HiGHS); one TimeCAP checkpoint exists (`finetune_TimeCAP_custom_sl96_baseline_4358062`, seq 96, pred 144 rows), so a weaker rung has to come from a longer lead or from a checkpoint trained and frozen on validation metrics before any carbon is read.

**Preregistration draft and its two review rounds (2026-09-06 01:20–02:10).** `reports/ERROR_LADDER_PLANNER_PREREG.md` v0 → Addendum A (Codex: simulator host power model RS500A_DYN with the MIPS-based utilisation, 65.64 W per 32-PE job, active-host count as a bounded approximation; truth rung must be OPTIMAL; exact ladder restricted to truth / shrink λ / shuffle / anti because the deployed TimeCAP issues a fresh 144-step forecast per decision and cannot supply a leak-free horizon curve, lead set withdrawn; offline solver = certification only) → Addendum B (Codex: all seven rungs OPTIMAL, no gap band; model–simulator closure ≤ 3 % on every rung's schedule with start alignment and contract, STOP_PLANNER_CLOSURE_RUNG; integer scaling in mW with a precomputable quantisation bound ≤ 0.1 % of C_brown_ref; 42 solves and replays; the 81.8 W source comment to be corrected). Committed at 9a103f51 (sha 196f1b4f…). The mechanical check caught an arithmetic error in B3 (the coefficient-rounding bound was about 7 × 10^−5 to 1 × 10^−4 kg, not 2 × 10^−6, against the 1.897 × 10^−5 threshold) → Addendum C: exact integer objective J = Σ(50 · brown_mW + green_mW), C = J / 3.6 × 10^11 (the factor ratio 0.5 / 0.01 = 50 is exact; preflight asserts the factors), the only remaining quantisation (green rounded to mW) bounded at 2.04 × 10^−7 kg, the three reference numbers stated once, per-job closure conditions (planned datacentre, exactly one dispatch/start/finish, all ledger counters zero). Committed at 24b5de60 (sha e0be0e95…); awaiting the final freeze; no carbon run before it.
