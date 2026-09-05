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
