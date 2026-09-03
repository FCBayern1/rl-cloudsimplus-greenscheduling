# Stage D preregistration (append-only) — DRAFT for Codex ruling, 2026-09-03

Steps 2 and 3 of the chain on the certified Scheme 2-HZ scene: does a vanilla RL policy that uses the forecast inherit the harm of the realistic forecast error, and does EU-CRD remove a significant part of that harm without giving up clean performance. Identity unchanged: accelerated-weather, marginal-carbon mechanism positive control. Nothing here has been trained or evaluated; the 50k health smoke may start only after this text is frozen and Codex has ruled. GPU (3060) stays PARKED until then.

## 1. Design: train clean, corrupt at deployment (Codex R-j)

Three training lines, same scene, same seeds, same hyper-parameters except the named difference:

| line | forecast channel during training | credit assignment |
|---|---|---|
| N  matched no-forecast | forecast inputs zeroed, observation shape identical | vanilla PPO |
| V  vanilla | clean truth-informed forecast (`green_oracle_mode: perturbed_godeye`, `perturb_tier: godeye`) | vanilla PPO |
| E  EU-CRD | the same clean forecast | EU-CRD block enabled (the frozen v5.2 configuration of `config_rl_step2_pilot.yml`, `crd.enabled: true`, nothing else changed) |

No policy is trained on a corrupted forecast. Corruption is applied only at deployment, so the measured effect is forecast error acting on a policy that learned to trust the forecast, not re-adaptation to a fixed error.

Deployment readings (frozen final checkpoint of each line, stochastic decode as registered for the scene, never argmax):

| line | clean | calibrated_shrink_v1 | shuffle | anti |
|---|---|---|---|---|
| N | C_N | (same input, reported once) | — | — |
| V | C_V0 | C_V1 | C_V,shuffle | C_V,anti |
| E | C_E0 | C_E1 | C_E,shuffle | C_E,anti |

Nine verdict-relevant readings (C_N, C_V0, C_V1, C_E0, C_E1 and the four negative-control readings). The primary corruption is `calibrated_shrink_v1` only (Codex R-k). shuffle and anti are deployment-time negative controls: never used to choose a checkpoint, tune a hyper-parameter, or train.

## 2. Scene and workload

- Fleet, power, planner-side physics: exactly the HZ scene (`config_s2hz_m2.yml`: zero-floor twins, 32-PE VMs, no splitting, brown 0.5, divisor 3000). RL blocks are derived from the ×2 block by a whitelisted key diff (`gen_stage_d.py`, test asserts no other key changes), the same way `gen_rl_step2_pilot.py` derived its blocks.
- Training workload: the six HZ cells are single traces; a policy is trained on the generator's c3_n35 trace (the RL step-2 pilot's cell, same runtime 48 / wait cap 72 / deadline 120 / 32-PE), which is not one of the six evaluation cells. Evaluation uses the six HZ cells.
- Reward: the registered per-action carbon objective of the scene's RL base (`global_reward_*`, `per_action_carbon_weight`), unchanged; gate 5 checks that reward improvement and physical carbon improvement have the same sign.

## 3. Windows (Codex R-l)

The HZ windows k=2/3/4/10/18/26/34/42 have been read for this fleet; k=1/9/17/25/33/41 were burned by Scheme 2. They are certified benchmark windows, not held-out data.

- Training: a frozen development allowlist of unread offsets, k ∈ {6, 8, 12, 14, 16, 20, 22, 24} (offsets 1009·k mod 44950; every one ≥ 2018 rows from any read window, episodes span ≈ 1050 rows). Implementation: env key `green_episode_offset_allowlist` (cycles the list instead of the 1009·k schedule); test `test_offset_allowlist.py`.
- 50k health evaluation: the certified HZ CONFIRMATION windows k=26/34/42 (already read; named "certified benchmark evaluation").
- Final long-run judgement: k ∈ {28, 30, 32, 36, 38, 40}, unread, non-overlapping with every window above. Offsets, turbines (E discovery map T123/10, T51/53, T112), config hash and trace hashes are frozen in `stage_d_manifest.json` before any of them is read. If these windows turn out unusable (data-range or overlap check fails at freeze time), the final judgement is run on the certified windows and is named "certified benchmark evaluation", with no held-out generalisation claim.

## 4. Checkpoint and seeds

- Checkpoint rule: the last checkpoint at the registered timestep count of each line. No selection on any evaluation number.
- Health smoke: 1 seed, 50k timesteps, checkpoints at 0 and 50k.
- Long run (separate prereg after the health gate): ≥ 3 paired seeds (5 if the GPU budget allows), identical seeds across N/V/E, direction gates ≥ 2/3 (or ≥ 4/5).

## 5. 50k health gate (no effect claim; pass/fail only)

- ck0 and ck50 saved and loadable for all three lines.
- Policy is neither all-route nor all-defer (defer rate in (2%, 98%) on the certified windows).
- clean vs calibrated_shrink_v1 change the action distribution of V (KL between action marginals > 0; `rl_step2_probe.py` instruments).
- V is forecast-aware: the probe's control-channel sensitivity is non-zero.
- EU-CRD: delta-r non-zero with non-zero variance; critic disagreement not pinned at its bound; responsibility gate active (rho not saturated).
- Reward and physical carbon move in the same direction between ck0 and ck50 for V and E.
- SLA, completion, capacity, forced/stale/wrong-DC contracts green on every evaluation episode.

Failure of any item is a pipeline finding to fix and re-run, not an effect result; a fix is committed as an addendum before re-running.

## 6. Long-run verdict gates (frozen now; evaluated only in the long-run prereg)

With C = pooled carbon intensity over the judgement grid (six cells × judgement windows), per line and reading:

1. Vanilla uses the forecast: (C_N − C_V0) / C_N ≥ 5%.
2. The error hurts vanilla: (C_V1 − C_V0) / C_V0 ≥ 5%, and V gives back ≥ 50% of its clean gain: (C_V1 − C_V0) ≥ 0.5 (C_N − C_V0).
3. EU-CRD does not buy robustness by ignoring the forecast: (C_N − C_E0) / C_N ≥ 5%.
4. EU-CRD removes at least half of vanilla's corruption increment: (C_E1 − C_E0) / C_E0 ≤ 0.5 · (C_V1 − C_V0) / C_V0.
5. Absolute conditions: C_E1 < C_V1; C_E0 ≤ 1.05 · C_V0; SLA / completion / capacity contracts green; reward and physical carbon improvements have the same sign; EU-CRD's delta-r, uncertainty gate and auditors are demonstrably active (logged statistics, not assumed).

Direction gates: each of 1–4 must hold in ≥ 2/3 (≥ 4/5) seeds. Negative controls are reported for V and E (shuffle, anti) but do not enter the verdict.

## 7. Stop rule

If the health gate cannot be passed after pipeline fixes, or the long run fails gates 1–2, the chain stops at "forecast error hurts a planner but a trained policy either ignores the forecast or is not hurt"; no re-tuning toward a pass. If gates 1–2 pass and 3–5 fail, the paper reports EU-CRD's failure on this scene as a negative result; the user's instruction that no negative paper is submitted is then a submission decision, not a reason to change the gates.

## 8. Implementation checklist before the 50k smoke (each with a test)

- `gen_stage_d.py`: N / V / E blocks derived from `config_s2hz_m2.yml` + the RL base of `config_rl_step2_pilot.yml` by whitelisted diff; manifest of hashes.
- Env `green_episode_offset_allowlist`; matched no-forecast channel zeroing verified to keep the observation shape.
- Deployment corruption for RL: `perturbed_godeye_provider` must expose `calibrated_shrink_v1` (audit file `timecap_error_audit.json`) besides godeye / shuffle / anti; equality test against `forecast_perturb.audited_future` on one window.
- Evaluation harness: `evaluate.py --global rllib` on the six cells × certified windows with the provider tier set per reading; result rows carry `planner_static_total_w`-equivalent hidden quantities for the RL path (provider tier, audit hash, checkpoint hash).
- 3060 work order written only after Codex approves this text.

## Addendum B (2026-09-03, Codex rulings R-m…R-p and the two hard blockers; supersedes conflicting text above)

- **R-m, four training lines.** N_V (vanilla, future forecast hollowed), V (vanilla, clean), N_E (EU-CRD, hollowed), E (EU-CRD, clean). Ten verdict readings: C_NV; V clean/shrink/shuffle/anti; C_NE; E clean/shrink/shuffle/anti. Gate 1 uses C_NV, gate 3 uses C_NE. Hollowing = the existing `forecast_mode: none` (the four `dc_future_*` fields zeroed, every other observation key bit-identical, shape unchanged; test `drl-manager/tests/test_forecast_hollow.py`). Current green, current carbon intensity and all non-forecast fields are kept.
- **R-n.** Training trace = c3_n35 at 32 PEs (`traces/s2/s2_r48_w72_c3_n35_pes32.csv`, generated by `gen_stage_d.train_trace`), for the 50k smoke and the long run alike; it may not be changed after any 50k result is seen. Before launch: check that the six evaluation cells' MI / PES / deadline observations fall inside the training block's normalisation support (`obs_cloudlet_mi_high` etc.).
- **R-o.** The EU-CRD subtree is copied from the frozen v5.2 block of `config_rl_step2_pilot.yml` with only `enabled` flipped; canonical SHA256 `700f3da6b8be34f0135f6e02bdfa10b4efe51acdb51b1e06da25447a0237fec6` (`stage_d_manifest.json`, built at a5829a38). N_V/V keep `enabled: false`. Physics, model, training budget identical across the four lines; the HZ keys (`max_cloudlet_pes: 32`, `split_large_cloudlets: false`, zero-floor twins, divisor 3000) override the old pilot's 8-PE settings; the exact diff against the HZ block is whitelisted and tested (`test_gen_stage_d.py`).
- **R-p.** 3060 runs the 50k smoke; Isambard may run the long run after a short equivalence smoke (same inputs, action shapes, loadable checkpoints, identical metric fields); a paired seed's N_V/V/N_E/E run as one block on the same hardware class; container, CUDA, PyTorch, Ray, jar, config and source hashes frozen; hardware never tied to an arm.
- **Hard blocker 1, windows (replaces §3).** `window_preflight.py` computes every window's row footprint from indices only (pre-margin 4, tz max 108, latest trace deadline, runtime 48, horizon 144, spline 4, safety 100; evaluation footprint 2922 rows for the six cells, training footprint 1075 rows for c3_n35) and confirms Codex's finding that 1009-row-spaced windows overlap. Read windows k ∈ {0,1,2,3,4,9,10,17,18,25,26,33,34,41,42} carry the evaluation footprint. Result (`reports/manifests/stage_d/stage_d_windows.json`, SHA in file): six final-judgement windows at offsets 13016, 21088, 29160, 37232, 45304, 48230 and six training windows at 6962, 15942, 24014, 32086, 40158, 51156, pairwise disjoint from each other and from every read window; historical read-vs-read overlaps are listed, not repaired. Fewer windows than required is STOP_WINDOW_SPLIT; the fallback to certified windows is deleted. Offsets are applied on both sides by `green_episode_offset_allowlist` (Java `SimulationSettings.parseIntList` / `episodeOffsetFor(index, range, allowlist)`; Python `episode_offset_rows`; tests on both). The health smoke evaluates on the certified HZ windows k=26/34/42 and is named "certified benchmark evaluation".
- **Hard blocker 2, reward truth table P0 (before any training).** Replay on the V block (training reward configuration, training windows): frozen blind `reactive_wait_planner`, truth-informed `godeye`, `calibrated_shrink_v1`, `always_defer`. Pass requires, per window and pooled: reward ordering equals carbon ordering for (clean vs blind), (shrink vs clean), (always_defer vs blind); clean improves reward and carbon over blind; shrink worsens both against clean; carbon-normalisation clip rate ≤ 5% of samples and zero cap hits (`ep_carbon_norm_clip_count`, `ep_carbon_norm_sample_count`, `ep_global_carbon_cap_count`); always_defer's reward strictly below the blind's (no defer arbitrage) with SLA and cap contracts closed. Reader `p0_verdict.py`; failure returns to Codex, no reward change before a ruling, no 50k.
- **Health gate split.** Wiring failures (missing checkpoint, tier not in effect, hash mismatch, missing log fields) may be fixed append-only and re-run. Substantive failures (policy collapse, zero forecast sensitivity, delta-r without variance, gate pinned at a bound, reward and carbon in opposite directions) are STOP_STAGE_D_HEALTH with no re-tuning.

## Addendum C (2026-09-03, P0 outcome and the reward variant put to Codex)

P0 on the legacy reward (the C-regime reward the HZ block inherited) is **STOP_P0** (`reports/manifests/stage_d/p0_verdict_legacy.json`). Decomposition of the global reward sum per episode, window 0: blind −238 = physical carbon −52.7 (0.00263 kg / 5e-05) + defer cost −219 + instant carbon price −1 + completion +35; truth-informed −3435 = −29.5 − 3440 − 1 + 35; calibrated shrink −2862 = −66.9 − 2822 − 9 + 35. The physical carbon term orders the arms correctly; the defer cost (`defer_cost_mode` unset → Java default "flat": −0.5 − 2.0·urgency charged at every sighting of a deferred job, the repeated-charging case Codex named) is 10–100× larger and inverts the ordering, and the instant per-action carbon price (job MI × the DC's green ratio at the dispatch instant) cannot see the 48-row run and sits at a floor of 0.11 per job for both the blind and the truth-informed arm. Clip rate 0, cap hits 0: normalisation is not the problem. always_defer fails the contract (ontime 0.09–0.37, forced 35) and scores below the blind, as it should.

Proposed variant "physical" (Codex ruling requested before any training): `defer_base_cost 0`, `defer_urgency_weight 0`, `per_action_carbon_weight 0`; everything else identical (`global_reward_beta 1`, `carbon_normalization_mode FIXED`, `carbon_normalization_fixed_max 5e-05`, completion shaping unchanged, deadline backstop and SLA contract unchanged). The global reward is then the physical ledger carbon of the step plus completion shaping, so reward ordering equals carbon ordering by construction; completion is enforced by the backstop, and lateness by the evaluation contract. Predicted P0 sums on window 0: blind −17.7, truth-informed +5.5, shrink −31.9. Generated as `config_stage_d_physical.yml` (`gen_stage_d.py 50000 physical`, diff against the legacy blocks limited to those three keys by test). P0 is replayed under it as a zero-training design check, labelled `p0_physical_*`; no policy is trained until Codex rules.
