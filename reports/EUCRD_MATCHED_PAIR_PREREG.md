# Matched imperfect-forecast pair, V_err vs E_err (frozen 2026-09-08)

Frozen by the commit that adds this file, before any run of the pair. Supersedes reports/EUCRD_MATCHED_PAIR_PREREG_DRAFT.md (v2), which it carries over with the launch preconditions, seeds and evaluation completed. Precondition for launch: the light independent re-check of the switch comparison under reports/EUCRD_SWITCH_AB_PREREG.md Addendum B returns SWITCH_AB_PASS (W1, W2, W3′); the runner refuses to start otherwise. Nothing in this document is a carbon claim.

## 0. The question

Trained on the same imperfect forecast, with the same data, budget and paired seeds, does EU-CRD leave a policy that handles that error better than vanilla, and does that carry to further degradation? The RL_V2 result (`FAIL_DET_SMOKE`, both lines trained on a perfect forecast) stands unchanged and is not reinterpreted. What the switch comparison established is only that the forecast responsibility changes the gradient of the policy surrogate loss, targeted on the transitions where the forecast changed a decision; whether that helps is this experiment's question.

## 1. Lines and their identity

| line | prior | credit | training forecast |
|---|---|---|---|
| V_err | fixed cover prior, zero-initialised residual | `crd.enabled: false` | `perturb_tier: shrink75` |
| E_err | identical | EU-CRD, `crd.forecast.source: candidate_carbon_regret` | identical |

Both derive from the RL_V2 CPU-learner twin (`rl2_V_s2_r48_w72_c3_n35`, `rl2_E_s2_r48_w72_c3_n35`), whose blocks differ only in the `crd` block and the run identity. The generator asserts: the two lines differ only in `crd` and identity; within `crd`, only `enabled` and `forecast.source` differ; gradient clipping (`global_model.max_grad_norm` 20, `local_model.max_grad_norm` 0.5, global-norm over all module parameters, critic included) is identical; the window allowlist, action space, prior gain, batch sizes and budget are identical. Its manifest records the config hash.

## 2. Budget, seeds, checkpoint of record

120 000 steps per line (15 PPO iterations of 8 000), **paired seeds 20260911, 20260912, 20260913**, fixed here. Six trainings run serially in pairs (V then E per seed) so that every completed seed is a complete pair. The **last checkpoint** of each run is the checkpoint of record; no intermediate checkpoint is selected and the budget is not extended on the basis of intermediate readings. Measured cost on this machine: vanilla 755 s and EU-CRD 1 561 s per 8 000 steps, so EU-CRD costs 2.07× the wall clock per step, reported as an overhead. Six trainings ≈ 29 h plus evaluation. Three seeds give preliminary repeatability, not a guarantee of stability, and the reading says so.

## 3. Evaluation

Deterministic decode (argmax, index ties), the six reading windows of the RL_V2 manifest, both lines and the zero-parameter `cover_argmax` rule on every tier: godeye (clean), **shrink75** (the trained-on contamination), shrink50, shrink25, shrink0, shuffle, anti; and `calibrated_shrink_v1` as a secondary deployment-flavoured evaluation (audit block `timecap_error_audit_hz_v2.json`, this scene's turbines; a shrink forecast built from measured error statistics, never to be called the deployed TimeCAP model's predictions). `cover_argmax` shows how the rule itself degrades and is not a bar the learner must clear.

Reported per line, per seed, per tier, per window and pooled: absolute carbon (primary), relative degradation against the line's own clean carbon, completion, capture against the causal expert, the gap to `cover_argmax`, and cost (training wall clock and per-iteration time; deterministic decision latency mean/p50/p95/p99 µs, per the repository's efficiency rule).

## 4. The primary comparison, fixed with numbers

**Primary**: the paired difference in pooled absolute carbon at **shrink75**, E_err minus V_err, one pair per seed.

- **Benefit**: the paired difference is negative in **all three** seeds and the pooled improvement is **≥ 3 %** relative to V_err. Sign-consistent but below 3 %, or ≥ 3 % without sign consistency, is reported as *suggestive, not established*.
- **Clean guard**: E_err's pooled clean (godeye) absolute carbon exceeds V_err's by **no more than 3 %** relative. If it does, the primary claim fails regardless of the contaminated result: a flatter degradation curve bought by a worse starting point is arithmetic, not robustness.
- **Completion**: reported alongside every carbon figure; a carbon advantage accompanied by lower completion is reported as such, not as an advantage.
- The stronger rungs (shrink50, shrink25, shrink0, shuffle, anti) are reported rung by rung as the secondary question, whether the benefit carries to unseen degradation. No rung is promoted to the headline after the fact; the headline is shrink75 or nothing.

## 5. Statistical reading

The three paired differences are reported individually (per window and pooled) with their mean and range. No noise floor is transplanted from other comparisons; the spread of the three paired differences is the dispersion evidence this round has. A claim resting on fewer than half of the windows is reported as inconclusive.

## 6. Attribution

E_err better than V_err would show that EU-CRD as configured helps. It would not show that the regret signal is the reason; the ablation for that — the same EU-CRD machinery with the responsibility source held uninformative, budget and seeds unchanged — is registered as required for any causal claim about the mechanism, and is a separate run.

## 7. Sealed

The 2020 confirmation windows stay sealed. Nothing here touches the anomaly gate, ρ_min, the shrink strength, the warmup or the signal's scale.

## Addendum B — platform, and which run is of record (frozen 2026-09-09 12:2x UTC, before any Isambard job of the pair starts)

The pair is launched twice, and the rule for which one counts is fixed **here, in advance**, so that no selection on results is possible.

- **Isambard-AI Phase 2** (aarch64, one GPU per line, six lines in parallel, same frozen config `8668f98342e6ff21`, same seeds 20260911/12/13, `--time=15:00:00`) is the **run of record if and only if all six lines complete before the announced maintenance window** (2026-09-10 06:00–20:00 BST = 05:00–19:00 UTC, docs.isambard.ac.uk/service-status/planned_maintenance). Measured there: 2 923 s per iteration on the E line against 1 561 s locally, so 15 iterations ≈ 12.2 h against a 16.6 h margin.
- **The workstation run** (started 2026-09-09 12:06 UTC, six lines serial, ≈ 29 h) continues as the hedge. It becomes the run of record **only if** the Isambard set does not complete in full before the window.
- If both complete, the Isambard set is read and the workstation set is reported alongside as a cross-platform replication, never as an alternative headline. Neither set may be partially mixed with the other: all six lines of whichever set is read come from one platform.
- The platform of the run of record is stated in the reading, together with the per-iteration cost on it.

Nothing else changes: the config, the seeds, the budget, the tier, the last-checkpoint rule, the primary comparison, the clean guard and the sealed 2020 windows all stand as registered.
