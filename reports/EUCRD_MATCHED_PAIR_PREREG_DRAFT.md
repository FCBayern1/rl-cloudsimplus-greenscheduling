# Matched imperfect-forecast pair, V_err vs E_err: DRAFT v2 (2026-09-07)

**Draft, not frozen.** It is frozen only when G5 passes (reports/EUCRD_SIGNAL_GATE_PREREG.md §2 + Addendum A); nothing in it authorises a run. v2 carries the ruling of 2026-09-07: the training contamination is fixed at shrink 0.75 rather than chosen from a measurement, the budget is 120 000 steps × 3 paired seeds, the primary comparison and its numbers are fixed here, the transplanted noise floor is removed, and `cover_argmax` is reported without being made a bar the learner has to clear.

## 0. The question

Trained on the *same* imperfect forecast, with the same data, budget and paired seeds, does EU-CRD leave a policy that handles that error better than vanilla — and does it still handle further degradation better? The RL_V2 result (`FAIL_DET_SMOKE`) does not answer it: both lines trained on a perfect forecast, so the forecast-credit channel had nothing to attribute. That result stands unchanged.

## 1. Lines

| line | prior | credit | training forecast |
|---|---|---|---|
| V_err | fixed cover prior, zero-initialised residual | none (`crd.enabled: false`) | `perturb_tier: shrink75` |
| E_err | identical | EU-CRD on `candidate_carbon_regret` | identical |

Identical in every other respect, asserted by the config generator, which fails unless the only differing keys are the `crd` block and the run identity: same base block, action space, prior gain, budget, window allowlist and seeds.

**Why shrink 0.75 and not a tier picked from a reading.** The instrument has validated it, G5 is checking it in the learner path, and fixing it here means the next round answers one question — having seen this error, does EU-CRD handle it better than vanilla — without a round of choosing the training difficulty from results. The harsher rungs exist to test whether that carries to *further* degradation, which is a different question and is reported per rung.

**The calibrated tier's place.** `calibrated_shrink_v1` (audit block `timecap_error_audit_hz_v2.json`, turbine assignment 0:[133,78], 1:[22,81], 2:[94], matching this scene) is fixed here as a **secondary deployment-flavoured evaluation**, not part of the verdict. It is a shrink forecast constructed from measured error statistics, not the deployed TimeCAP model's own predictions, and must never be described as the latter.

## 2. Budget, fixed

120 000 steps × 3 paired seeds per line. Measured on this machine: vanilla 755 s per 8 000 steps, EU-CRD 1 561 s — **EU-CRD costs 2.07× the wall clock per step**, itself a reported overhead. Six trainings run serially ≈ 29 h, plus evaluation. The last checkpoint is the checkpoint of record, the seeds are fixed before the first run, and the budget is not extended on the basis of intermediate readings. Three seeds give preliminary repeatability, not a guarantee of stability, and the reading says so.

## 3. Evaluation

Deterministic decode (argmax, index ties), six reading windows, both lines evaluated identically on: godeye (clean), shrink75 (the trained-on contamination), shrink50, shrink25, shrink0, shuffle, anti, and `calibrated_shrink_v1` as the secondary evaluation. `cover_argmax` is reported at every tier as the zero-parameter reference — it shows how the rule itself degrades. The learner is **not** required to beat it at every rung; this round's question is E_err against its matched V_err.

Reported per line, per tier, per window and pooled: absolute carbon (primary), relative degradation against that line's own clean carbon, completion, capture against the causal expert, the gap to `cover_argmax`, and cost (training wall clock, per-iteration time, deterministic decision latency mean/p50/p95/p99 µs).

## 4. The primary comparison, fixed with numbers

**Primary**: the paired difference in pooled absolute carbon at **shrink 0.75**, E_err minus V_err, one pair per seed.

- **Benefit criterion**: the paired difference is negative in **all three** seeds, and the pooled improvement is **≥ 3 %** relative to V_err. Sign-consistent but below 3 %, or ≥ 3 % without sign consistency, is reported as *suggestive, not established*.
- **Clean guard**: E_err's pooled clean (godeye) absolute carbon exceeds V_err's by **no more than 3 %** relative. If it exceeds that, the primary claim fails regardless of the contaminated result, because a flatter degradation curve bought by a worse starting point is arithmetic, not robustness.
- The harsher rungs (shrink50, shrink25, shrink0, shuffle, anti) are reported **rung by rung** as a secondary question — does the benefit carry to unseen, stronger degradation. No rung may be promoted to the headline after the fact; the headline is shrink 0.75 or nothing.

## 5. Statistical reading

The paired differences of the three seeds are reported individually (value per seed, per window and pooled) alongside their mean and range. **No noise floor is transplanted into this comparison**: the 16 % cross-machine and 10–13 % cross-checkpoint figures recorded elsewhere describe different sources of variation and do not apply to a same-platform, last-checkpoint, paired-seed contrast. The spread of the three paired differences is the dispersion evidence this round has, and a claim resting on fewer than half the windows is reported as inconclusive.

## 6. Attribution

Showing E_err better than V_err would show that EU-CRD as configured helps. It would **not** by itself show that the new responsibility signal is the reason. The ablation that would show that — the same EU-CRD machinery with the responsibility source held uninformative, budget and seeds unchanged — is registered here as required for any causal claim about the mechanism, and is a separate run, not a reinterpretation of this one.

## 7. What passing licenses

One sentence: under the same imperfect-forecast training, EU-CRD's credit assignment lowers the absolute carbon of a policy facing that error, at a stated 2.07× training-time cost. Nothing about perfect-forecast training, deployment-time correction, the deployed TimeCAP model, or the sealed 2020 confirmation windows.
