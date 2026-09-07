# Matched imperfect-forecast pair, V_err vs E_err: DRAFT (2026-09-07)

**Draft, not frozen.** It cannot be frozen before G5 passes (reports/EUCRD_SIGNAL_GATE_PREREG.md §2 + Addendum A), and the two open choices in §2 need a ruling. Nothing in it authorises a run.

## 0. The question this pair answers

Trained on the *same imperfect forecast*, with the same data, budget and paired seeds, does EU-CRD leave a policy that loses less carbon when the forecast degrades further — without buying that robustness by being worse to begin with? The RL_V2 result (`FAIL_DET_SMOKE`, EU-CRD +30.9 % vs vanilla +27.7 % at λ0.75) does not answer it: both lines trained on a perfect forecast, so the forecast-credit channel had nothing to attribute. That result stands as its own finding and is not reinterpreted here.

## 1. Lines

| line | prior | credit | training forecast |
|---|---|---|---|
| V_err | fixed cover prior, zero-initialised residual | none (`crd.enabled: false`) | the contaminated tier of §2 |
| E_err | identical | EU-CRD on `candidate_carbon_regret` | the same tier, the same rows |

Identical in every other respect, asserted by the config generator: same base block, same action space, same prior gain, same budget, same seeds, same window allowlist. The only two differing keys are the `crd` block and the run identity.

## 2. Two choices that need a ruling

**(a) Which contaminated tier trains both lines.** The instrument measures how often each tier changes a decision on the reading windows: shrink 0.75 changes 59 of 105, shrink 0.50 79, shrink 0.25 82, shrink 0 89. Recommendation: **`calibrated_shrink_v1`**, the tier generated from the real-error audit, so the training contamination is the realistic one and the synthetic ladder stays available as unseen degradation at evaluation. Verified available for this scene: `g1/compressed_timecap_s2/timecap_error_audit_hz_v2.json` carries the audit block for exactly this turbine assignment (`dc_turbines` 0:[133,78], 1:[22,81], 2:[94]); the training twin does not yet set `perturb_error_params`, so that key has to be added to both lines identically. Before the pair is frozen, three zero-training runs of the instrument on this tier should report how often it changes a decision, so the training contamination's strength is a measured number rather than an assumption. Fallback if that measurement shows the calibrated tier barely moves any decision: shrink 0.50, the middle rung.

**(b) Budget and seeds.** Measured on this machine (CPU learner, RL_V2 lines): vanilla 755 s per 8 000 steps, EU-CRD 1 561 s — EU-CRD costs **2.07× the wall clock per step**, which is itself a reportable overhead. Options: 56 000 steps (the RL_V2 budget; 1.5 h + 3.0 h per seed, 3 seeds ≈ 13.5 h) or 120 000 steps (3.1 h + 6.5 h per seed, 3 seeds ≈ 29 h). Recommendation: **120 000 steps × 3 paired seeds**, because 56 000 was a smoke budget and a null result at smoke length would be uninformative about the mechanism.

## 3. Evaluation, fixed before the runs

Deterministic decode (argmax, index ties), the six reading windows, the frozen error ladder godeye → shrink75 → shrink50 → shrink25 → shrink0 plus shuffle and anti, both lines evaluated identically, the zero-parameter `cover_argmax` rule reported at every tier as the reference the learner has to beat.

Reported for each line, each tier, per window and pooled:

1. **absolute carbon** — the primary quantity, both clean and contaminated;
2. **relative degradation** against that line's own clean carbon;
3. **capture** against the causal expert, and the gap to `cover_argmax`;
4. **completion**, so no carbon reading is bought by dropping work;
5. **cost**: training wall clock, per-iteration time, and deterministic decision latency (mean/p50/p95/p99 µs) for both lines, per the repository's efficiency rule.

## 4. Pre-registered reading rules

- A smaller *relative* degradation is **not** sufficient. If E_err's clean absolute carbon is worse than V_err's, a flatter degradation curve is the arithmetic of a worse starting point, not robustness. The claim requires E_err to be no worse than V_err on clean carbon within a stated tolerance **and** lower on contaminated absolute carbon.
- Both figures are read per window and pooled; the pooled figure is primary, and a claim that rests on fewer than half the windows is reported as inconclusive, not as a win.
- The cross-machine noise floor recorded for this testbed (16 % across machines, 10–13 % across checkpoints) applies: an effect inside it is reported as not resolvable at this seed count.
- The 2020 confirmation windows stay sealed.

## 5. What passing would license

A single sentence in the paper: under the same imperfect-forecast training, EU-CRD's credit assignment reduces the carbon cost of further forecast degradation, at a stated training-time overhead. It would not license any claim about perfect-forecast training, about deployment-time correction, or about the sealed confirmation windows.
