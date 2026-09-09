# Disclosures the EU-CRD write-up must carry (2026-09-09)

Fixed while the matched pair is still training, so that the paper's account of the verification chain cannot be assembled after the fact from whichever checks happened to pass.

## 1. The wiring verification did not pass on the first attempt

The registered global-share gate (G5b) **failed**: at the last iteration the forecast channel's batch-mean responsibility share was 0.0087 against a registered 0.01. That verdict stands and is not revised. It was later established that the statistic mixes two different things — how *often* the signal fires (3.4 % of transitions, because at most decision states the forecast does not change which candidate is chosen) and how *strongly* it acts when it does (26 % of the responsibility on those transitions) — so a batch mean over a 96.6 %-zero column cannot separate a decorative channel from a sparse but decisive one. The wiring check was therefore re-registered as an independently frozen switch comparison rather than repaired in place.

Suggested appendix wording:

> The original global-share gate did not pass; it was subsequently found to conflate the frequency of the signal with its conditional strength. The original verdict is retained, and wiring was verified instead through an independently registered switch comparison.

## 2. The switch comparison's criterion was revised after a diagnostic

The comparison first measured the gradient of the **whole** PPO loss and did not meet its threshold (cosine 1.0000). A decomposition then showed that the value term's gradient norm is roughly 300× the policy surrogate's, so the surrogate's change was diluted about 300-fold in that quantity. The criterion was re-registered on the **policy surrogate gradient**, carrying over the original threshold and aggregation unchanged and changing only the measured object. **This revision followed the diagnostic and must be disclosed as such**; the paper may not present only the final PASS.

## 3. Wording limits carried from the rulings

- "The forecast responsibility changes the gradient of the policy surrogate loss" is supported. "The policy update changed by X %" is **not**: global-norm gradient clipping over actor and critic together, and the optimiser state, sit between the gradient and the update.
- The truth curve used to compute the responsibility is **privileged training supervision** provided by the simulator. That the actor's observation is unaffected was verified for the gated runs; that does not make the training procedure free of future information.
- `calibrated_shrink_v1` is a shrink forecast constructed from measured error statistics. It must never be described as the deployed TimeCAP model's own predictions.
- The one-iteration screening trial reads LITTLE_CHANGE and carries no conclusion about EU-CRD's effectiveness in either direction.

## 4. Cost

EU-CRD's training-time overhead is measured and reported alongside any benefit: 1 561 s against vanilla's 755 s per 8 000 steps on the workstation (2.07×), 2 923 s per 8 000 steps on Isambard-AI Phase 2. Deterministic decision latency is reported per line from the evaluation.
