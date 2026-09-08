# Switch comparison reading: the responsibility reaches the gradient, and barely moves it (2026-09-08, 14:0x)

Preregistration reports/EUCRD_SWITCH_AB_PREREG.md (frozen 140d79c5 before any run). Four runs; runs 1–3 are archived as invalid or unresolvable (`reports/manifests/eucrd_gates/switch_ab_run{1,2,3}_*`), run 4 (code c5983010: random state and the critic's value-target variance EMA both reset before each gradient evaluation and restored afterwards) is the run of record. 198 loss calls on the `g5_err` configuration over 16 000 steps, so the state is a lightly trained one and the conclusion is about the reweighting once enabled, not about the whole of training. Verdict **STOP_SWITCH_AB:W3_gradient_differs**. The matched pair does not start under this document.

## 1. The diagnostic is closed

The paired null — the same advantages scored twice — is **exactly zero on every one of the 198 calls** (relative L2 median 0, max 0; cosine 1.0 at every call). Run 3, with only the random state reset, still had a null of 0.0031; the one remaining state write found by the source audit was the critic EMA, and restoring it is what brought the null to zero. That is now supported by measurement, not only by reading the code. Every A/B difference below is therefore above its own null (198/198 calls), and the question W3 asks can finally be answered.

## 2. Gates

| gate | requirement | run 4 | |
|---|---|---|---|
| W1 weights differ | mean \|Δw\| > 1e-3, max > 1e-2 | 0.0129 / 0.475 | pass |
| W2 change is targeted | firing / non-firing ≥ 2 | 0.212 / 0.0066 = **32.4** | pass |
| W3 gradient differs | cosine < 0.9999 (clamped) | **min 0.999969, mean 0.9999994** | **fail** |

W1 and W2 now stand on three consistent runs (Δw mean 0.0123 / 0.0124 / 0.0129; targeting 33.5 / 33.4 / 32.4).

## 3. Direction and magnitude, reported separately

- **Direction** (the cosine, which W3 is registered on): the closest any call comes to the threshold is 0.999969, i.e. 1 − cos = 3.1e-5 at the worst call and 6e-7 on average. The update direction is essentially unchanged.
- **Magnitude** (relative L2, a diagnostic that does not stand in for W3): median 6.4e-4, max 8.2e-3, every call above its zero null. Gradient norm ratio B/A 1.0000048. The update's size is essentially unchanged too.

So the mechanism *reaches* the gradient — the change is real and resolvable — but it moves it by about six parts in ten thousand. The arithmetic behind that is visible in the same dump: the weights change strongly on 63 of 2 029 valid transitions per call (3.1 %), by 0.21 on average there, and by 0.0066 elsewhere from the mean-preserving renormalisation; a gradient summed over two thousand transitions barely notices a 0.2 change on three percent of them.

## 4. Advantage sign (W4, reported)

Negative-advantage transitions: 2.2 % are damped (w_A < w_B), mean ratio 0.9970. Positive: 1.0 % damped, mean ratio 1.0022. The most-changed transitions are 100 % firing cells, 58 % of them with negative advantage. Damping a negative advantage weakens the signal to stop doing something; this is recorded as a property of the mechanism at this state, and nothing about benefit or harm is claimed from it.

## 5. What this establishes and what it does not

Established: the forecast responsibility changes the sample weights (W1), does so where the forecast changed a decision (W2), and the change propagates to the policy gradient (every call above its null). Not established: that it changes the *direction* of learning by the registered margin (W3 fails), or that any of this helps (untested). The result is not evidence that EU-CRD is ineffective; it is evidence that, at this state and budget, the reweighting perturbs the update very little.

## 6. For a ruling

Under the frozen document W3 fails and step 3 does not start. The reading is the one the ruling anticipated ("空对照归零，但 W3 不过"): the diagnostic is closed and the number is what it is. The options are the ruling's to choose: accept that the effect on the update is below the registered margin and stop; or decide that a mechanism which reaches the gradient at all is enough of a wiring check for the effect experiment to be worth running, and re-register W3 with a margin that matches the measured scale — a decision to be frozen before any further run, not fitted to this one. The responsibility coefficients, the anomaly gate and the shrink strength were not touched.
