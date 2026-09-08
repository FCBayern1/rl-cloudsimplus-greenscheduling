# Switch comparison: does the forecast responsibility change learning at all? (frozen 2026-09-08)

Frozen by the commit that adds this file, before the comparison is implemented and before it is run. It is step 2 of the ruling of 2026-09-08: step 1 archives the G5 STOP without moving any threshold, this step asks whether the mechanism reaches the gradient, and only step 3 asks whether that helps. Nothing here reinterprets the G5 verdict, and nothing here is a carbon claim.

## 1. What is compared

Within a single loss call, on the same sampled batch and the same module weights, with no update between the two variants:

| variant | responsibility input |
|---|---|
| A | the pipeline as configured, `crd_forecast` as computed from `candidate_carbon_regret` |
| B | identical, with `crd_forecast` set to zero before the shares are formed |

Both variants produce responsibility weights and reweighted advantages; both are pushed through PPO's own loss to obtain the policy gradient with respect to the module's parameters. Neither gradient is applied — the comparison is read, then the run continues on variant A exactly as it would have without the diagnostic. The running estimator state the shares depend on (share-scale EMA, anomaly EMA, reweight call counter) is snapshotted before variant B and restored after it, so the diagnostic cannot shift the run it observes.

The reweighting is exercised in both variants regardless of the warmup counter: the question is what the weights would do to learning, not whether this particular call was past the warmup.

## 2. Criteria, fixed before running

Pooled over the recorded loss calls, on loss-masked cells only:

- **W1 the weights differ.** mean |w_A − w_B| > 1e-3 and max |w_A − w_B| > 1e-2. Below that the channel is numerically inert whatever the shares say.
- **W2 the change is targeted.** mean |w_A − w_B| on transitions where the forecast signal is non-zero is at least **2×** the mean on transitions where it is zero. The mean-preserving normalisation rescales every cell slightly when a channel is removed, so some change off the firing cells is expected; a mechanism that acts *where the forecast mattered* must still change those cells more.
- **W3 the gradient differs.** cosine(∇_A, ∇_B) < 0.9999, with the relative L2 difference reported. A change in weights that leaves the gradient direction untouched has not reached learning.
- **W4 reported, not gated.** The weight change split by the sign of the advantage: the fraction of negative-advantage transitions damped (w_A < w_B) and by how much, the same for positive ones, and the largest-changed transitions with their advantage sign and firing flag. Damping negative advantages weakens the signal to stop doing something; that is recorded as a property of the mechanism, and no benefit or harm is claimed from it alone.

Verdict SWITCH_AB_PASS iff W1, W2 and W3 hold. A failure is STOP_SWITCH_AB: the effect experiment does not start, and the finding goes back for a ruling. Nothing is tuned to obtain a pass; in particular the anomaly gate, ρ_min, the shrink strength, the warmup and the signal's scale are untouched.

## 3. Run

The G5-err configuration (`g5_err`, `perturb_tier: shrink75`, EU-CRD on `candidate_carbon_regret`), 16 000 steps, recording the comparison at every loss call. The state at which the comparison is read is therefore a lightly trained one, and the reading says so: if the ruling wants the comparison at the fully trained state, the G5-err checkpoint is on disk and the same diagnostic can be re-run against it. The run's own training result is discarded — the extra loss evaluations pollute its logged metrics, and nothing but the dumped comparison is read from it.

## 4. What a pass licenses

Only this: the forecast responsibility changes the weights and the policy gradient, and it changes them more where the forecast actually changed a decision. It says nothing about whether that change helps, which is step 3's question and needs the matched pair.
