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

## 3a. Addendum A — gradient decomposition, diagnostic only (2026-09-08, after run 4 was read)

Run 4's W3 compared the gradient of the **whole** PPO loss: policy surrogate, value term, entropy and KL. EU-CRD's reweighting acts on the surrogate alone, so a small change in the total gradient can be the surrogate's change diluted by terms it does not touch. That was missed in the registration and in every review of it until after run 4. The explanation offered in the run-4 reading ("3 % of transitions fire, so the update barely moves") is therefore demoted to a hypothesis: transitions do not contribute equally to a gradient, and a sample fraction does not settle it.

One bounded decomposition is added, **as a diagnostic**. It does not re-judge run 4's W3, and no new threshold is chosen from it.

- On a fixed batch and a fixed model state, the gradient is split by the linearity of the loss: the surrogate term alone (value coefficient, entropy coefficient and KL coefficient set to zero *inside the diagnostic only*, the critic itself untouched and the run's coefficients restored afterwards), the value term, and the entropy-plus-KL remainder. The surrogate gradient is taken with respect to every parameter it reaches, shared trunk included; nothing is truncated to a final layer.
- Reported for A vs B on the surrogate gradient: cosine (clamped), relative L2, and the same null as before; plus the norms of each term so the dilution, if any, is visible.
- State: the G5-err final checkpoint's global module is loaded through the existing strict warm-start path, the learning rate is set to zero so no optimisation update happens, and one sampling iteration of diagnostic batches is recorded. The local policy and the learner's running estimators are not part of that checkpoint and start fresh, which is stated rather than hidden: the first call of the diagnostic sits in the estimators' own warm-up and is excluded.

What the decomposition can say: if the surrogate gradient also barely changes, the mechanism's small effect is real and the anomaly gate, guardrail and reweighting deserve study; if the surrogate gradient changes clearly and was swamped by the value gradient in the total, the wiring check was measuring the wrong object and must be re-registered on the right one — before any further run, not fitted to this one.

## 3b. Addendum B — the wiring check is re-registered on the policy surrogate gradient (2026-09-08, ruling after the decomposition)

**Disclosure.** This revision was made after the decomposition of Addendum A was read. It is a development revision, not a pre-specified one: the measured object changes, and nothing else does. W3's cosine threshold (0.9999, clamped to [−1, 1]) and its aggregation rule (the maximum over the recorded calls must be below the threshold) are carried over unchanged so that no number is chosen from the reading. Run 4's verdict stands as recorded on the object it measured; the decomposition is recorded independently.

**W3′ the surrogate gradient differs.** The gradient compared is that of the policy surrogate term as training computes it — the same loss code, the same loss mask, the same PPO clipping, the same parameter path (every parameter the surrogate reaches, shared layers included) — obtained from the training loss with the value, entropy and KL coefficients set to zero inside the diagnostic only. Its null is the same object scored twice on the same advantages and must be exactly zero on every call; a call with an undefined cosine (a zero gradient in either variant) is counted and excluded. Verdict SWITCH_AB_PASS iff W1, W2 and W3′ hold; the total-gradient cosine is still reported, as a diagnostic.

**Re-check protocol, light.** No training round: the frozen G5-err global module (185 tensors, strict), fixed sampled batches, learning rate zero, one sampling iteration, a seed different from the decomposition run's. The existing decomposition data serve as an engineering check of the same judge (surrogate path identical to training's, restore complete, null zero) and are reported as such; they are not a second independent confirmation.

**Wording rules for any reading under this document.** "The forecast responsibility changes the gradient of the policy surrogate loss" may be written. "The policy update changed by X %" may not: gradient clipping and the optimiser state sit between the gradient and the update. No carbon claim of any kind.

**Implementation fact, recorded.** The global module clips gradients by global norm over all of its parameters (`max_grad_norm` 20 → `grad_clip`, RLlib default `grad_clip_by="global_norm"`), critic included even though the critic has a separate trunk. With the value term's gradient norm near 18.5 and the total near 19.3, the critic decides whether clipping engages and so can scale the actor's update indirectly. Both lines of the matched pair must carry identical clipping settings, asserted by their generator.

**If W3′ passes**, the matched effect experiment is prepared: shrink 0.75 training, 120 000 steps, 3 paired seeds, V_err against E_err, the primary comparison, clean tolerance and contract rules completed before launch, the last checkpoint fixed as the checkpoint of record, the 2020 confirmation windows sealed.

## 4. What a pass licenses

Only this: the forecast responsibility changes the weights and the policy gradient, and it changes them more where the forecast actually changed a decision. It says nothing about whether that change helps, which is step 3's question and needs the matched pair.
