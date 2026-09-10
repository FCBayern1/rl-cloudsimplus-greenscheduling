# Gated residual over a fixed prior: draft preregistration (2026-09-10)

**Draft, not frozen.** It proposes the minimal mechanism change the degradation investigation points at, and the gate that would license it. Nothing is run under this document until it is frozen, and the archived `NO_BENEFIT` verdict of the matched pair is not reinterpreted by it.

## 1. What the evidence says the fix has to do

Measured, not assumed (reports/RL_POLICY_DEGRADATION_INVESTIGATION_2026_09_10.md, reports/PRIOR_COLLAPSE_FIRST_ITERATION_2026_09_10.md):

- The initialised policy **is** the rule: identical carbon and 100 % action agreement with `cover_argmax` on 210 decisions at both godeye and shrink 0.75.
- **One PPO epoch (about four minibatch updates) destroys it**: agreement falls to ≈ 5.7 %, shrink 0.75 carbon rises ≈ 20 %.
- The updates are individually tiny — adjacent-update KL ≈ 0.00095 — yet 94 % of greedy actions change, because **the prior has almost no decision margin**: at real states the top-1/top-2 cover gap is exactly zero in ~50 % of decisions, below 0.001 in 84 %, and under the gain of 20 the logit margin is below 0.1 in 96 % of them. A learned residual reaches ‖W‖ ≈ 0.3 after one epoch — one to two orders of magnitude above the margin it has to respect.
- The changes are **not** a wash: over 105 matched decisions the local carbon regret of the policy's choices against the rule's is **+0.00197 kg in total with not a single improvement**, and 55 % of that comes from decisions where the cover was exactly tied (equal coverage does not mean equal carbon: the candidates differ in energy and in site factor).

Consequence: **shrinking the step size cannot be the fix.** To respect a 0.02-logit margin the updates would have to be smaller than the noise they are made of. The repair has to change *what the update is allowed to move*, not how far it moves per step.

## 2. The mechanism

Replace the unconstrained sum with an explicitly gated residual on the offset logits:

    logit(d, κ) = gain · cover(d, κ) + g · residual(d, κ)

- `gain` stays a fixed buffer (20.0), exactly as now.
- `residual` is the existing learned path (`dc_encoder`, `ctx_to_dc`, `offset_head`), still zero-initialised.
- `g` is a single **learned scalar initialised at 0**, passed through `softplus` so it cannot go negative, and carrying an explicit cost `λ_g · g²` in the loss.

At initialisation `g = 0`, so the policy is the rule exactly — the property the investigation showed we currently have and then lose. The residual can only take effect by paying `λ_g · g²`, so it must earn its deviation. Everything else — PPO, the reward, the learning rate, the entropy coefficient, the prior gain, the EU-CRD mathematics — is untouched.

Two knobs, both fixed before any run: `λ_g` and whether `g` is global or per-slot. The draft proposes a **single global scalar** and `λ_g` chosen so that a residual large enough to flip a median-margin decision costs about as much reward as that flip typically gains; the number is computed from the margin distribution above and frozen before the first run, not tuned afterwards.

**Why a gate rather than KL-to-prior.** KL measures distributional distance, and this failure is not distributional: a KL of 0.00095 already flips 94 % of the greedy actions. A constraint expressed in KL is therefore blunt exactly where this problem is sharp. The gate acts on the quantity that matters — how much the learned part is allowed to contribute to the decision.

## 3. Gate on the mechanism (what would license it)

One vanilla line, one seed, 24 000 steps, the same environment, reward and seed as the prior-gate run, evaluated by the frozen evaluator (deterministic decode, six development windows, godeye and shrink 0.75):

- **G-a prior preserved**: carbon within **3 %** of `cover_argmax` at both tiers, contracts intact.
- **G-b the gate is honest**: `g` is reported per iteration; a run that keeps the prior only because `g` never leaves 0 is reported as "no learning", not as a success.
- **G-c the deviations pay**: on the decisions where the policy differs from the rule, the local carbon regret (the §1 measure) must be **≤ 0**, i.e. deviations must not be systematically harmful as they are today.

Failing G-a stops the line. Passing G-a with `g ≈ 0` (G-b) means the mechanism protects the prior but the scene offers nothing to learn — a result to report, not to tune away.

## 4. What comes after, in order

1. Gate above (one seed, 24 000 steps).
2. If it passes with a non-trivial `g`: one vanilla line at the full 120 000 steps, asking whether the policy can **beat** the rule rather than merely match it.
3. Only then the matched EU-CRD pair, under its already-frozen protocol (shrink 0.75, three paired seeds, all three negative and ≥ 3 % pooled, clean guard 3 %).

**A design consequence worth recording now**: in this framework the forecast responsibility has a natural role — it can modulate the gate (open where the forecast is trustworthy, closed where the local decision regret says it is not) instead of reweighting advantages after the fact. That is a hypothesis for a later registration, not part of this one.

## 5. What this draft does not do

It does not revisit the `NO_BENEFIT` verdict, does not touch the reward or the learning rate, does not change the prior gain, and does not run anything. If the frozen-actor diagnostic shows the first-iteration damage is driven by noisy advantages from an untrained critic, that finding is complementary: the gate protects the prior regardless of what drives the first updates, and a critic warm-up would be a separate, additional registration.
