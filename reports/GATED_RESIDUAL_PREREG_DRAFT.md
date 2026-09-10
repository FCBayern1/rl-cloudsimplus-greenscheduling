# Gated residual over a fixed prior: draft preregistration (2026-09-10)

**Draft, not frozen.** It proposes the minimal mechanism change the degradation investigation points at, and the gate that would license it. Nothing is run under this document until it is frozen, and the archived `NO_BENEFIT` verdict of the matched pair is not reinterpreted by it.

## 1. What the evidence says the fix has to do

Measured, not assumed (reports/RL_POLICY_DEGRADATION_INVESTIGATION_2026_09_10.md, reports/PRIOR_COLLAPSE_FIRST_ITERATION_2026_09_10.md):

- The initialised policy **is** the rule: identical carbon and 100 % action agreement with `cover_argmax` on 210 decisions at both godeye and shrink 0.75.
- **One PPO epoch (about four minibatch updates) destroys it**: agreement falls to ≈ 5.7 %, shrink 0.75 carbon rises ≈ 20 %.
- The updates are individually tiny — adjacent-update KL ≈ 0.00095 — yet 94 % of greedy actions change, because **the prior has almost no decision margin**: at real states the top-1/top-2 cover gap is exactly zero in ~50 % of decisions, below 0.001 in 84 %, and under the gain of 20 the logit margin is below 0.1 in 96 % of them. A learned residual reaches ‖W‖ ≈ 0.3 after one epoch — one to two orders of magnitude above the margin it has to respect.
- The changes are **not** a wash: over 105 matched decisions the local carbon regret of the policy's choices against the rule's is **+0.00197 kg in total with not a single improvement**, and 55 % of that comes from decisions where the cover was exactly tied (equal coverage does not mean equal carbon: the candidates differ in energy and in site factor).

Consequence: **shrinking the step size cannot be the fix.** To respect a 0.02-logit margin the updates would have to be smaller than the noise they are made of. The repair has to change *what the update is allowed to move*, not how far it moves per step.

## 2. The mechanism (v2, after review; v1 had two defects recorded in §6)

    logit(d, κ) = gain · cover(d, κ)  +  g · tanh( residual(d, κ) )

- `gain` stays a fixed buffer at 20.0, exactly as now.
- `residual` is the existing learned path (`dc_encoder`, `ctx_to_dc`, `offset_head`) with its
  **output layer exactly zero-initialised**, as today. `tanh(0) = 0`, so **the initial policy is
  the rule because the residual is zero — not because the gate is zero.**
- `g = g_max · sigmoid(θ)`, a single learned scalar, **bounded in (0, g_max)** and initialised
  small but **strictly non-zero** (θ₀ chosen so g₀ = 0.01, i.e. 2 % of g_max). Non-zero g is
  what lets the residual receive gradient at all; a gate that is exactly zero would multiply
  the residual's gradient by zero and the branch would never start learning.
- `g_max = 0.5`, fixed from the measured margin distribution: the largest top-1/top-2 logit
  margin observed at real states is 0.417, so a fully open gate can overturn any decision in
  the record, and no more.

**Penalty on the effective perturbation, not on the gate**:

    λ · mean[ ( g · tanh(residual) )² ]

penalised over the legal candidates of each decision. Penalising `g²` alone would be
bypassable: the network could scale the residual up 100× and the gate down 100×, leaving the
logits unchanged while the penalty falls by 10⁴. Bounding the residual through `tanh` and
charging the quantity that actually reaches the logits removes that degeneracy — the gate's
value then has a physical meaning (the maximum logit perturbation it permits) and deviation
genuinely has to be paid for.

`λ` is computed before the first run from the margin distribution and the reward scale — set so
that a perturbation large enough to flip a median-margin decision costs about as much reward as
such a flip typically gains — and frozen. It is not tuned afterwards.

Everything else — PPO, the reward, the learning rate, the entropy coefficient, the prior gain,
the EU-CRD mathematics — is untouched. The gate is global in this first version; a per-slot or
state-conditioned gate is a later question, not this one.

**Why not KL-to-prior as the main constraint.** KL measures distributional distance, and this
failure is not distributional: a KL of 0.00095 already flips 94 % of greedy actions, so a
KL-shaped constraint is blunt exactly where the problem is sharp. KL is **not discarded** — it
is kept as an auxiliary diagnostic of distributional stability, reported per iteration
alongside the effective perturbation. The binding constraints are the effective perturbation
and the carbon regret.

## 3. Gate on the mechanism (what would license it)

One vanilla line, one seed, 24 000 steps, the same environment, reward and seed as the prior-gate run, evaluated by the frozen evaluator (deterministic decode, six development windows, godeye and shrink 0.75):

- **G-a prior preserved**: carbon within **3 %** of `cover_argmax` at both tiers, contracts intact.
- **G-b the gate is honest** — three quantities, not one, reported per iteration:
  the **effective perturbation** `mean|g·tanh(residual)|` and its maximum over legal candidates;
  the **number of decisions whose greedy action differs** from the rule;
  and the **gradient norm reaching the residual's output layer** (a non-zero `g` with a residual
  that never moves is not learning either). A run that preserves the prior with all three at
  zero is reported as **"prior preserved, nothing learned"** — not as a success.
- **G-c the deviations pay**, with the aggregation fixed here: at each decision, on the **same
  state and the same legal candidate set**, the local carbon difference between the policy's
  chosen candidate and the rule's is computed under the frozen cost model
  (`E · (brown·(1−cover) + green·cover)`); the reading reports the **pooled sum**, the **count of
  improvements** and the **count of deteriorations**. Passing requires the pooled local regret
  to be **< 0** *and* **at least 10 genuine improvements** over the ~105 decisions, so a run that
  simply never changes an action cannot pass with a pooled zero.

Failing G-a stops the line. Passing G-a with `g ≈ 0` (G-b) means the mechanism protects the prior but the scene offers nothing to learn — a result to report, not to tune away.

## 4. What comes after, in order

1. Gate above (one seed, 24 000 steps).
2. If it passes with a non-trivial `g`: one vanilla line at the full 120 000 steps, asking whether the policy can **beat** the rule rather than merely match it.
3. Only then the matched EU-CRD pair, under its already-frozen protocol (shrink 0.75, three paired seeds, all three negative and ≥ 3 % pooled, clean guard 3 %).

**A design consequence worth recording now**: in this framework the forecast responsibility has a natural role — it can modulate the gate (open where the forecast is trustworthy, closed where the local decision regret says it is not) instead of reweighting advantages after the fact. That is a hypothesis for a later registration, not part of this one.

## 5. What this draft does not do

It does not revisit the `NO_BENEFIT` verdict, does not touch the reward or the learning rate, does not change the prior gain, and does not run anything. If the frozen-actor diagnostic shows the first-iteration damage is driven by noisy advantages from an untrained critic, that finding is complementary: the gate protects the prior regardless of what drives the first updates, and a critic warm-up would be a separate, additional registration.


## 6. Corrections to v1 (review, 2026-09-10)

Two defects in the first draft, both of which would have produced a mechanism that looks like a
gate but is not one:

1. **`softplus(0) ≈ 0.693`, not 0.** v1 said "g initialised at 0, passed through softplus", which
   would have started the policy at a *0.693-weighted* residual rather than at the rule. Forcing
   the gate to exactly zero instead would have been worse: with `residual = 0` and `g = 0` the
   product's gradient vanishes on both factors and the residual branch could never start
   learning. v2 guarantees "initial policy = rule" through the zero-initialised residual and
   keeps `g` small but strictly non-zero.
2. **Scale degeneracy.** Penalising `λ·g²` alone can be evaded by scaling the residual up and the
   gate down at constant logits, shrinking the penalty quadratically for free. v2 bounds the
   residual with `tanh` and penalises the effective perturbation `g·tanh(residual)` that actually
   reaches the logits.

Both were found in review before anything was run.

## 7. Order of work, after the frozen-actor diagnostic

If that diagnostic shows the first-iteration damage is started by noisy advantages from an
untrained critic, the small experiment should combine, in this order:
(1) a brief critic warm-up; (2) the bounded gated residual of §2; (3) the effective-perturbation
regularisation; (4) the 24 000-step single-seed prior-preservation gate of §3; (5) one vanilla
line at 120 000 steps asking whether it can beat the rule; (6) only then the EU-CRD comparison.
