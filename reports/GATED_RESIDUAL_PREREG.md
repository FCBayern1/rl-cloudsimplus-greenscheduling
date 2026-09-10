# Anchored gated residual over a fixed prior (frozen 2026-09-10)

Frozen by the commit that adds this file, before any implementation and before any run. It supersedes `GATED_RESIDUAL_PREREG_DRAFT.md` (v1, v2), whose defects are recorded in §7. It does not reinterpret the archived `NO_BENEFIT` verdict of the matched pair, and it changes neither the reward, the learning rate, the entropy coefficient, the prior gain, nor the EU-CRD mathematics.

## 1. The problem the mechanism must solve

Measured on this scene (`RL_POLICY_DEGRADATION_INVESTIGATION_2026_09_10.md`, `PRIOR_COLLAPSE_FIRST_ITERATION_2026_09_10.md`):

- the initialised policy **is** `cover_argmax` — identical carbon, 100 % action agreement on 210 decisions at godeye and shrink 0.75;
- **one PPO epoch (~4 minibatch updates) takes agreement to 5.7 %** and shrink 0.75 carbon to +19.9 %;
- the prior has **no decision margin**: the top-1/top-2 cover gap is exactly zero in ~50 % of decisions and below 0.001 in 84 %, so 96 % of prior logit margins are under 0.1 at gain 20, while the learned residual reaches ‖W‖ ≈ 0.3 within one epoch;
- the flips are one-directional: 102 changed actions, **zero improvements**, +0.00197 kg of local carbon regret, **55 % of it from exact-tie flips**;
- with the actor frozen for a full iteration and only the critic learning, the prior is **100 % intact**, so the driver is the actor following advantages from a critic at `vf_explained_var` 0.023.

Two consequences fix the design. Shrinking the step size cannot work — safety would need updates smaller than the noise they are made of. And **a quadratic penalty on the perturbation cannot protect the exact ties**: where the gap is 0, an arbitrarily small residual flips the action at an arbitrarily small cost, and those are precisely the decisions that carry 55 % of the measured loss.

## 2. The mechanism

    a₀(s)      = cover_argmax(s)                       # the rule's action, from the observation
    e(a)       = g · tanh( residual(a) )
    logit(a)   = 20 · cover(a) + 0.1 · 1[a = a₀] + e(a)

- **Anchor margin 0.1** on the rule's own action. It does not change the initial deterministic action (a₀ is already the argmax of `20·cover` under the index tie rule, so adding to it keeps it the argmax), and it lifts every decision — including the exactly tied ones — off the knife edge. The value comes from the measured distribution: 96 % of prior logit margins are below 0.1.
- **`residual`** is the existing learned path (`dc_encoder`, `ctx_to_dc`, `offset_head`) with its output layer exactly zero-initialised. `tanh(0) = 0`, so **initial equality to the rule comes from the zero residual, not from a zero gate**.
- **`g = g_max · sigmoid(θ)`**, one learned scalar, `g_max = 0.5`, initialised at **g₀ = 0.01** — small but strictly non-zero, so the residual receives gradient from the first step. At g₀ the largest achievable difference between two candidates is 2·g₀ = 0.02, **below the 0.1 anchor**: the gate must genuinely open before any action can change. At `g_max` the reachable difference is 1.0, so a fully open gate can overturn any decision in the record.

## 3. The regulariser

Penalise the perturbation that can actually overturn the rule, not the gate and not the raw residual:

    d(s) = max over legal a ≠ a₀ of  [ e(a) − e(a₀) ]₊
    L_gate = λ · mean over decision states of d(s)²        λ = 100

- The maximum is taken over **legal** candidates only; `mean` is over **slots carrying a real job**, padding excluded, consistent with every other measurement in this line of work.
- λ = 100 is fixed by the anchor: 100 × 0.1² = 1, so a residual that has just accumulated enough to overturn the rule pays about one unit of PPO's own (advantage-normalised) loss scale. The calibration is deliberately **not** derived from the raw reward scale (PPO normalises advantages, so `20000 × carbon` cannot be used to size a loss term) and **not** from a margin quantile (the median margin is exactly 0, which would send λ to infinity).
- Because it is not an average over the 365 candidates, the penalty is not diluted by the action-space size.

**Averaging semantics, stated to prevent a later misreading**: `mean` runs over decision states, so "λ = 100 makes overturning cost 1" holds when *every* decision is overturned. Overturning 10 % of decisions costs 0.10; overturning one decision in 105 costs 0.0095. The quantity being priced is *how often the policy departs from the rule*, which is the intended semantics.

## 4. Critic warm-up

**Fixed at 3 full iterations (24 000 steps) with the actor completely frozen** — the `FREEZE_ACTOR_ITERS` mechanism already implemented and verified (actor tensors change by exactly 0, critic by 3.9e-2, prior 100 % intact). Not adaptive: an "explained variance ≥ 0.5" rule would give each seed a different warm-up budget and would not be reproducible. The measured trajectory on this configuration is 0.023 / 0.40 / 0.64 over the first three iterations.

## 5. The gate on the mechanism

After the warm-up, one vanilla line, one seed, **24 000 further steps** of ordinary training, evaluated by the frozen evaluator (deterministic decode, six development windows, godeye and shrink 0.75, `cover_argmax` as the reference):

- **G-a prior preserved** — carbon within **3 %** of `cover_argmax` at both tiers, contracts intact (completion, on-time, forced deadlines).
- **G-b something was actually learned** — all three must hold, and `g` alone is not evidence:
  1. the residual's output layer receives a **non-zero gradient**;
  2. the effective relative perturbation `d(s)` is **non-zero** on some decisions;
  3. at least **10 decisions** change their greedy action relative to the rule.
- **G-c the deviations pay** — computed at each decision on the **same state and the same legal candidate set** under the frozen cost model `E · (brown·(1−cover) + green·cover)`, reporting the pooled sum, the count of improvements and the count of deteriorations: the pooled local carbon regret must be **< 0** *and* there must be **at least 10 genuine improvements**.

Passing G-a while failing G-b is reported as **"prior preserved, nothing learned"**. Passing G-a and G-b while failing G-c is reported as **"the policy departs from the rule and it costs carbon"**. Neither is a success, and neither licenses a parameter change without its own registration.

KL to the initial policy is reported per iteration as an auxiliary diagnostic of distributional stability. It is **not** a gate: a KL of 0.00095 already flipped 94 % of greedy actions, so it is blunt exactly where this problem is sharp.

## 6. Order of work

1. Implement §2 and §3 behind a config flag, with the tests of §8. Default off; every existing configuration keeps its current behaviour bit-for-bit.
2. Critic warm-up, 24 000 steps, actor frozen (§4).
3. Gate run, 24 000 steps of ordinary training, judged by §5.
4. If it passes: one vanilla line at 120 000 steps, asking whether it can **beat** the rule rather than match it.
5. Only then the matched EU-CRD pair, under its already-frozen protocol.

No EU-CRD ablation is run while there is no learning gain to decompose.

## 7. What the earlier drafts got wrong

- **v1**: `softplus(0) ≈ 0.693`, so "gate initialised at 0" would not have started at the rule; and forcing the gate to exactly zero would have killed the residual's gradient, leaving the branch unable to learn. Penalising `λ·g²` alone was also evadable by scaling the residual up and the gate down at constant logits.
- **v2**: fixed both of those, but kept a **quadratic penalty on the effective perturbation with no anchor**, which leaves the exact ties unprotected — at a zero margin any perturbation flips the action at near-zero cost, and those decisions carry 55 % of the measured carbon loss. The anchor margin of §2 and the relative-perturbation regulariser of §3 are the repair.

All three defects were found in review before anything was run.

## 8. Implementation contract and tests (must pass before the warm-up starts)

1. **Default off**: with the flag disabled, logits are bit-identical to the current model.
2. **Initial equality**: with the flag enabled and the residual zero-initialised, the greedy action equals `cover_argmax` on every legal state, at both tiers, and the anchor does not move it.
3. **Anchor arithmetic**: the margin between the rule's action and the best competitor is ≥ 0.1 at initialisation, including on exactly tied states.
4. **Gate floor**: at g₀ = 0.01 no residual, however large, can change the greedy action (max achievable difference 0.02 < 0.1).
5. **Regulariser**: `d(s)` is zero when the residual is zero; positive only when a non-rule candidate is pushed above the rule's; computed over legal candidates and over real jobs only; `λ·mean(d²)` matches a hand-computed value on a small example.
6. **Train/rollout identity**: the anchored logits, their probabilities, the greedy actions and the log-probs are identical between the sequence path and the single-step rollout path, as already required for the unanchored model.

---

## Addendum A — G-c replaced; the frozen version was unsatisfiable (2026-09-10, before any training result was read)

**The defect.** G-c required the pooled *local* carbon regret against `cover_argmax` to be
negative with at least ten improvements. Under this scene's cost model that cannot happen for any
policy: the five sites share carbon factors (brown 0.5, green 0.01) and the flip concerns one
job, so the local cost `E · (brown·(1−cover) + green·cover)` is strictly decreasing in cover, and
`cover_argmax` **is** its argmin over the legal set. Every other action is therefore equal or
worse by construction. Measured confirmation on the 105 archived decisions: negative local regret
occurs **0 times**, and flips between equal-cover candidates cost **exactly 0.000000000 kg**. The
gate was unsatisfiable by definition, not by any failure of the algorithm
(`reports/GATED_RESIDUAL_GATE_PREFLIGHT_STOP_2026_09_10.md`, `STOP_GATE_DEFINITION_UNSATISFIABLE`).

**Why the local measure was the wrong object.** It is myopic: dispatching now consumes residual
green on the reservation grid and lowers the coverage available to later jobs. A policy can be
locally suboptimal at individual decisions and still finish an episode with less carbon. Judging
a policy against the local formula's own optimum can only ever measure how far it departs from
that formula — never whether the departure was worthwhile.

**G-c′ (replacing G-c).** Judged on **executed** episode carbon, the same quantity the matched
pair and every other reading in this line use:

- the gated line's pooled carbon over the six development windows must be **strictly lower** than
  `cover_argmax`'s on the same windows, at **shrink 0.75** (the trained-on tier);
- the improvement must hold in **at least four of the six windows** — fixed here, before any
  result, so a pooled gain carried by one window is reported as conditional and not as a pass;
- contracts intact in every window (completion, on-time share, forced deadlines), as in G-a;
- the clean tier (godeye) is reported alongside and must not regress by more than the 3 % of G-a.

The local-regret measure of the withdrawn G-c is **retained as a reported diagnostic**, not as a
gate: it says how far the policy departs from the myopic optimum, which remains informative when
read next to the executed carbon.

**Unchanged**: G-a (prior preserved within 3 %), G-b (non-trivial learning: non-zero gradient to
the residual's output layer, non-zero `d(s)`, at least ten changed actions), the mechanism of §2,
the regulariser and λ of §3, the 3-iteration critic warm-up of §4, and the order of work of §6.
The implementation already committed (`b8538934`, seed wiring `f7f7e507`, 47 tests passing, 1
skipped) needs no change; only this gate definition does.

**Audit note.** The earlier claim that equal-cover flips carried 55 % of the measured carbon loss
was an artefact of a misclassification and has been withdrawn and corrected in
`reports/PRIOR_COLLAPSE_FIRST_ITERATION_2026_09_10.md`. The surviving facts are unchanged: 102
changed actions, zero improvements, +0.00197 kg of local regret, all of it from choices with
genuinely lower coverage.

**Two places in this document carry the withdrawn claim** and are corrected by the audit note
above rather than edited, because frozen text is amended only by addenda: §1's bullet "102 changed
actions, zero improvements, +0.00197 kg … **55 % of it from exact-tie flips**" and §7's v2
paragraph "those decisions carry 55 % of the measured carbon loss". The correct attribution is
that equal-cover flips cost **exactly zero** and all of the +0.00197 kg comes from choices with
lower coverage. The design consequence is unaffected: exact ties are still the states where an
arbitrarily small perturbation flips the action at an arbitrarily small penalty, which is what the
0.1 anchor exists to prevent — the anchor's justification was the margin geometry, not the
mis-attributed carbon.
