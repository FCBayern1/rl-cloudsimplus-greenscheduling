# Anchored gated residual gate — preflight stop (2026-09-10)

Status: **STOP_GATE_DEFINITION_UNSATISFIABLE** before the first training
iteration completed and before any training carbon result was read.  This is a
registration/preflight defect, not evidence for or against the anchored gated
residual mechanism.

## What was implemented and checked

The mechanism frozen in `reports/GATED_RESIDUAL_PREREG.md` at commit
`e82701e8` was implemented behind a default-off configuration flag.  The model,
loss, configuration generator and focused tests were committed in `b8538934`;
the launcher was then corrected to preserve an explicitly frozen
`PYTHONHASHSEED=0` in `f7f7e507`.

The focused implementation suite passed (47 passed, 1 skipped).  The full
Python suite produced 1554 passes, 1 skip and 25 pre-existing failures in
unrelated gateway/checkpoint/config/plot tests; none involved the new mechanism.

One run was launched at 20:35 Europe/London with seed `20260915`, dense offsets,
`PYTHONHASHSEED=0`, and 60 actor-frozen loss calls.  It was manually stopped at
20:39, before an iteration or post-update checkpoint was written.  Only the
initial checkpoint exists.  No run metric or carbon result was used to find the
problem below.

## The impossible registered condition

Gate G-c requires, on the same state and legal candidate set, both:

1. pooled local carbon regret below zero; and
2. at least ten genuine improvements over `cover_argmax`.

It fixes the candidate cost as

`C(a) = E * (brown * (1 - cover(a)) + green * cover(a))`.

The generated run configuration gives every datacenter the same factors:
`brown = 0.5` and `green = 0.01`.  For every real job `E > 0`, hence

`C(a) = 0.5 E - 0.49 E * cover(a)`.

Minimising this cost is therefore exactly equivalent to maximising `cover(a)`.
The registered reference action `a0 = cover_argmax` is consequently a global
minimiser of G-c's local cost over the legal candidate set.  For every legal
action `a`,

`C(a) - C(a0) >= 0`.

It follows mechanically that pooled regret cannot be negative and the number
of genuine local improvements cannot exceed zero.  No model, optimiser, seed,
training duration or gate value can pass G-c as written.

Exact cover ties do not change this proof: under the registered formula and
the homogeneous factors, equal cover gives equal local cost.  This conflicts
with the earlier claim in
`reports/PRIOR_COLLAPSE_FIRST_ITERATION_2026_09_10.md` that exact-cover tie
flips carried non-zero local regret because candidates differed in energy or
site factor.  That earlier calculation or its stated cost semantics therefore
needs a separate audit; it must not be used to justify G-c until reconciled.

## Consequence and admissible next action

The frozen gate is not reinterpreted and no threshold is relaxed.  This launch
is an invalid preflight attempt and supplies no mechanism verdict.

Before another run, an append-only registration decision must choose a
quantity that the residual can in principle improve.  The minimal scientifically
coherent choice is the **executed trajectory carbon** relative to
`cover_argmax`, while retaining G-a as the 3% safety guard and G-b as the
non-trivial-learning check.  If a same-state counterfactual is required instead,
it must include the future system effect of the action; the current one-job
myopic cost cannot certify improvement over its own argmin.

