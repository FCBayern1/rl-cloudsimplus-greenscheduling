# Gradient decomposition reading: the policy gradient moves, the total gradient hides it (2026-09-08, 20:5x)

Diagnostic under reports/EUCRD_SWITCH_AB_PREREG.md Addendum A (86ef8070, frozen before this run). It does not re-judge run 4's W3 and chooses no threshold. State: the G5-err final global module restored in full (145 base tensors + 40 Q-ensemble tensors = 185, strict, `fcb48574`), learning rate zero (no optimisation update), one sampling iteration, 103 loss calls, the first excluded as the estimators' own warm-up. Same-advantages null: **exactly zero** on every call, for the total gradient and for the surrogate gradient alike.

## 1. The number that answers the ruling's question

| gradient compared, A vs B | cosine (min / mean) | relative L2 (median / max) | norm, variant A (median) |
|---|---|---|---|
| **policy surrogate term only** | **0.9393 / 0.9677** | **0.317 / 0.463** | 0.0645 |
| value term | — | — | 18.47 |
| entropy + KL remainder | — | — | 0.0032 |
| whole PPO loss (what W3 measured) | 1.0000 / 1.0000 | 0.00104 / 0.0039 | 19.3 (≈ value) |

The forecast responsibility changes the **policy** gradient by 32 % in relative L2 at the median and turns its direction by a cosine of 0.968 on average, 0.939 at the worst call. The **total** gradient moves by 0.1 %, because the value term's gradient norm is 286× the surrogate's (18.47 vs 0.0645): the surrogate is 0.33 % of the total norm, and the dilution ratio between the two readings is **305×** at the median. Run 4's W3 was measuring the value gradient's inertia, not the policy's response.

This scene's global module trains the critic on a separate trunk (`critic_separate_trunk: true`), so the surrogate gradient here is the gradient on the actor's own parameters, shared layers included, and a cosine over the concatenation of actor and critic parameters says almost nothing about the actor's update.

## 2. Consistency with the earlier runs

Weights: mean |Δw| 0.0153, max 0.475, firing 0.253 vs non-firing 0.0078, targeting ratio **32.5** — the fourth consistent reading (33.5 / 33.4 / 32.4 / 32.5), now at the fully trained state. Advantage sign: 2.1 % of negative-advantage transitions damped (ratio 0.9981), 1.3 % of positive (1.0018); the most-changed cells are all firing cells, 56 % with negative advantage. Total-gradient figures reproduce run 4 (rel L2 1.0e-3 vs 6.4e-4; norm ratio 1.0000028).

## 3. What this establishes and what it does not

Established: at the trained G5-err state, with an exact null, the forecast responsibility changes the actor's policy gradient substantially in both magnitude and direction, and does so through weights concentrated on the transitions where the forecast changed a decision. Run 4's small total-gradient change is explained by dilution, measured, not assumed; the earlier "3 % of transitions" explanation is withdrawn as the cause.

Not established: anything about benefit. A 32 % change in the policy gradient is a change, not an improvement; whether damping 2 % of the corrective signals costs anything, and whether the redirected update lowers carbon, is the effect experiment's question.

Scope, as registered: the reweighting once enabled (the warmup bypassed inside the diagnostic); the learner's running estimators started fresh; the local policy was not restored.

## 4. For a ruling

Run 4's verdict stands as recorded on the object it measured. The ruling anticipated this branch: if the policy gradient changes clearly and was swamped by the value gradient, the wiring check should be re-registered on the right object. That re-registration — the surrogate gradient, its own null, and a margin fixed before any further run — is the ruling's to make, not this reading's. Nothing in the mechanism was touched.
