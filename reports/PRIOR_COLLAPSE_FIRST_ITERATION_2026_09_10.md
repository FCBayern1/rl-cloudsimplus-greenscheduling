# Prior collapse within the first PPO iteration — diagnostic record

Status: diagnostic record, not a new verdict.  The matched-pair results and the
original prior-preservation gate remain unchanged.

## What is established

The true pre-update checkpoint (`drl-manager/logs/prior_gate/checkpoint_init`) is
deterministic and exactly reproduces `cover_argmax` on the audited decision corpus:

| checkpoint | tier | carbon | vs rule | action agreement | site agreement | offset agreement |
|---|---|---:|---:|---:|---:|---:|
| init | godeye | 0.001606 | 0.0% | 100.0% | 100.0% | 100.0% |
| init | shrink75 | 0.001919 | 0.0% | 100.0% | 100.0% | 100.0% |

The first saved post-training checkpoint is at 8,000 environment steps.  Its
registered reading is already far from the rule:

| checkpoint | tier | carbon | vs rule | action agreement | site agreement | offset agreement |
|---|---|---:|---:|---:|---:|---:|
| 8,000 | godeye | 0.001653 | +2.9% | 11.0% | 62.9% | 13.3% |
| 8,000 | shrink75 | 0.002301 | +19.9% | 1.9% | 46.2% | 3.8% |

Thus the large deviation is present by the first saved iteration, not only after
120,000 steps.  The later KL growth describes post-collapse drift as well as the
initial collapse; it must not be used as evidence that the collapse itself was
slow.

## Frozen probe

The first-epoch probe is frozen in commit `82cd0d9d`:

- same prior-gate environment, model, and seed `20260914`;
- 8,000 environment steps;
- `num_sgd_iter=1` instead of 5;
- no reward, learning-rate, clipping, prior, or EU-CRD changes;
- init checkpoint and one post-update checkpoint retained.

The purpose is to distinguish whether roughly four minibatch updates (one PPO
epoch with batch 8,000 and minibatch 2,048) are already sufficient to destroy the
prior, or whether the destruction requires the roughly twenty updates in the
five-epoch iteration.  The probe is not part of the registered evidence chain.

## A third mechanism the probe cannot separate

If the one-epoch probe also collapses, that result alone does not distinguish:

1. a single update whose magnitude is too large;
2. four ordinary-sized updates accumulating too much drift; and
3. an update direction that is mostly noise because the critic/advantages are not
   yet reliable.

The existing first-iteration diagnostics show `vf_explained_var` increasing from
0.13 to 0.40 to 0.64 across the first three learner calls, with `vf_loss` falling
from 0.97 to 0.76 to 0.50.  The first call therefore has weak value explanation,
which is compatible with noisy advantages.  This is evidence for a live
hypothesis, not proof of causality.  The global value coefficient is 10 and the
global gradient norm cap is 20; these facts make critic-to-actor scale coupling a
candidate mechanism, but do not identify the direction by themselves.

If the one-epoch probe fails, the smallest follow-up is a separately frozen
diagnostic with actor learning rate zero during the first iteration while the
critic is trained.  Preservation of the prior in that branch would support
"noisy advantage drives the actor"; prior loss despite a frozen actor would point
to an implementation or evaluation path problem.  That follow-up must be
registered before its result is read.

## Reading rule

The same reader must be used for init and every post-update checkpoint: absolute
carbon, relative gap to `cover_argmax`, exact action agreement, site-axis agreement,
and offset-axis agreement, both per window and pooled.  No checkpoint is selected
by carbon and no hyperparameter is changed after reading the probe.

## Probe run invalidation

The first launch was caught before reading any outcome: its resolved action
space was `5 x 9 = 45`, because the default dyadic offset grid was active.
The frozen prior-gate run uses the certified dense `0..72` grid, `5 x 73 =
365`. The run is archived as invalid at
`reports/manifests/policy_degradation/prior_collapse_probe_sgd1/INVALIDATED.md`.
It supplies no evidence about one-epoch collapse. A replacement launch must
set `OFFSET_GRID_DENSE=1` and record the resolved action-space cardinality
before training.

## Corrected one-epoch probe

The replacement run used the dense grid and resolved to 365 actions per batch
slot and `(128, 365)` candidate observations. It completed 8,000 environment
steps with `num_sgd_iter=1` (four minibatch updates, rather than the frozen
five-epoch iteration's roughly twenty). On a held decision corpus, deterministic
decoding changed from exact prior agreement to 2/35 (5.7%) agreement; the mean
absolute offset error was 16.34 steps and the site histogram moved from
`[25, 8, 2, 0, 0]` to `[7, 23, 5, 0, 0]`. The recurrent and non-recurrent
diagnostic paths agreed on this result.

The training-side metrics were `vf_explained_var=0.02296`, `vf_loss=0.8106`,
`mean_kl=0.000949`, and `grad_norm=10.09` for the global module. Thus the
prior is already destroyed within one PPO epoch, while the reported adjacent
policy KL remains small. This rules out the claim that roughly twenty SGD
updates are necessary. It does not yet distinguish one unusually harmful update
from four smaller harmful updates, nor does it prove that low critic quality is
the causal direction; an actor-frozen first-iteration diagnostic is the next
minimal discriminator.

## Why small KL can flip the argmax

The existing 105-state shrink75 dump was also audited without rerunning the
environment. The cover gap between the best and second-best legal candidate has
quantiles `p50=0`, `p75=0.00017345`, `p90=0.00140023`, `p99=0.02063159`, and
maximum `0.02083337`. After the prior gain of 20, these correspond to logit
gaps of `0`, `0.00347`, `0.0280`, `0.4126`, and `0.4167`. There are 78 exact
ties (74.3%) and 88/105 gaps below `0.001` (83.8%).

This makes the apparently paradoxical pair `mean KL=0.000949` and roughly 94%
argmax disagreement mechanically compatible: a small residual can cross a
tie-breaking boundary without materially changing the probability distribution.
The dump alone gives a conservative bound, not an exact attribution of every
changed action: at most 78 changes can be tie-only, so at least 21 of 105
changes must cross a nonzero gap if the 94% count is paired to this same corpus.
The carbon consequence of tie flips is not inferred here and requires paired
action/cost analysis.

---

## Appended 2026-09-10 (Claude): the carbon of the changed actions, and the frozen-actor diagnostic

### A. The tie flips are not free

The paired action/cost analysis the section above asks for, computed on the 105 matched
decisions at shrink 0.75. Each decision is scored on the **same state and the same legal
candidate set** under the frozen cost model `E · (brown·(1−cover) + green·cover)`, comparing the
candidate the one-iteration policy chose with the candidate the rule chose. Cover, mask, PEs and
MI come from the rule arm's observation dump; the rule arm's trajectory is the initialised
policy's trajectory, since the two are identical action for action.

| | n | pooled local carbon regret |
|---|---|---|
| actions unchanged | 3 | 0 |
| **exact-tie flips** (top-1/top-2 cover gap = 0) | **77** | **+0.001086 kg** |
| flips crossing a non-zero gap | 25 | +0.000886 kg |
| **total** | 105 | **+0.001971 kg** |

Largest single loss +0.000192 kg; **largest single improvement 0.000000 kg — no changed action
improved carbon**.

Two things follow. First, **equal coverage does not mean equal carbon**: the tied candidates
differ in the job's energy and in the site's carbon factor, so flipping a tie costs real carbon —
55 % of the total loss comes from decisions the earlier analysis had to treat as possibly free.
Second, the loss is **one-directional**. A noise-driven random walk would improve about half the
flips; zero improvements over 102 changes says the rule's choice is the optimum of the cost model
these decisions are scored under, so any departure is a loss or a tie, never a gain.

### B. Frozen-actor diagnostic: the prior survives when only the critic learns

One iteration (8 000 steps) of the same configuration, same seed 20260914, same reward, same
prior, with `FREEZE_ACTOR_ITERS` zeroing the gradient of every non-critic parameter
(`src/learners/normalized_critic_loss.py`, off unless the environment variable is set; the run
logged "call 1: zeroed 134 gradients, kept 73 critic ones"). Verified afterwards from the
checkpoints: **the 74 actor tensors changed by exactly 0.000e+00**, while the 71 critic tensors
moved by up to 3.883e-02 — the critic learned, the actor did not move.

| arm | tier | carbon | vs rule | agreement | site | offset |
|---|---|---|---|---|---|---|
| init (no update) | godeye | 0.001606 | −0.0 % | 100.0 % | 100.0 % | 100.0 % |
| init (no update) | shrink75 | 0.001919 | +0.0 % | 100.0 % | 100.0 % | 100.0 % |
| **actor frozen, 1 iteration** | godeye | 0.001606 | −0.0 % | **100.0 %** | 100.0 % | 100.0 % |
| **actor frozen, 1 iteration** | shrink75 | 0.001919 | +0.0 % | **100.0 %** | 100.0 % | 100.0 % |
| normal training, 1 iteration | godeye | 0.001653 | +2.9 % | 11.0 % | 62.9 % | 13.3 % |
| normal training, 1 iteration | shrink75 | 0.002301 | +19.9 % | 1.9 % | 46.2 % | 3.8 % |

Read as registered: **the prior is preserved**, so the first-iteration damage is driven by the
actor following advantages produced by a critic that has not yet learned (`vf_explained_var`
0.023 at that point), not by a defect in the PPO update, the state wiring or the parameter path.
The comparison is clean: environment, seed, reward and critic training are identical between the
frozen and the normal run, and the only difference is whether the actor receives gradient.

### C. What this does and does not license

Established: with noisy advantages withheld from the actor, one iteration leaves the rule intact
to the last decision; with them applied, 94 % of decisions flip and every flip costs carbon. The
mechanism is the combination of **noise-quality advantages** and a **prior with no decision
margin** (50 % exact ties, 96 % of logit margins below 0.1 against a residual that reaches ‖W‖
≈ 0.3 in one epoch).

Not established: that a critic warm-up alone would fix it. Warm-up improves advantage quality but
cannot change the margin geometry; once the critic is good, a residual of the same scale still
overwhelms a 0.02-logit margin. The two repairs are complementary, which is why
`reports/GATED_RESIDUAL_PREREG_DRAFT.md` §7 orders them warm-up first, bounded gated residual
second, and does not treat either as sufficient alone. The archived `NO_BENEFIT` verdict of the
matched pair is unchanged by any of this.
