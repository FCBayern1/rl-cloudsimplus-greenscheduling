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

