# Matched pair V_err vs E_err: NO_BENEFIT, and both lines end well behind the rule they started from (2026-09-10, 06:03)

Preregistration reports/EUCRD_MATCHED_PAIR_PREREG.md with Addenda A–E, all frozen before the runs. Run of record: **Isambard-AI Phase 2**, decided mechanically on the maintenance deadline, six exit codes and checkpoint integrity, with no result figure participating (`reports/manifests/matched_pair/PLATFORM_DECISION.md`). Six lines, 120 000 steps each, paired seeds 20260911/12/13, last checkpoint, config sha `8668f98342e6ff21`. Deployment: deterministic decode, six development windows, seven tiers evaluated (the eighth failed, §5), plus the zero-parameter `cover_argmax` rule. The 2020 confirmation windows stay sealed.

## 1. The registered primary criterion

**NO_BENEFIT.** Paired difference in pooled absolute carbon at shrink 0.75 (E − V, relative to V):

| seed | V (kg/window) | E (kg/window) | E − V |
|---|---|---|---|
| 20260911 | 0.002615 | 0.002395 | **−8.4 %** |
| 20260912 | 0.002286 | 0.002530 | **+10.7 %** |
| 20260913 | 0.002396 | 0.002410 | **+0.6 %** |
| mean | 0.002432 | 0.002445 | +1.0 % |

The signs disagree, so the criterion (all three negative **and** pooled ≥ 3 %) is not met, and the mean is on the wrong side anyway. The clean guard passes on its own terms — E's clean carbon is 2.1 % above V's, inside the 3 % tolerance — but it passes because both lines are close to each other, not because either is good.

Contracts are intact everywhere: completion 100 %, completion by MI 100 %, on-time share 1.0, forced deadlines 0, in every cell of every arm and tier. No cell is missing from the tiers that ran, and the judge raised no flags. Nothing here is a carbon advantage bought with service quality.

## 2. The result that matters more: training moved the policy away from a better rule

Mean carbon per window (kg), against the zero-parameter `cover_argmax` rule the lines started from as a fixed prior:

| tier | rule | V (mean) | E (mean) | V vs rule | E vs rule |
|---|---|---|---|---|---|
| godeye (clean) | 0.001606 | 0.001698 | 0.001730 | +5.7 % | +7.7 % |
| **shrink 0.75** (trained on) | 0.001919 | 0.002432 | 0.002445 | **+26.7 %** | **+27.4 %** |
| shrink 0.50 | 0.002782 | 0.003299 | 0.003437 | +18.6 % | +23.6 % |
| shrink 0.25 | 0.003618 | 0.004528 | 0.004542 | +25.1 % | +25.5 % |
| shrink 0 (flat) | 0.005092 | 0.004893 | 0.004999 | **−3.9 %** | **−1.8 %** |
| shuffle | 0.004276 | 0.004526 | 0.004520 | +5.9 % | +5.7 % |
| anti | 0.004841 | 0.005432 | 0.005615 | +12.2 % | +16.0 % |

At the tier both lines trained on, 120 000 PPO steps left them **about a quarter worse than the rule they were initialised to reproduce**. The only tier where learning helps is shrink 0, the completely uninformative forecast, where the rule follows a flat curve and the policies do slightly better by not following it.

## 3. What the policies actually do (question 3 of Addendum E)

Over the 210 real decisions at shrink 0.75 (padding slots excluded; action 0 is a legal choice and cannot stand in for "no job"):

| arm | site distribution | mean κ |
|---|---|---|
| `cover_argmax` | 151 / 47 / 12 | 35.1 |
| V_s20260911 / 12 / 13 | 52/125/33, 56/114/40, 76/102/32 | 46.1, 45.8, **17.2** |
| E_s20260911 / 12 / 13 | 88/75/47, 66/102/42, 65/102/43 | 22.0, **13.1**, **50.2** |

Two things stand out. The trained policies **agree with the rule on 0.5–1.9 % of decisions**, put far more work on site 1 than the rule does, and differ from it by 18–24 steps of dispatch offset on average: they did not stay near their prior, they left it. And the three seeds of a single line disagree with each other as much as the lines disagree — mean κ of 13, 22 and 50 within E, and 17, 46 and 46 within V — so the waiting behaviour is not converging to anything stable. Between the matched V and E of the same seed, identical choices are 0–3.8 % and mean |Δκ| is 28–34.

## 4. Reading, under the rules fixed in Addendum D and E

The registered outcome is "the two lines are nearly identical" on the carbon axis and "neither is better than the initial rule". Addendum E's wording limit applies directly: **this round's training gain is not merely limited, it is negative at the trained-on tier**, and nothing here may be written as learning finding better scheduling. EU-CRD neither helped nor hurt beyond the seed spread; the pair's question is answered, in the negative, for this configuration and budget.

The diagnostic order fixed in Addendum D now applies, and question 1 (verify the evaluation, contracts and checkpoints) is already answered: contracts are perfect, checkpoints are the registered last ones with complete weights, and the same evaluator reproduces the rule's own numbers. Question 2 is answered above: the policies changed their actions substantially and for the worse. What follows is question 3's territory — the same-structure ablation separating the forecast responsibility from reweighting as such and from the extra network training — but the more pressing question this result raises is prior: **why does 120 k steps of PPO on top of a good prior degrade it**, when the RL_V2 smoke at 56 k steps preserved it (clean capture 1.002 against the rule's 1.010)?

That contrast is now the first thing to explain, and it is a fact of record, not a hypothesis: at 56 000 steps the trained vanilla policy sat within 1 % of the rule; at 120 000 steps it sits 5.7 % above it clean and 26.7 % above it at shrink 0.75.

## 5. What failed and is not hidden

`calibrated_shrink_v1`, the secondary deployment-flavoured tier, produced **no cells at all**: all 42 evaluations raised `ValueError: The element returned by reset() has agent_ids that are not the names of the agents in the env`. It is an evaluation-harness fault on that tier's config block, not a result, and the tier is therefore absent from every table above. It is reported here rather than quietly dropped, and it does not affect the primary criterion, which is registered on shrink 0.75.

## 6. Cost

Training: E cost 2 972–3 225 s per 8 000 steps against V's 1 910–1 931 s on the platform of record (1.6×), and 1 588 s against 819 s on the workstation pair (1.94×). Deployment: both lines share one backbone and are indistinguishable at inference. The overhead bought no carbon reduction in this round.
