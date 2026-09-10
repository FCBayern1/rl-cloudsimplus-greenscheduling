# Why a policy initialised to a good rule degrades under PPO: investigation (2026-09-10)

**Provenance note.** The narrative of this file was written by the reviewing agent and then **overwritten by me (Claude) at 10:5x UTC** with a parallel draft of my own, before it had been committed. The original prose is lost; everything below is reconstructed from the artefacts that survived (`reports/manifests/policy_degradation/`, `reports/EUCRD_LOCAL_PAIR_TRAINING_REVIEW_2026_09_10.md`, the code changes in the working tree) plus the conclusions as they were restated to me. Numbers are read back from the artefacts, not retyped from memory. The overwrite was my error; the fix is that this file and its artefacts are committed immediately.

---

## 0. What the question turned out to be

The framing "56k was fine, 120k is broken" does **not** survive contact with the record: the 56 000-step reading (RL_V2) and the 120 000-step reading (matched pair) are **not two points on one curve**. They differ in the training tier (godeye vs shrink 0.75), in the seeds and in the platform. The real question is the one the investigation answers: a policy initialised on a strong fixed prior leaves that prior steadily under PPO, and part of the measured gap was an evaluation fault.

## 1. Correctness faults found in the training/evaluation path (reproduced)

1. **Evaluation discarded the GTrXL recurrent memory.** The deployed policy was scored without the state it was trained to carry.
2. **Dropout stayed active at evaluation** — the module was never put in `.eval()`.
3. **Positional encoding differed between the training sequence path and the single-step rollout path.**
4. **GAE actually ran with γ = 0.99 / λ = 0.95 while the configuration asks for 0.999 / 0.98** — the per-module wiring does not deliver the configured values.

Faults 1 and 2 have been fixed (module switched to `.eval()`, recurrent state threaded through and cleared at episode boundaries, regression test added: `drl-manager/tests/test_rllib_scheduler_inference_state.py`; 70 passed, 1 expected xfail; repeated evaluations after the fix are identical action for action, trajectory for trajectory, and in carbon). Faults 3 and 4 are **not** fixed and are the next actions.

**The evaluation fault was not the whole story.** With the memory and dropout faults corrected, the three vanilla seeds still sit **13.6 %–27.8 % above `cover_argmax` at shrink 0.75, mean 22.4 %** (`corrected_primary_vanilla.json`, rule reference 0.001919). The training side still has to be repaired.

## 2. The reward is not misaligned with carbon

Checked over 369 training episodes: the episodic reward satisfies `constant − 20000 × carbon` exactly. Carbon enters through the legacy global term (`calculateGlobalReward`: `− β · normalizedCarbon` with `global_reward_beta: 1.0`), while the per-action routing reward carries none (`per_action_carbon_weight: 0.0`, `routeReward = −w_carbon·normalizeCarbon + completionTerm`). Whatever else is wrong, the objective the policy optimises is a strictly decreasing function of carbon.

## 3. The drift, measured

Same fixed states, policy against its own initial prior (`bisect_summary.csv`, mode `recurrent`, tier shrink 0.75, one window):

| steps | KL from prior | residual max logit | rule agreement | entropy | carbon | gap vs rule |
|---|---|---|---|---|---|---|
| 8 000 | 0.025 | 0.335 | 0.06 | 4.74 | 0.001033 | +7.5 % |
| 24 000 | 0.133 | 1.550 | 0.00 | 4.67 | 0.001035 | +7.7 % |
| 40 000 | 0.282 | 2.751 | 0.03 | 4.55 | 0.001136 | +18.2 % |
| 56 000 | 0.334 | 3.444 | 0.06 | 4.37 | 0.001395 | +45.1 % |
| 72 000 | 0.388 | 3.897 | 0.00 | 4.50 | 0.001252 | +30.2 % |
| 88 000 | 0.508 | 4.360 | 0.00 | 4.32 | 0.001852 | +92.7 % |
| 104 000 | 0.656 | 5.074 | 0.00 | 4.24 | 0.001510 | +57.1 % |
| 120 000 | **0.925** | **5.538** | 0.00 | 3.96 | 0.001660 | +72.7 % |

KL from the prior and the learned residual's magnitude grow **monotonically and without saturation**; carbon degrades with them, noisily, on a single window. The same signature appears in the parameter norms measured independently from the checkpoints: `offset_head` ‖W‖ 0.373 → 1.566 and `dc_encoder` ‖W‖ 0.215 → 1.228 from an exact-zero initialisation, with no plateau at 120 000 steps.

**There is no cliff.** The degradation is the accumulation of many individually legal PPO updates, exactly the mechanism the brief warned about: clipping and adaptive KL bound *consecutive* updates and say nothing about distance from the initial prior. Nothing in the configuration anchors the policy to it: the prior is a fixed additive logit (`cover_prior_gain: 20.0`, a buffer that is never trained) and every learned contribution starts at zero and is unconstrained thereafter.

One number deserves its own line: the probability the policy assigns to the rule's own action is **0.008–0.019 throughout**, and the deterministic agreement with the rule is 0–6 % **from the first checkpoint onwards**. The sampled policy never carried the prior — which is the same fact the RL_V2 smoke recorded as `STOP_INIT_PRIOR_NOT_CARRIED` and which is why deployment was moved to deterministic decode.

## 4. Root cause, as far as the evidence goes

- **Established**: two evaluation-path faults (memory, dropout), now fixed; two further train/eval inconsistencies (positional encoding, GAE parameters), not yet fixed; the reward is monotone in carbon; the policy's distance from the prior grows monotonically with no anchor; the residual grows without saturation; the corrected evaluation still leaves vanilla 22.4 % worse than the rule.
- **High probability**: the dominant training-side mechanism is unanchored cumulative drift away from a strong prior in an objective whose local gradient is weak relative to the noise, with the entropy bonus supplying a persistent outward force.
- **Hypothesis, not established**: that the positional-encoding mismatch and the wrong GAE parameters are themselves large contributors. They are correctness faults and must be fixed before any further attribution.

## 5. Next actions, in order

1. Fix the training-time positional encoding and make the single-step and sequence log-probabilities consistent.
2. Fix the per-module GAE wiring so γ and λ are the configured 0.999 / 0.98.
3. Add a no-update identity test: with a zero-length or zero-learning-rate update, evaluation must reproduce the pre-update policy exactly.
4. Run **one** vanilla line, one seed, short (8 000–24 000 steps) and verify the prior-preservation gate.
5. Only after that gate passes, restore three seeds and the EU-CRD comparison.

No new training has been started, no existing verdict has been rewritten, and no change has been made to PPO, the reward, the learning rate or the EU-CRD training mathematics.
