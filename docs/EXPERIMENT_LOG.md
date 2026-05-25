# Hierarchical MARL for Carbon-Aware Scheduling — Experiment Log & Ablation

_Last updated: 2026-05-25_

This document records the method evolution, ablations, and final results for the
GTrXL + PPO hierarchical multi-datacenter green-scheduling system. It is intended
to feed directly into the paper's Method (§4), Ablation (§5), and Results (§6).

---

## 1. Problem setup

- **Task**: route incoming cloudlets across N datacenters (DCs) to minimise carbon
  emission while maintaining task completion (SLA).
- **Objective metric**: `c/c = total_carbon_kg / completion_rate_mi` (carbon per
  unit of MI-weighted work completed). Lower is better.
- **Agents** (hierarchical, cooperative):
  - **Global agent** (1): routes a batch of cloudlets to DCs each step.
  - **Local agents** (N, parameter-shared): assign queued cloudlets to VMs within
    their DC.
- **Algorithm**: on-policy PPO (RLlib new API stack), GTrXL backbone with
  truncated-BPTT recurrence.
- **Constraint handling**: Lagrangian dual ascent on an SLA cost (pending-ratio).

### Classification (for related-work positioning)
| Axis | This system |
| --- | --- |
| Train/exec | Hierarchical DTE (CTDE path exists but off by default) |
| Paradigm | On-policy Actor-Critic (PPO), IPPO-style |
| Task nature | Cooperative w/ team-reward **decomposition** → principal-agent tension |
| Communication | Implicit only (stigmergy via env state + local parameter sharing) |
| Structure | 2-level hierarchy + permutation-equivariant score-based action factorisation |

---

## 2. Method evolution (chronological)

Each stage = one identified problem + the change + the empirical effect.

### Stage 1 — Per-action difference reward (2026-05-16)
- **Problem**: episode-level reward gives all routing slots the same advantage →
  no per-slot credit assignment for the global agent's `MultiDiscrete` action.
- **Change**: additive per-action reward `rᵢ = -w_c·(margᵢ - marg_RR) +
  w_compl·(probᵢ - prob_RR)`, a *difference reward* vs the next Round-Robin pick.
- **Effect**: carbon trended down (1.763→1.703 over 27 ep) but completion dropped
  (0.886 RR → 0.798); c/c ≈ 2.13, **worse than RR (2.077)**. The RR baseline in
  the diff reward flips sign once the policy beats RR → noisy/misleading gradient.

### Stage 3 — Score-based action factorisation (2026-05-17)
- **Problem**: `MultiDiscrete([N_dc]^N_batch)` joint action ≈ 10^10; 10 independent
  per-slot heads can't share "DC d is overloaded" knowledge.
- **Change**: `GTrXLScoreBasedGlobalRLModule` — logits `[i,d] = <q_i, k_d>/√D`
  with shared cloudlet/DC encoders. Permutation-equivariant over DCs and slots.
  Collapses the joint space into N_batch independent N_dc-way softmaxes.
- **Effect**: same MultiDiscrete API, PPO unchanged. Sample efficiency improves
  structurally (DC features learned once, reused across all slots).

### Stage 3 stability fix (2026-05-17)
- **Problem**: first score-based run diverged — iter-1 PPO update gave
  `global_entropy 22→0.348`, `mean_kl=inf`, `grad_norm=255`. Root cause: shared
  `dc_encoder`/`ctx_to_dc` receive ~N_batch× the gradient → effective lr too high;
  also raw obs (mi≈2e6) dominated encoder output.
- **Change** (5 measures): (1) input normalisation by `obs_space.high`;
  (2) encoder init gain 0.5; (3) score temperature ÷2; (4) global lr 0.0001→0.00006;
  (5) clip_range 0.2→0.15.
- **Effect**: init softmax ≈ uniform (0.094–0.105), `mean_kl < 0.05`,
  `grad_norm < 5`. Stable training restored. Locked by
  `test_init_logits_are_near_uniform_for_softmax_stability`.

### Reward weight rebalance "A+B" (2026-05-17)
- **Problem**: `w_compl=0.05` made carbon dominate 20× → agent gave up completion.
- **Change**: `w_compl 0.05→0.5`; lr 0.00003→0.00006.
- **Effect**: 10-DC c/c best 2.063 (iter 40), first time matching/slightly beating
  RR — but drifted back to 2.10 by iter 100.

### Route 2.5 Phase 2 — Dual critic trunk (2026-05-19)
- **Problem**: `vf_explained_var ≈ 0` throughout training; value loss back-prop
  through shared encoders perturbs the policy → "find good policy at iter 40, drift
  away by iter 100."
- **Change**: independent critic encoder + GTrXL trunk (`critic_separate_trunk`),
  LayerNorm+MLP value head. Actor/critic gradients fully isolated (unit-tested).
  Param count 405K → 820K.
- **Effect**: best c/c improved to 2.0497 (+1.3% vs RR) but **drift got worse**
  (best→final 2.05→2.12); `vf_explained_var` still ≈ 0. Conclusion: gradient
  isolation alone doesn't fix critic learnability.

### Reward redesign — absolute reward (2026-05-20)
- **Problem**: diagnosis that the RR-baseline diff reward points the wrong way once
  the agent beats RR (empirically `per_action_sum` went +159→-267 at the *best* c/c
  point).
- **Change**: drop the RR baseline. Absolute reward
  `rᵢ = -w_c·(margᵢ / marg_normalizer) + w_compl·probᵢ`, normalised to ~[0,1].
  Also vf_coef 0.5→0.25 (critic was diverging, vf_loss 1→8.5), Lagrangian
  lambda_lr 0.5→0.1, c_ep_tolerance 0.02→0.05.
- **Effect**: per-action signal became monotone (always points toward greener
  picks). Lagrangian no longer saturates. **But** completion locked at 0.79 because
  (a) absolute completion term (prob_complete) has low gradient variance vs carbon,
  (b) Lagrangian became too gentle.

### Reward rebalance round 2 (2026-05-21)
- **Change**: `w_compl 0.5→2.0`, `overflow_sharpness 3→6` (sharper prob_complete →
  more completion gradient), `lambda_lr 0.1→0.3`.
- **Effect (10-DC, 100 iter)**: completion recovered to 0.83 but carbon also rose →
  c/c best 2.11, final 2.16 — **worse than before**. 5 reward configurations all land
  in c/c ∈ [2.05, 2.16]. Conclusion: **this is the Pareto bandwidth on 10-DC, not a
  tuning problem.**

### Pivot — 5-DC Pareto sweep (2026-05-23)
- **Rationale**: 10-DC RR is near-optimal (completion 0.886), leaving little room.
  Instead of chasing single-best, sweep reward weights to map the Pareto front, and
  validate on 5-DC where RR is far from optimal.
- **Setup**: identical architecture to 10-DC latest (score-based + dual trunk +
  absolute reward). Only `(w_carbon, w_completion)` and the env differ. 5 configs.

---

## 3. Ablation table (10-DC v2, 100-iter runs)

| Config | best c/c | final c/c | drift | vf_explained_var | note |
| --- | --- | --- | --- | --- | --- |
| RR baseline | — | 2.077 | — | — | completion 0.886 |
| Stage 1 (diff reward) | 2.13 | 2.13 | — | ~0 | signal inverts when > RR |
| + Stage 3 score-based | — | — | — | ~0 | KL=inf without stability fix |
| + Stage 3 + A+B | **2.063** | 2.102 | +0.04 | ~0 | first to ~match RR |
| + Route 2.5 dual trunk | **2.0497** | 2.121 | +0.07 | ~0 | best point ↑, drift ↑ |
| + Reward redesign (abs) | 2.13 | 2.16 | — | ~0 | completion locked 0.79 |
| + Rebalance (w_compl=2, k=6) | 2.11 | 2.16 | +0.05 | ~0 | completion 0.83 but carbon ↑ |

**Take-away**: on 10-DC, every architectural fix shifts the operating point within a
narrow Pareto band (2.05–2.16) but never decisively beats RR (2.077). The persistent
`vf_explained_var ≈ 0` indicates the value function never learns to attribute the
(noisy, ratio-shaped) return — the true bottleneck.

---

## 4. Final results — 5-DC v2 Pareto sweep (in progress)

Architecture identical across all configs; only reward weights vary.
RR baseline on 5-DC: completion 0.382 (count) / carbon 0.526 / c/c ≈ 1.376.

| Config | (w_c, w_compl) | best c/c | completion_mi | carbon | vs RR |
| --- | --- | --- | --- | --- | --- |
| B | (1.0, 0.5) | 0.986 | 0.580 | 0.572 | **+28.3%** |
| C | (1.0, 1.0) | 0.976 | 0.622 | 0.607 | **+29.1%** |
| D | (1.0, 2.0) | 1.043* | 0.664 | 0.693 | +24.2%* |
| A | (2.0, 0.5) | _pending_ | | | |
| E | (0.5, 2.0) | _pending_ | | | |

_*D still running (iter ~25); value will improve._

**Pareto trend confirmed**: as `w_compl` rises (B→C→D), completion rises
(0.58→0.62→0.66) and carbon rises (0.57→0.61→0.69) — a clean trade-off front.
No drift (best ≈ final), unlike 10-DC.

### Heuristic baselines (5-DC, godeye, consistent `completion_rate_mi`)
Generated by `scripts/run_5dc_baselines.py` (seed 42, 1 episode). For these
heuristics MI-completion ≈ cloudlet-count completion (e.g. RR 0.3821 vs 0.3815),
so c/c is directly comparable to the RL runs.

| Scheduler | completion_mi | carbon (kg) | c/c | note |
| --- | --- | --- | --- | --- |
| green_aware | 0.107 | 0.385 | **3.606** | extreme green-bias → completion collapse |
| green_queue_balanced | 0.266 | 0.414 | **1.556** | over-greens → overloads green DCs |
| round_robin | 0.382 | 0.526 | **1.376** | naive |
| min_queue | 0.394 | 0.533 | **1.354** | strongest heuristic |
| **RL (config B)** | 0.580 | 0.572 | **0.986** | +27% vs best heuristic |
| **RL (config C)** | **0.622** | 0.607 | **0.976** | **+28% vs best heuristic (min_queue)** |

**RL dominates every heuristic**: config C's c/c (0.976) is 28% lower than the best
heuristic (min_queue 1.354), and its completion (0.62) is 58% higher than the best
heuristic (0.39). The greener a heuristic tries to be, the worse its c/c
(green_aware 3.6 ≫ min_queue 1.35) — only RL achieves low carbon *and* high
completion simultaneously.

---

## 5. Key findings (paper-worthy)

1. **Why 5-DC succeeds where 10-DC doesn't**: 10-DC RR is already near-optimal
   (completion 0.886) so RL has no headroom; 5-DC RR is far from optimal (0.38, hurt
   by PE-to-VM mismatch) so RL beats it by ~29%. RL's advantage scales with how far
   the baseline is from the frontier.
2. **All green-aware heuristics collapse on completion** (green_aware 0.11, GQB
   0.26): naively chasing green DCs overloads them. **Only the RL policy learns the
   "green but not overloaded" balance.** Strong argument for RL over hand-tuned rules.
3. **Reward design is the dominant lever, not architecture**: 5 architectural/reward
   variants on 10-DC stayed in a 0.10-wide c/c band. The decisive change was the
   *environment* (5-DC, far-from-optimal baseline), not more tuning.
4. **Difference-reward baselines must be on the Pareto frontier**: an RR baseline in
   a diff reward inverts the gradient once the agent surpasses RR — switching to an
   absolute reward fixed the signal direction.
5. **Critic learnability is the open bottleneck**: `vf_explained_var ≈ 0` persists
   across all 10-DC variants (dual-trunk isolation didn't fix it). Candidate future
   work: centralized critic (MAPPO Phase 1+3), value normalisation (PopArt).

---

## 6. Open items / future work

- [ ] Finish 5-DC sweep (configs A, E) → full Pareto figure
      (`scripts/plot_pareto_front.py`).
- [ ] Regenerate baseline table with `completion_rate_mi`
      (`scripts/run_5dc_baselines.py`).
- [ ] `batch_size` ablation: 5-DC uses batch=20 but arrivals ~6/step → ~70% padding.
      Try batch ∈ {8,12,20} at fixed reward to test credit-assignment effect.
- [ ] Centralized critic (Route 2.5 Phase 1+3) — give critic team-level obs to fix
      `vf_explained_var ≈ 0`.
- [ ] (optional) value normalisation / PopArt for the ratio-shaped return.

---

## 7. Reproducibility pointers

| What | Where |
| --- | --- |
| Score-based global module | `drl-manager/src/models/rlmodule_gtrxl_models.py::GTrXLScoreBasedGlobalRLModule` |
| Per-action reward (absolute) | `cloudsimplus-gateway/.../MultiDatacenterSimulationCore.java::accumulatePerActionReward` |
| Lagrangian callback | `drl-manager/src/callbacks/lagrangian_callback.py` |
| 5-DC sweep launcher | `drl-manager/scripts/run_5dc_pareto_sweep.sh` |
| Pareto plotter | `drl-manager/scripts/plot_pareto_front.py` |
| Baseline comparison | `drl-manager/scripts/run_5dc_baselines.py` |
| Configs | `config.yml::experiment_multi_{5,10}dc_carbon_v2` |
| Key tests | `drl-manager/tests/test_score_based_global.py`, `test_logger_per_action_reward.py`, `test_auto_plot_reward_breakdown.py` |
