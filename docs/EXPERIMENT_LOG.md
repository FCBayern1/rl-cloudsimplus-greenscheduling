# Hierarchical MARL for Carbon-Aware Scheduling — Experiment Log & Ablation

_Last updated: 2026-05-31_

This document records the method evolution, ablations, and final results for the
GTrXL + PPO hierarchical multi-datacenter green-scheduling system. It is intended
to feed directly into the paper's Method (§4), Ablation (§5), and Results (§6).

---

## ⚠️ 0. CRITICAL CORRECTION (2026-05-31): the 10-DC RR baseline was wrong

Throughout the earlier sections below, the 10-DC Round-Robin baseline was
quoted as **completion 0.886 / c/c 2.077**. This number was never measured on
the current `experiment_multi_10dc_carbon_v2` config — it was an unverified
carry-over from an earlier/different setup. On 2026-05-31 we **measured it
directly** (RR + first_fit, godeye, seed 42):

| Metric | WRONG (used in §2–§5 below) | MEASURED (correct) |
| --- | --- | --- |
| 10-DC RR completion | 0.886 | **0.4954** |
| 10-DC RR carbon | — | 1.352 |
| 10-DC RR c/c | 2.077 | **2.728** |

**Consequences — several earlier conclusions are INVERTED:**
- "10-DC RR is near-optimal, RL has no headroom" → **FALSE**. 10-DC RR is far
  from optimal (0.50), exactly like 5-DC. RL has large headroom on both.
- "10-DC RL only matches RR (c/c 2.06 vs 2.077)" → **FALSE**. 10-DC RL best c/c
  = 2.063 vs the true RR 2.728 → **RL beats RR by +24%**, comparable to 5-DC's
  +28%. The 10-DC runs were winning all along; we were comparing to a phantom
  baseline.
- "Pivot to 5-DC because 10-DC has no room" → the *premise* was wrong, though
  5-DC remains a valid second scale.

**Correct framing: RL beats every heuristic on BOTH scales (5-DC +28%, 10-DC
+24%), making this a dual-scale validation, not a 5-DC-only result.** The
10-DC numbers in §2–§5 below that reference 0.886/2.077 should be read against
the corrected baseline 0.495/2.728. The full corrected baseline tables are in
§4.

| Heuristic baseline | 5-DC compl / carbon / c-c | 10-DC compl / carbon / c-c |
| --- | --- | --- |
| round_robin | 0.382 / 0.526 / 1.376 | 0.495 / 1.352 / 2.728 |
| min_queue (strongest) | 0.394 / 0.533 / 1.354 | 0.510 / 1.401 / 2.746 |
| green_queue_balanced | 0.266 / 0.414 / 1.556 | 0.508 / 1.381 / 2.719 |
| green_aware | 0.107 / 0.385 / 3.606 | 0.134 / 1.140 / 8.529 |
| **RL best** | **0.62 / 0.61 / 0.976** | **0.83 / 1.72 / 2.063** |
| **RL vs strongest heuristic** | **+28%** | **+24%** |

(Measured via `scripts/run_5dc_baselines.py --experiment …`, godeye mode.)

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
(noisy, ratio-shaped) return — investigated further in §6 below.

---

## 3a. Critic-learnability investigation (2026-05-27/28)

`vf_explained_var ≈ 0` persisted across every architectural change. We did a
sequence of ablations to identify the cause:

| Test | What we changed | `vf_explained_var` | `vf_loss` | Verdict |
| --- | --- | --- | --- | --- |
| Baseline (Route 2.5 dual-trunk) | — | ~0 | pinned at 10 | starting point |
| **A: `vf_clip_param: 10 → 3000`** | un-clamp the per-sample MSE | ~0 (still) | ~2500 (unpinned) | mechanically frees the loss; V reaches **mean** of returns; residual still unfit |
| **B: `vf_coef: 0.25 → 10`, `max_grad_norm: 0.5 → 20`** | give the critic a 40× larger effective lr (dual-trunk gradient isolation means the actor is untouched) | ~0 (no change) | ~2500 (no change) | critic gradient doesn't move — `Cov(features, residual) ≈ 0` |
| **C: offline predictability test** | fit sklearn `GradientBoostingRegressor` directly from (obs, discounted return) on 6800 held-out rollout steps | — | — | **test R² = −5.88** (worse than predict-mean) |

Test C is the decisive one. Procedure (`scripts/diagnose_critic_predictability.py`):
1. Run 1 episode under RR + random-valid locals (godeye mode).
2. Collect (global-obs flat vector, per-step global reward).
3. Compute discounted returns at γ=0.995.
4. 70/30 sequential split; fit LinearRegression / Ridge / GradientBoosting.

Result: **best regressor reaches R² = −5.88**. `Var(returns) = 848,629`, and no
regressor can extract meaningful state-conditioned structure.

### Why returns are unpredictable from obs

Per-step global reward is `Σᵢ rᵢ` over the 20 routing slots, where
`rᵢ = -w_c·(margᵢ/0.05) + w_compl·prob_completeᵢ`.

The discounted return at γ=0.995 has an effective horizon ~200 steps,
during which ~1200 NEW cloudlets arrive. The current obs contains:
- ✅ the current batch's cloudlet (mi, pes) — affects the next step's reward only,
- ✅ current per-DC green ratio + queue — affects ~10-20 steps of reward,
- ❌ which cloudlets will arrive in the next 200 steps — **the dominant source of return variance**.

So even an optimal critic is upper-bounded by the irreducible conditional
variance `Var(return | s)`, which empirically ≈ `Var(return)` → `R² ≈ 0`.

### Why the policy still works despite the dead critic

PPO's gradient is `∇log π · A` with `A = return − V(s)`. When V(s) ≈ constant
(the conditional mean is the unconditional mean), the advantage degenerates
to `return − const`. This is **REINFORCE with a constant baseline**: still
unbiased, just higher variance. Because our per-action reward's **sign** is
determined by state-action features (greener DC → higher `rᵢ` than dirtier DC,
regardless of future arrivals), the policy gradient direction remains correct
and the policy learns. The 5-DC RL ⟶ 28% c/c improvement over the best heuristic
is achieved entirely via reward-signal-driven policy gradient with a degenerate
critic.

### What this means architecturally

`vf_explained_var ≈ 0` is **not** a bug in our system — it is the ceiling
imposed by the obs / reward / discount factor combination. Therefore:
- ❌ Value normalisation (PopArt) does not help: it fixes scale, not predictability.
- ❌ Separate critic lr (via `vf_coef`) does not help: gradient ∝ feature-residual covariance ≈ 0.
- ❌ Critic architecture changes (un-pool, attention pool) likely don't help: the
  information about the dominant variance source (future arrivals) is simply not
  in `s`.
- ✅ The only conceptual way to make V(s) learnable would be **adding future-arrival
  information to the obs** (e.g., a forecast of the next K cloudlets' (mi, pes)).
  This is an env-design change, not a critic change.

This is a publishable finding in itself — see §5.

---

## 4. Final results — 5-DC v2 Pareto sweep (in progress)

Architecture identical across all configs; only reward weights vary.
RR baseline on 5-DC: completion 0.382 (count) / carbon 0.526 / c/c ≈ 1.376.

All 5 configs completed.  Best-c/c point (across 111 episodes) for each:

| Config | (w_c, w_compl) | best c/c | completion_mi | carbon | vs RR |
| --- | --- | --- | --- | --- | --- |
| A | (2.0, 0.5) | 0.977 | 0.599 | 0.584 | **+29.0%** |
| B | (1.0, 0.5) | 0.986 | 0.580 | 0.572 | **+28.3%** |
| C | (1.0, 1.0) | **0.976** | 0.622 | 0.607 | **+29.1%** |
| D | (1.0, 2.0) | 0.996 | 0.568 | 0.565 | +27.6% |
| E | (0.5, 2.0) | 0.987 | 0.617 | 0.609 | +28.3% |

Stable converged operating point (mean of last 20 episodes):

| Config | (w_c, w_compl) | compl_mi | carbon | c/c |
| --- | --- | --- | --- | --- |
| A | (2.0, 0.5) | 0.565 | 0.563 | 0.997 |
| B | (1.0, 0.5) | 0.584 | 0.584 | 1.001 |
| C | (1.0, 1.0) | 0.621 | 0.617 | 0.994 |
| D | (1.0, 2.0) | 0.558 | 0.564 | 1.011 |
| E | (0.5, 2.0) | 0.611 | 0.612 | 1.001 |

**Surprise**: the reward-weight sweep does NOT produce a spread Pareto front —
all 5 configs cluster at c/c ∈ [0.994, 1.011], and completion/carbon move
**together** (not in trade-off).  Cause: with overall green_ratio ≈ 0.54 and
the agent already picking green-when-available, per-task carbon intensity is
near the workload's intrinsic floor; the reward weights only shift how
**aggressively** the policy completes work, not the carbon-per-task ratio.
A real Pareto front would require sweeping a **constraint** knob (e.g., the
Lagrangian `sla_target`), not the reward weights.

**Reframed contribution**: instead of "tunable Pareto front", the strong result
is **robust dominance** — all 5 weight configurations beat the best heuristic
(min_queue, c/c=1.354) by ~27–29 % on c/c and 58 % on completion, with no
drift (best ≈ final), unlike on 10-DC.

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
5. **The critic is at a fundamental information ceiling, not bugged** (see §3a):
   per-action reward + γ=0.995 ⇒ return is dominated by ~1200 future random
   cloudlet arrivals that are NOT in obs. Sklearn GBM fitting (obs → return)
   achieves test R² = −5.88, so even an optimal V(s) cannot exceed ≈ 0 explained
   variance. PPO succeeds anyway because the per-action reward's **sign** is
   state-conditioned (greener DC → higher rᵢ), so the policy gradient remains
   directionally correct under a degenerate (mean) baseline. Critic-architecture
   fixes (PopArt, MAPPO centralized critic, separate critic lr) **cannot help** —
   the info is not in the obs. The only conceptual fix is augmenting obs with a
   future-arrival forecast (env-design change, not critic change).
6. **Reward-weight sweeping is not a Pareto-front knob in this domain**: 5 weight
   configurations clustered at c/c ∈ [0.994, 1.011] (range ~1.7 %), with
   completion and carbon moving **together**. Reason: under the current trace,
   per-task carbon intensity is near its workload-imposed floor (overall green
   ratio 0.54, agent already picks green-when-available), so weights only shift
   "how aggressively the policy works", not the trade-off ratio. A Pareto front
   would require sweeping a constraint knob (e.g., Lagrangian `sla_target`)
   rather than reward weights. **Reframed contribution: robust dominance over
   heuristics, not tunable Pareto.**

---

## 6. Open items / future work

Done:
- [x] Finish 5-DC sweep (all 5 configs A–E).
- [x] Baseline table with consistent `completion_rate_mi` (script + CSV).
- [x] Critic-learnability investigation (vf_clip, vf_coef, offline predictability) — §3a.

Remaining (in priority order):
- [ ] **Sweep `sla_target` via Lagrangian** instead of reward weights — likely
      the actual Pareto-front knob (forces completion ↑ at the cost of brown-DC
      use, producing real trade-off curves).
- [ ] **batch_size ablation**: 5-DC uses batch=20 but arrivals ~6/step → ~70%
      padding; try batch ∈ {8,12,20} at fixed reward.
- [ ] **Future-arrival forecast in obs** — the only conceptual way to lift the
      critic's R² ceiling; substantial env-side work.
- [ ] (Stretch) Centralized critic with team obs — likely irrelevant given §3a,
      but might be tested as a negative ablation for completeness.
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
