# Scene and observation-interface design (v0 draft, 2026-09-05 23:20; for the user, then Codex; nothing frozen, nothing run)

This is the prospective design Codex asked for after closing the action-space line (STAGE_D_PRIME_DESIGN §35). It does not reopen the option or offset verdicts. It builds on two established facts and three open questions.

Established (design log §3, §32, §35 and its diagnostic): the value in the HZ ×2 scene is temporal; an oracle-driven (DC, dispatch-offset) action with a fine grid reproduces the reserving planner exactly on every window; the frozen dyadic grid's downward quantisation, not the action, cost the k4/k5 windows.

Open, in the order they must be answered: (i) whether a fixed reservation alone captures the gain (necessity against the strong blind family); (ii) whether a policy can learn the decision from forecast features it is allowed to see (representation learnability); (iii) whether PPO and EU-CRD then inherit or resist forecast error (the thesis).

## 1. Scene

### 1.1 Turbines and year: two options, one recommendation

- (A) Same five HZ turbines (123/10, 51/53, 112), 2021 file (52,559 rows), unread windows only. Known constraint (design log §17): once every read window and its full 2,922-row footprint is excluded, the largest remaining gap is 2,119 rows, so six disjoint unread windows do not exist; only a read-only exclusion (footprint not excluded) yields six, and the headroom gate of §1.2 will remove more. Cheapest, but the pool is nearly spent and every further read burns it.
- (B) A never-used turbine set (106 eligible ids with complete 2020 and 2021 files, `stage_a_out/turbine_usage_inventory.json`), chosen by the frozen hash rule of `stage_d_prime_turbines.py` (tag stage-d-prime-turbines-v1, first five in hash order mapped onto the HZ layout DC0 ← 2, DC1 ← 2, DC2 ← 1). Costs the full re-certification the ruling of §24 already required for any successor: the zero-training scene gate (mechanism control on the reactive-wait blind, shrink/shuffle/anti below the blind), the TimeCAP error calibration on these turbines (`calibrated_shrink_hz_v2`), the margin probe, and P0′. Gives a clean 2021 pool of 52,559 rows for development and judgement, and a 2020 file for a cross-year confirmation.

Recommendation: (B). The window pool is what every gate consumes, and (A) cannot supply a headroom-gated, unread six-window judgement set plus a development set.

### 1.2 Window rule with a headroom gate (frozen before any window is read)

Candidate offsets: every legal start of a 2,922-row footprint; order by sha256("scene-interface-v1:" + offset); walk in that order and accept a window iff (a) it does not overlap an accepted window and (b) it passes the headroom gate, until six judgement windows and six development windows are accepted (development first in hash order, then judgement, disjoint from each other and from everything read). Fewer than twelve → STOP_WINDOW_SPLIT.

Headroom gate, read from two analytic arms only, both step-wise and already frozen: B = `reactive_wait_planner` (blind) and ST = the reserving godeye planner. A window passes iff (C_B − C_ST) / C_B ≥ 0.15 and C_B − C_ST ≥ h_min kg, with h_min set from the certification runs of the new turbines as the 25th percentile of the per-window gap over the first twelve hash-ordered candidates that pass the relative gate (recorded, then frozen; no policy is read). The rule reads carbon of two analytic arms and nothing else; the arms' rows on rejected windows are archived and never reused.

Interpretation fixed now: a window that fails the gate is a scene in which the forecast has nothing to add, and no method is judged on it.

## 2. Action

(DC, dispatch-offset) as in OPTION_ACTION_DESIGN §8 / Addendum C, with two changes decided from the §35 diagnostic before any row exists:

- grid: every step from 0 to W (W = the scene's wait cap in steps; on the HZ cell W = 72, 73 values); the module's offset head has W + 1 outputs;
- quantisation: none needed for the every-step grid; if a coarser grid is ever preregistered, the rule is nearest-legal, not down.

Executor, legality mask, ledger, timing truth and gate-3 contract exactly as Addendum C (fixed-start reservation, no green read, illegal offsets masked never clipped, route→start ≤ 1 step).

## 3. Observation interface

The policy's forecast view changes from four per-site summaries to features aligned with the candidate actions, every one computed from the forecast the arm is given (truth for godeye, the TimeCAP output for timecap arms, the perturbed curve for shuffle / anti / calibrated arms) through one shared function, so no arm can see the oracle's answer:

- `cand_green_cover[j, d, κ]`: the share of job j's energy draw that green would cover if it started at t + κ + lag at site d and ran its full runtime, under the arm's curve (0..1);
- `cand_feasible[j, d, κ]`: the legality mask (deadline and reservation), already an obs key;
- `cand_deadline_margin[j, κ]`: (latest start − (t + κ + lag)) / deadline scale, clipped;
- the existing per-site held-ledger keys and per-job timing keys (obs_v31) stay; the four summaries stay as well (they are cheap and the ablation in §4.3 needs them).

Shape: (NB, n·(W+1)) per feature; on the HZ cell 128 × 365 per key. The score-based module's offset mode consumes them as candidate-aligned inputs added to logit(d, κ) through a small per-candidate MLP (site score + offset head + candidate-feature term); the candidate-feature term is the only new parameter block and is the object of the ablation.

## 4. Gates (zero training until §4.4), same order as before

### 4.1 Expressibility
oracle_off (every-step grid) against B and ST on the six development windows. Pooled capture ≥ 0.80 and headroom-weighted window robustness: Σ_w gap_w · [capture_w ≥ 0.70] / Σ_w gap_w ≥ 0.80 (a window's vote weighted by its own avoidable carbon), with the pooled and the weighted rule both frozen now. From the §35 diagnostic this gate is expected to pass; it is kept because the scene is new.

### 4.2 Predictive necessity
Blind family of Addendum C3 (fixed_off(κ) on the every-step grid is 73 arms; frozen to fixed_off(κ) for κ ∈ {0, 4, 8, …, 72} = 19 arms, plus reactive_off, persistence_off, climatology_off), run first, blind* frozen by pooled carbon, then oracle / shuffle / anti; the three conditions of §6 gate 2 with 0.95, on the pooled sum and the headroom-weighted window rule.

### 4.3 Representation learnability (replaces gate 4 and Addendum D2)
Three supervised fits, same corpus (oracle_off decisions on the development windows), same split, same frozen hyper-parameters (Addendum C4):
- F1: the D′ observation (four summaries, no candidate features);
- F2: the candidate-aligned features computed from the truth curve (the interface with perfect forecast);
- F3: the candidate-aligned features computed from the TimeCAP forecast (the interface the RL will actually have).
Classification gate on p_delay (lift ≥ 0.10, balanced AUC ≥ 0.60) and executed capture ≥ 0.50 on the held-out windows, read for F3 as the gate and for F1 / F2 as the ablation that says where the information is lost (F1 fails and F2 passes → representation; F2 passes and F3 fails → forecast quality on this scene; all fail → sample or architecture).

### 4.4 RL and EU-CRD (separate preregistration after 4.1–4.3 pass)
Vanilla PPO and EU-CRD on the offset action with the F3 interface, five new seeds, the six judgement windows one-shot, the four-line design of Stage D, the contract of Stage D, perturbation arms shuffle / anti / calibrated_shrink_hz_v2; the SMDP credit question of OPTION_ACTION_DESIGN §3.2 resolved in that document, not before.

## 5. Costs and order

Re-certification of the new turbines (scene gate, calibration, margin probe, P0′): about one day of zero-training runs. Window rule with headroom gate: one hour. Gates 4.1–4.3: two to three hours. RL: 400k steps × 5 seeds × 4 lines on two machines, two to three days. Nothing runs before Codex approves this document and it is frozen.

## 6. Decisions requested from the user before this goes to Codex

1. Option (A) or (B) for the turbines (recommendation: B).
2. A target date, so the number of seeds and gates can be sized to it.
