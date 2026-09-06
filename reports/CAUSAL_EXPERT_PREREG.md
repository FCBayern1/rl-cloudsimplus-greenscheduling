# Causal rolling expert: reachability of the certified ladder's value, and the causal error gate (preregistration, 2026-09-06)

Status: FROZEN by the commit that adds this file (hash in STAGE_D_PRIME_DESIGN.md §49). Written after the offline ladder's reading (reports/LADDER_V2_READING_2026_09_06.md) and before the causal expert is run on any development window. Ordered by the user's ruling of 2026-09-06: offline exact ladder → causal expert and causal error gate → F1–F3 → RL preregistration → training. The 2020 confirmation windows stay sealed.

## 1. Question

The offline exact planner knows every future job. An online policy sees only the jobs that have arrived. Is the value the ladder certified reachable with that information, and does forecast error still hurt an expert that has only that information? If not, the ladder's harm is not the harm an RL policy can be exposed to, and RL must not start.

## 2. The expert (`causal_expert`, `src/baselines/global_schedulers.py`)

At every step t it decides only the jobs the simulator has presented and it has not yet committed, from three things: those jobs (PEs, MI, seconds to deadline as presented), its own committed reservations (draw in mW and occupancy per site and row, exactly what the every-step offset executor will run), and its own future green curve. It never reads a job before its sighting. Each decision is the version-2 model of `ladder_planner` (per-site host profiles, one host per job, occupancy premise, envelope cuts) with the committed load as base, on the candidate starts the executor's legality mask allows per site, solved to proven optimality (120 s limit; an unproven step is counted, the affected jobs are routed now, and the run is reported as such). The decision is emitted as (site, κ = start − t − 1) on the dense grid; the executor executes it exactly (certified in the settlement diagnostic). Settlement is the run itself on the certification twin: the simulator's carbon is the reading.

Curves: the arm builds its curve from the wind files for the window like the ladder (`truth_curve`, `rung_curve`): truth, shrink λ ∈ {0.75, 0.5, 0.25, 0} around the site's full-year 2021 mean, seeded shuffle, anti. The curve signature and the truth signature are recorded in every run.

## 3. Runs (development windows k0–k5 of `scene_v2_dev.json`, certification offset twin, dense grid)

Seven arms × six windows: causal_truth, causal_shrink_0.75, causal_shrink_0.5, causal_shrink_0.25, causal_shrink_0, causal_shuffle, causal_anti. Every run records the option ledger, the decision dump, the counters (`causal_unsolved`, `causal_fallback`, contract counters) and the simulator carbon. causal_truth runs first and is judged (gate A) before any other arm's carbon is read.

## 4. Gates

Reference numbers, simulator-settled, from the frozen ladder (`reports/manifests/ladder_v4/run1/ladder_verdict.json`): per window C_truth (offline exact on truth) and C_flat (offline exact on the λ = 0 curve); headroom_k = C_flat,k − C_truth,k.

Gate A, reachability: capture_k = (C_flat,k − C_causal_truth,k) / headroom_k. PASS iff pooled capture Σ_k (C_flat,k − C_causal_truth,k) / Σ_k headroom_k ≥ 0.80 AND capture_k ≥ 0.70 on at least 5 of 6 windows. Contract: every run completion ≥ 0.995 by MI, ontime ≥ 0.995, deadline_forced 0, no stale, `causal_unsolved` 0 (an unsolved step voids the window; the window is reported, not replaced).

Gate B, causal error: loss_k(λ) = C_causal_λ,k − C_causal_truth,k. PASS iff for λ = 0.75 loss_k > 0 on at least 5 of 6 windows AND Σ_k loss_k(0.75) ≥ 0.05 · Σ_k C_causal_truth,k. The other rungs are reported as the causal loss profile (no gate), read together with the offline profile.

Verdicts: A fails → STOP_CAUSAL_UNREACHABLE (the ladder's value needs future arrivals; RL is not started; the next design question is the expert's information set, not the policy). A passes and B fails → STOP_CAUSAL_ERROR_HARMLESS (error hurts only the clairvoyant planner; RL is not started). Both pass → CAUSAL_READ; F1–F3 proceed with the causal expert's decisions as labels.

## 5. Not changed, not read

No ladder number is recomputed. No tolerance, gate or curve definition of the ladder is touched. The 2020 windows are not read. Nothing is tuned after a reading; the expert's time limit and horizon are fixed here (120 s, 1400 rows).
