# RL / EU-CRD on the certification twin (preregistration DRAFT, 2026-09-06)

Status: DRAFT, not frozen. Conditional on F3 passing its executed-capture gate (reports/F_FITS_PREREG.md). Written after the offline ladder (LADDER_READ), the causal expert (CAUSAL_READ: capture 0.968, λ = 0.75 harms 6/6 windows by +51.8 %) and before any RL training. Order per the user's ruling of 2026-09-06. Freezing this document commits GPU-days on two machines; it is frozen only after the user has read it. The 2020 confirmation windows stay sealed until the freeze names them.

## 1. Claim under test

On a scene where forecast error is certified to hurt an exact planner (offline and causal), does a vanilla RL policy that uses the forecast inherit that harm along the same controlled ladder, and does EU-CRD reduce the harm without giving up clean performance. Everything else in the chain is now certified; this document only adds the learned policy.

## 2. Scene, action, interface (fixed)

- Simulator: the certification twin (`ladder_run.cert_config`, step-aligned datacenter updates, dense every-step (DC, κ) executor, `OFFSET_GRID_DENSE=1`), new turbines 133/78 | 22/81 | 94, 32-PE jobs, 48 s, wait cap 72, LAG 2 semantics as certified.
- Action: `offset_v1`, a = site·73 + κ, mask as in the interface twin.
- Observation: the F3 interface (four per-site summaries + `cand_green_cover` computed from the arm's own forecast channel). Nothing else is added.
- Reward: the registered per-action carbon objective of the scene's RL base, unchanged (Stage D §2); reward–carbon sign agreement is a gate, not an assumption.
- Training workload: the generator's c3_n35 trace (the causal expert's own trace); evaluation on the same trace over the judgement windows (the six HZ evaluation cells of Stage D are not used: the certified chain is on this trace).

## 3. Lines (train clean, corrupt at deployment; Stage D §1)

| line | forecast channel during training | credit assignment |
|---|---|---|
| N | forecast hollowed (`forecast_mode: none`; `cand_green_cover` computed from the persistence curve so the key exists with no future information; shape identical) | vanilla PPO |
| V | clean truth curve (`perturbed_godeye`, tier `godeye`) | vanilla PPO |
| E | the same clean curve | EU-CRD (the frozen v5.2 block, `crd.enabled: true`, nothing else changed) |

Reference arms on every judgement window (frozen, zero parameters): the causal expert (truth curve), the offline exact planner (truth and λ = 0) and `cover_argmax` on the arm's own interface key (truth key for the clean reading, TimeCAP key for the timecap reading; 0.765 / 0.637 pooled capture on the development windows). Gate 1 below is read against `cover_argmax` as well: a learned policy that does not beat the zero-parameter rule on its own interface has not learned the interface.

Deployment readings of the frozen last checkpoint (stochastic decode as registered; never argmax): N on its own channel once; V and E under the tiers godeye (clean), shrink λ ∈ {0.75, 0.5, 0.25, 0} (the ladder's rungs: shrink towards the site's full-year 2021 mean, implemented in `perturbed_godeye_provider` and equality-tested against `ladder_run.rung_curve` on one window), shuffle and anti (negative controls), and timecap (the natural TimeCAP error, descriptive only; it does not override the A2 STOP).

## 4. Windows

- Training: a frozen allowlist of unread 2021 offsets chosen at freeze time by the scene-v2 pool rule (≥ 2922 rows from every read window: the six development windows, the certification candidates 49625 / 36713 / 30299 and the 2021 pool windows), recorded with hashes; `green_episode_offset_allowlist`.
- Health smoke and development readings: the six 2021 development windows (already read; "certified benchmark", no generalisation claim).
- Final judgement: the six 2020 confirmation windows of SCENE_INTERFACE_DESIGN (24398, 10829, 7479, 20843, 523, 14997), read once after every model, reader and gate is frozen. The causal expert and the offline planner are run on them in the same pass (their curves and gates are frozen; they provide C_flat and C_causal per window).

## 5. Seeds, checkpoints, budget

- Paired seeds identical across N/V/E: 3 minimum, 5 if Isambard is available for the run (decided at freeze from machine availability, never from a deadline). 400k timesteps per line and seed (the Stage D long-run count). Last checkpoint at the registered count; no selection on any evaluation number.
- Health smoke first (1 seed, 50k, local GPU): Stage D §5 items plus one lesson of Stage D: the clean policy's wait distribution is neither collapsed to κ = 1 nor to κ = 72 (share of decisions at either end < 0.90) and the E line's contract is green on the six development windows (Stage D's E failed 33/36 contracts). Failure is a pipeline finding, fixed as an addendum before the long run.

## 6. Verdict gates (pooled over the judgement windows; per seed; each direction gate in ≥ 2/3 seeds, ≥ 4/5 with five)

With C_X(tier) the pooled simulator carbon of line X under a tier, C_flat and C_causal from the frozen planners on the same windows:

1. Vanilla uses the forecast: (C_N − C_V(godeye)) / C_N ≥ 5 % AND V captures ≥ 0.50 of the causal expert's headroom: (C_flat − C_V(godeye)) / (C_flat − C_causal) ≥ 0.50.
2. Vanilla inherits the ladder: loss_V(λ) = C_V(λ) − C_V(godeye) > 0 for every λ ∈ {0.75, 0.5, 0.25, 0}, non-decreasing as λ falls (one inversion between adjacent rungs tolerated), and loss_V(0.75) ≥ 5 % of C_V(godeye).
3. EU-CRD uses the forecast and keeps clean performance: (C_N − C_E(godeye)) / C_N ≥ 5 % and C_E(godeye) ≤ 1.05 · C_V(godeye).
4. EU-CRD reduces the degradation: the area under the pooled loss curve over λ (trapezoid on λ = 0.75, 0.5, 0.25, 0) A_E ≤ 0.5 · A_V, and loss_E(0.75) ≤ 0.5 · loss_V(0.75).
5. Absolute conditions: contracts green on every evaluation episode (completion ≥ 0.995, on-time ≥ 0.995, forced 0, stale 0); reward and physical carbon improve in the same direction from the init to the last checkpoint for V and E; EU-CRD's Δr, uncertainty gate and auditors demonstrably active (logged statistics).

shuffle / anti are reported for V and E, never gated; timecap is reported as a separate descriptive row.

## 7. Stop rule

Gates 1–2 fail → the chain stops at "forecast error hurts the exact and the causal planner but a trained policy either ignores the forecast or is not hurt", reported as such; no re-tuning toward a pass. Gates 1–2 pass and 3–5 fail → EU-CRD's failure on this scene is the result; submission decisions are separate from the gates.

## 8. Implementation checklist before the health smoke (each with a test)

- `perturbed_godeye_provider`: tiers `shrink_0.75`, `shrink_0.5`, `shrink_0.25`, `shrink_0` (full-year 2021 site mean, the ladder's `_mu_w`), equality test against `rung_curve` on one window; `timecap` tier through the interface config's checkpoint.
- `gen_rl_cert.py`: N / V / E blocks derived from `config_ladder_cert_interface.yml` by a whitelisted diff (test asserts no other key changes), manifest of hashes (configs, jar, planner, curves, allowlist).
- Training allowlist script with the pool rule and its record.
- Evaluation harness rows carry tier, curve signature, checkpoint hash, contract counters; the causal expert and offline planner readings on the judgement windows produced by the same harness.
- Isambard: gateway jar rebuilt from the freeze commit (JDK 21 via the miniforge env), `timecap.device: cpu`, the RAY_LIMIT_CPUS and tmpdir notes of the migration guide.
