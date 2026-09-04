# Stage D health smoke, run 2 — PASS_HEALTH (HEALTH_SMOKE, no effect claim)

Date 2026-09-04. Freeze commit 287ba36f (clean tree), jar 6931ed570c50, seed 20260903, 56000 steps per line (7 PPO iterations of 8000), ledger-aligned reward (`config_stage_d_physical.yml`), four lines N_V / V / N_E / E trained on the local RTX 5080 two at a time (≈56 min per pair). `checkpoint_init` saved before the first SGD step and every periodic checkpoint kept; all eight init/final checkpoints load as RLModules; 252 = 180 final + 72 init evaluations on the six HZ cells × certified windows k=26/34/42 (stochastic decode), 0 failed. Reader `stage_d_health_verdict.py`; artefacts in `reports/manifests/stage_d/run2/`.

## 1. Health gate

| item | result |
|---|---|
| verdict | **PASS_HEALTH** |
| wiring failures | none |
| substantive failures | none |
| rows | 252/252 |
| defer rate, final checkpoint, clean tier (gate: 2%–98%) | N_V 3.8%, V 6.3%, N_E 10.3%, E 9.8% |
| contract on every clean deployment (completion ≥ 0.995, ontime ≥ 0.995, forced 0) | green on all 72 rows |
| reward and carbon same direction init → final | V carbon 0.0089→0.0073, reward -142.9→-110.4; E carbon 0.0095→0.0080, reward -154.9→-124.1 |
| forecast sensitivity (L1 shift of action probabilities, forecast keys only, eps 0.25) | V 0.0234 (control 0.0287), E 0.0366 (control 0.0277) |
| argmax flip rate under forecast perturbation | V 3.0%, E 22.1% |
| EU-CRD Δr spread (proxy: rho_routing_std / reweight_w_std; dr_std now logged for future runs) | N_E 0.046, E 0.049; Δr mean ≈ 0 as expected |
| carbon-normalisation clip | 0 on all rows |

## 2. Pooled deployment readings (final checkpoint; carbon = sum over 18 runs, kg; not an effect claim)

| line | tier | defer rate | carbon sum | mean reward |
|---|---|---|---|---|
| NV | hollow | 3.8% | 0.1333 | -112.8 |
| V | godeye | 6.3% | 0.1311 | -110.4 |
| V | calibrated_shrink_v1 | 5.7% | 0.1297 | -108.9 |
| V | shuffle | 5.7% | 0.1336 | -113.2 |
| V | anti | 6.1% | 0.1301 | -109.3 |
| NE | hollow | 10.3% | 0.1344 | -114.0 |
| E | godeye | 9.8% | 0.1435 | -124.1 |
| E | calibrated_shrink_v1 | 9.1% | 0.1423 | -122.8 |
| E | shuffle | 10.0% | 0.1442 | -124.9 |
| E | anti | 10.0% | 0.1471 | -128.2 |

## 3. Reading (what the smoke says and does not say)

- **Pipeline is proven end to end**: training with init/final checkpoints, deployment through the perturbed-forecast provider (all four tiers), the ledger-aligned reward, the certified-window evaluation harness, the probe and the mechanical reader. The long run can use exactly this chain.
- **Policies are alive but early**: after 7 PPO iterations every line still has per-slot entropy at ≈97% of uniform, top-1 probability ≈0.22–0.27, defer rates 4–10%. Rewards and carbon moved together from init to final on every line.
- **The forecast channel is connected but barely used yet**: forecast-key perturbations move V's action probabilities by L1 0.023 (control keys 0.029) and E's by 0.037 (control 0.028), argmax flips 3% (V) and 22% (E). Consistently, V's carbon under the calibrated error (0.1297) is within 1% of clean (0.1311); the same for E. This is the expected state at 56k steps and is why the smoke is not an effect measurement: gates 1–5 belong to the long run.
- **EU-CRD internals are active**: Δr and ΔQ have non-zero spread, reweighting has spread, the responsibility shares are not pinned on the routing/forecast channels (rho_scheduling sits at its floor because the local layer has no reward mass under this reward, which is structural).
- Numbers here may not be quoted as results. In particular, E's higher carbon sum than V's (0.1435 vs 0.1311) at 56k is not evidence about EU-CRD.

## 4. Disclosed wiring fixes made during run 2 (append-only, no retraining)

Eval blocks for `calibrated_shrink_v1` lacked the audit parameter path; the RLlib evaluation loop lacked reward/defer accounting; six evaluation workers oversubscribed the CPU until each process was pinned to one BLAS/torch thread; the reader looked for `_last/_first` directories where the runner writes `_final/_init`, and one level too shallow for the trial's result.json; the probe needed absolute checkpoint paths. Each fix is committed; the frozen checkpoints were never touched.
