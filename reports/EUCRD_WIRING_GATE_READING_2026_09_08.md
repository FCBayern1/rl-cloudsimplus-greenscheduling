# G5 wiring gate on the regret source: STOP (2026-09-08, 02:53)

Preregistration reports/EUCRD_SIGNAL_GATE_PREREG.md §2 with Addendum A, thresholds frozen at 12e5e96e (G5a–c, G5e) and the Addendum's narrowing of G5d frozen at 77f476cc, all before these runs. Two 40 000-step trainings differing only in the training forecast tier (`g5_err` shrink75, `g5_clean` godeye), both with EU-CRD on `candidate_carbon_regret`. Five iterations each, both exit 0. Verdict **STOP_EUCRD_WIRING:G5b_shares_not_pinned**. The matched V_err / E_err pair does not start. No threshold was moved and nothing was tuned.

## 1. Gates, read on the last iteration as registered

| gate | requirement | err run, iteration 5 | |
|---|---|---|---|
| G5a | `r_forecast_abs_mean` > 0 | 8.41e-7 | pass |
| G5b | `rho_forecast_mean` ≥ 0.01 | **0.008745** | **fail** |
| G5b | `rho_routing_mean` ≤ 0.99 | **0.991490** | **fail** |
| G5c | `reweight_applied` = 1 and `reweight_w_std` ≥ 1e-3 | 1.0 and 0.0877 | pass |
| G5d | clean run: `r_forecast_abs_mean` = 0 and `rho_forecast_mean` = 0 | 0.0 and 0.0 | pass |

G5e (both runs complete without a crash): both exit 0, five iterations each; 40 000 steps is a wiring budget and no statistic here is evidence about learning or carbon.

## 2. The per-iteration series, which is where the failure comes from

| iteration | ρ_forecast | ρ_routing | reweight applied | ρ_forecast on firing cells | anomaly gate pass |
|---|---|---|---|---|---|
| 1 | 0.01307 | 0.98711 | 0 (warmup) | — | — |
| 2 | 0.01294 | 0.98728 | 0 | 0.3266 | 0.403 |
| 3 | 0.01368 | 0.98654 | 0 | 0.4563 | 0.492 |
| 4 | 0.01432 | 0.98600 | 0 | 0.4821 | 0.534 |
| 5 | **0.00875** | **0.99149** | **1** | 0.2586 | 0.286 |

Iterations 1–4 clear the threshold; iteration 5, the first one past the 450-call warmup and therefore the first where the weights are actually applied, does not. The gate is registered on the last iteration, so the run fails.

## 3. What the batch mean is measuring here

The signal is sparse by construction: at iteration 5 it is non-zero on 70 of 2 070 valid transitions (3.4 %), because at most decision states the forecast does not change which candidate is chosen. Where it does fire, the forecast channel takes **26 %** of the responsibility (46–48 % at iterations 3 and 4). A batch mean over a 96.6 %-zero column cannot separate "the channel is decorative" from "the channel dominates exactly where the forecast mattered", and 0.0087 is what the second case looks like under this statistic.

Two further measurements say the small mean is not a unit problem and is partly a gate-of-a-gate problem:

- After scale normalisation the channels are within a factor of two (`abs_f_scaled_mean` 0.578, `abs_r_scaled_mean` 1.200), so the historical ρ-saturation defect — incommensurate units letting routing swamp the rest — is not what is happening.
- The anomaly gate passes only **28.6 %** of the firing cells at iteration 5 (40–53 % earlier). It runs on the raw signal and keeps the anomalous part, so with a signal that is already sparse it removes two thirds of what remains.

Reported under the ruling of 2026-09-07, not gated: the applied weight split by the sign of the advantage it multiplies is 1.00168 on positive advantages and 0.99741 on negative ones — a 0.26 % damping of the corrective direction, essentially symmetric, with no sign of the corrective signal being suppressed at this budget. The clean control's weights are uniform to 1.7e-6 with the forecast channel silent, which is consistent with, but not required by, a correct forecast.

## 4. What this does and does not establish

Established: with the regret source selected, forecast credit reaches the learner (G5a), the reweighting actually runs and produces a real spread (G5c), a correct forecast produces none of it (G5d), and the share carried by the forecast channel, measured as a batch mean at the last iteration, is below the threshold registered for it (G5b).

Not established: that the mechanism is inert. That question is not answerable from a batch mean, and it is what the switch comparison (reports/EUCRD_SWITCH_AB_PREREG.md, frozen 140d79c5 before this verdict was read) asks directly — whether the weights and the policy gradient change at all, and whether they change more where the forecast changed a decision. It started automatically at 02:53.

## 5. Open question for a ruling

G5b was written as a batch mean before the signal's sparsity was measured. Three ways forward, none taken here: accept the STOP and close the line; re-register the share criterion on the conditional share (`rho_forecast_mean_firing`) with a number frozen before any re-run; or keep the batch-mean criterion and accept that under it a sparse-but-decisive channel reads as decorative. The anomaly gate's 28.6 % pass rate is a separate mechanism parameter and is not touched without its own ruling.
