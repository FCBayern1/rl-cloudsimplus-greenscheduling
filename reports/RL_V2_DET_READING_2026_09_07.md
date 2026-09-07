# RL_V2 deterministic deployment: reading (2026-09-07, 18:03)

Preregistration: reports/RL_V2_DET_PREREG.md, frozen at 58240a28 with the four checkpoint hashes recorded before any reading. Run `stage_a_out/rl_v2/last_det`, archive `reports/manifests/rl_v2/det`. Verdict: **FAIL_DET_SMOKE on gate 5 (EU-CRD keeps more)**. Gates 1, 2, 3, 4 and 6 pass; gate 7's internals show the forecast-credit channel was inert, which qualifies the gate-5 reading.

## 1. Readings (six windows never trained on, deterministic decode, pooled carbon in kg)

| arm | carbon | capture of the causal expert's headroom | vs its own clean |
|---|---|---|---|
| NV clean (no forecast) | 0.027234 | −0.160 | |
| **V clean** | **0.009756** | **1.002** | |
| V shrink 0.75 | 0.012460 | 0.822 | +27.7 % |
| V shrink 0.5 | 0.016644 | 0.544 | +70.6 % |
| V shrink 0.25 | 0.023541 | 0.085 | +141.3 % |
| V shrink 0 | 0.023839 | 0.066 | +144.4 % |
| V shuffle | 0.026621 | −0.119 | +172.9 % |
| V anti | 0.030303 | −0.364 | +210.6 % |
| NE clean (no forecast) | 0.024549 | 0.018 | |
| **E clean** | **0.010125** | **0.978** | |
| E shrink 0.75 | 0.013255 | 0.770 | +30.9 % |
| E shrink 0.5 | 0.019106 | 0.380 | +88.7 % |
| E shrink 0.25 | 0.027711 | −0.192 | +173.7 % |
| E shrink 0 | 0.030006 | −0.344 | +196.4 % |
| E shuffle | 0.026548 | −0.115 | +162.2 % |
| E anti | 0.032032 | −0.479 | +216.4 % |

References on the same windows: causal expert 0.009789, offline flat planner 0.024826, `cover_argmax` clean 0.009634 (capture 1.010) and under shrink 0.75 0.011515 (+19.5 %), `cover_argmax` on the hollow channel 0.021295 (capture 0.235). Contracts green on all 96 readings; all four last checkpoints loaded and their hashes matched the record.

## 2. Gates

1. Init identity and contracts: pass (210/210 decisions and exact carbon per line at init; 96/96 readings contract-green).
2. Prior preserved: **pass.** V's clean capture 1.002 is 0.992 of `cover_argmax`'s 1.010 (bar 0.80). 56 000 PPO steps with a fixed cover prior and a zero-initialised residual leave the rule intact when deployed deterministically. E's 0.978 is 0.976 of V's.
3. The shrink hurts: **pass.** V loses 27.7 % of its clean carbon at λ = 0.75 (bar 5 %), and the zero-parameter rule loses 19.5 % on the same tier, so the loss is not an artefact of training: the policy loses about 8 points more than the rule it started from.
4. EU-CRD keeps more: **fail.** E's λ = 0.75 loss is 30.9 % against a bar of 0.5 × 27.7 % = 13.8 %. EU-CRD does not reduce the degradation; it is 3 points worse than vanilla, and worse at every rung except shuffle (E 162.2 % vs V 172.9 %).
5. Not by ignoring the forecast: pass. E's clean carbon is 58.8 % below NE's, and its action marginals differ between godeye and shrink 0.75 (KL 0.114).
6. EU-CRD internals: **the forecast credit channel carried nothing, and that was expected under this training protocol.** Last-iteration statistics of both credit lines: `rho_forecast_mean` 0.000 and `r_forecast_abs_mean` 0.000, `rho_routing_mean` 1.000 with standard deviation 0.000, `reweight_w_max` 1.000 with standard deviation 0.000 (uniform reweighting, a no-op), Δr mean −0.0002 with spread ±0.013, Δq mean −0.06 with spread ±1.1, τ 0.13. The ensemble and the Q-head ran.

   **Correction (2026-09-07, after checking the cause).** An earlier version of this section attributed the zero to the recorded ρ-saturation defect. That was over-attribution. All four lines trained with `perturb_tier: godeye`, and that tier returns the truth exactly (verified: maximum difference 0.0 between the tier's future and the wind file). The historical forecast term compares the predicted and the realised green of the **current** step, so under god's-eye training it is identically zero by construction, not by a wiring fault. A second, genuine gap compounds it: every shrink tier preserves lead 0 exactly (verified), while the offset policy acts on candidate coverage over the next 0–72 steps, so a current-step-only signal can stay zero even when the action-relevant part of the forecast is badly wrong. Both facts were established by direct measurement, not inference.

## 3. What this establishes

- Training does not break a correct prior. This was the smoke's first question and the answer is clean: with deterministic deployment the trained vanilla policy sits within 1 % of the zero-parameter rule it was initialised from, on six windows it never saw, with every contract green.
- Forecast degradation hurts the trained policy, monotonically along the ladder, and slightly more than it hurts the rule (27.7 % vs 19.5 % at the mildest rung; 210 % vs 201 % at anti). The chain's second question is answered positively for the learned policy.
- EU-CRD, in this configuration, does not protect. Its degradation is marginally worse than vanilla's at nearly every rung, and its clean performance is 2 points below vanilla's. This is a substantive failure of gate 5 and the line stops here for a ruling, with nothing tuned.
- The gate-5 reading must be read together with gate 6's correction: EU-CRD only reshapes training gradients, and it was trained on a forecast that was exact, so it had no forecast error to attribute and no opportunity to learn to discount one. What the smoke establishes is therefore narrower than "credit assignment cannot help": **a policy trained only on a perfect forecast does not gain robustness to a corruption it never saw, and EU-CRD in that protocol behaves like vanilla with slightly more variance.** Whether EU-CRD helps when the training forecast is itself imperfect is untested here, and needs a matched pair trained on the same frozen error.

## 4. Not done

No retuning, no gain change, no gate relaxation, no reuse of the smoke's stochastic readings, no multi-seed run, no 2020 window. The four checkpoints and all readings are archived.
