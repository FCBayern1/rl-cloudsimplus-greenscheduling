# EU-CRD repaired forecast signal: instrument and wiring gates before any training spend (2026-09-07)

Status: G1–G4 criteria were fixed in the header of `g1/compressed_timecap_s2/eucrd_signal_gate.py` before that gate was launched (19:00) and are restated here unchanged. G5 is frozen by the commit that adds this file, before any G5 run. No carbon reading, no policy comparison and no tuning happens under this document. The frozen RL_V2 verdict (`FAIL_DET_SMOKE`) is untouched.

## 0. Why

The RL_V2 lines trained on `perturb_tier: godeye`, which returns the truth exactly, and the historical responsibility term compares only the current step, so `R_forecast ≡ 0` was the expected consequence of the protocol, not a wiring fault (measured, design log §61 correction). Two things must now be shown separately, and neither is evidence for the other:

- the repaired source (`crd.forecast.source: candidate_cover_mae`) is a correct **instrument** (G1–G4);
- EU-CRD actually **acts** on it in the real learner path, after the anomaly gate, the scale normalisation, the 450-call reweight warmup and the shrink guardrail (G5).

## 1. G1–G4, zero training, instrument only (running)

Arm `cover_argmax` on the RL_V2 evaluation twin, tiers godeye, shrink75, shrink50, shrink25, shrink0, the first three reading windows, one run per (tier, window) with the repaired source and one identical control run with the historical source.

- **G1** godeye: max per-step signal ≤ 1e-9.
- **G2** shrink75: mean per-step signal > 0 on every window.
- **G3** the tier means are non-decreasing along godeye, shrink75, shrink50, shrink25, shrink0.
- **G4** no observation leak: every policy-visible observation key is bit-identical between the repaired and the control run of the same (tier, window).

Any failure is STOP_SIGNAL_GATE and the source is not trained on. Statistics are reported over all steps and over decision steps only (steps with at least one job to place); the decision-step figure is the primary one, because the signal is zero by construction where there is nothing to decide.

## 2. G5, wiring in the real learner path (frozen here; runs after G1–G4 pass)

Two short trainings on the RL_V2 training twin, identical in every way except the training forecast tier, each 40 000 steps (5 PPO iterations; measured on the RL_V2 E line, the 450-call reweight warmup is passed between iteration 4 and 5, so 5 iterations is the smallest budget that exercises the applied reweighting):

| run | training tier | forecast source | credit |
|---|---|---|---|
| G5-err | shrink75 | candidate_cover_mae | EU-CRD enabled |
| G5-clean | godeye | candidate_cover_mae | EU-CRD enabled |

Read from the learner's own logged statistics, last iteration and the per-iteration series:

- **G5a** G5-err: `crd/r_forecast_abs_mean` > 0.
- **G5b** G5-err: `crd/rho_forecast_mean` ≥ 0.01 and `crd/rho_routing_mean` ≤ 0.99 (the forecast channel carries a non-decorative share; routing is not pinned at 1).
- **G5c** G5-err: `crd/reweight_applied` = 1 and `crd/reweight_w_std` ≥ 1e-3 at the last iteration. The bar is three orders of magnitude above the numerical noise measured in the godeye RL_V2 run (`reweight_w_std` 1.6e-7 … 2.2e-6 with `rho` uniform), so it separates "weights actually differ" from floating-point spread.
- **G5d** G5-clean: `crd/r_forecast_abs_mean` = 0, `crd/rho_forecast_mean` = 0 and `crd/reweight_w_std` ≤ 1e-4 (a correct forecast must still produce no forecast credit and uniform weights).
- **G5e** both runs complete without a crash and their contracts on the training episodes are not degraded relative to the RL_V2 lines (reported, not gated: these are 40 000-step runs, not a performance claim).

Verdict WIRING_GATE_PASS iff G5a–G5d hold. Any failure is STOP_EUCRD_WIRING: the matched V_err / E_err training is not started, and the failing statistic is reported for a ruling. Nothing is tuned to obtain a pass; in particular the warmup, the anomaly gate, ρ_min, the shrink strength and the gain are not touched.

## 3. What passing does and does not license

Passing G1–G5 licenses exactly one thing: starting the matched pair V_err (vanilla) and E_err (EU-CRD), trained on the same frozen imperfect forecast with the same data and budget, and comparing their degradation along the error ladder under deterministic deployment. It does not license any claim about EU-CRD's effect, and it does not reinterpret the RL_V2 result. The 2020 confirmation windows stay sealed.
