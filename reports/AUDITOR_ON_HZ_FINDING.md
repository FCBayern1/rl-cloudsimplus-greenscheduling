# The deployment-time forecast auditor on the HZ scene: detects phase errors, blind to amplitude (2026-09-04)

Measured on the Stage D health-smoke EU-CRD checkpoint (`logs/stage_d/E_s20260903`, 56k steps), judgement-window evaluation blocks, stochastic decode, one cell per table, the same seed and window across tiers. Not a verdict on the method: the checkpoint is immature and its Q-ensemble has not specialised. The amplitude result, however, follows from the statistic's definition and will not change with training.

## 1. Detection

Cell c5_n20 for the Q-ensemble sentinel, c1_n50 for the residual monitor; 237–2387 logged decisions per condition.

| signal | clean | calibrated_shrink_v1 | anti |
|---|---|---|---|
| Q-ensemble disagreement σ² (mean) | 2.1146 | 2.1183 | 2.1209 |
| σ² above the clean p95 | 5% by definition | 10.1% | 10.1% |
| forecast-vs-realised correlation (mean) | 0.417 | 0.418 | 0.247 |
| correlation 5th percentile | −0.003 | 0.031 | −0.348 |

The Q-ensemble sentinel does not separate the conditions: the distributions overlap and the calibrated error and the sign-inverting error produce the same number, although one is far more extreme than the other. This reproduces the Phase-1 finding already recorded in the monitor's own docstring, which is why the residual monitor exists.

The residual monitor separates `anti` clearly and is blind to `calibrated_shrink_v1`. That is a property of its statistic, not of this checkpoint: a rolling Pearson correlation between forecast and realised green measures shape agreement, and an amplitude shrink toward the mean leaves the shape intact. The monitor's repair mode, which inverts a strongly anti-correlated site's forecast, is likewise aimed at sign inversion.

## 2. Does gating help carbon

Threshold 0.209, half of the clean mean correlation, as the monitor's own relative rule prescribes. Same cell, window, seed and checkpoint; the only difference is whether the gate may suppress the DEFER logit.

| corruption | carbon, no gate | carbon, gated | change | gate fired |
|---|---|---|---|---|
| calibrated_shrink_v1 | 0.01052 | 0.01052 | 0.0% | 21.9% of decisions |
| anti | 0.01198 | 0.01151 | −3.9% | 38.8% of decisions |

Under the primary corruption the gate fires on a fifth of decisions without changing carbon at all: those firings are false alarms, since the statistic cannot see the error that is actually present. Under the sign-inverting control it fires more often and buys 3.9%.

## 3. Consequence for the paper

The auditor as implemented is a phase-error detector. The preregistered primary corruption is an amplitude error measured from the deployed forecaster. Reporting the auditor as a working component would therefore rest on the negative control rather than on the primary error, which is exactly the substitution a reviewer looks for. Options, in the order I would take them:

1. Move the auditor to future work and state the finding: training-time credit assignment preserves the quality of forecast trust but cannot gate it, and the deployment-time monitor we tested gates phase errors and is blind to amplitude errors. This is honest, informative, and costs nothing.
2. Add a scale-sensitive statistic (ratio of forecast to realised means, or the slope of a rolling regression) alongside the correlation, gate on either. This is a new method component: it needs its own preregistration, its own clean-side calibration, and evidence that it does not fire on clean forecasts.
3. Keep it in the main text on the strength of the `anti` result alone. Not advisable.

## 4. To redo when the 400k checkpoints exist

The Q-ensemble sentinel on a mature ensemble, since σ² may separate once the heads have specialised. The correlation result needs no repeat.
