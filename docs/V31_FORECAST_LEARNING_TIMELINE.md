# V3 oracle forecast-learning timeline probe

Probe: `drl-manager/probe_forecast_sensitivity.py`, 40 synthetic observation trials per
checkpoint, seed 31001. The probe loads only the global RLModule on CPU; it does not launch
the simulator or Java gateway.

## Recoverable checkpoints

The configured checkpoint retention left only ck8, ck9, and ck10 for both `v3_oracle_s1`
and `v3_oracle_s2`. A repository and `/tmp` search found no ck1–ck7 directories for either
training run. Therefore this is a **late-stage scan**, not the requested complete ck1–ck10
trajectory; the evidence cannot distinguish early sign formation from a drift that happened
before ck8.

| Seed | Checkpoint | P(defer \| arriving) | P(defer \| leaving) | Delta (want >0) | Forecast TV | Forecast/control |
|---|---:|---:|---:|---:|---:|---:|
| s1 | 8 | 0.003312 | 0.015763 | **-0.012451** | 0.4956 | 0.6200 |
| s1 | 9 | 0.002993 | 0.015179 | **-0.012186** | 0.4786 | 0.5925 |
| s1 | 10 | 0.002493 | 0.011233 | **-0.008740** | 0.4427 | 0.5456 |
| s2 | 8 | 0.007822 | 0.028893 | **-0.021071** | 0.4805 | 0.6375 |
| s2 | 9 | 0.007393 | 0.024226 | **-0.016833** | 0.3721 | 0.4814 |
| s2 | 10 | 0.008357 | 0.028900 | **-0.020543** | 0.3838 | 0.4772 |

## Interpretation

- The temporal sign is negative in all 6/6 recoverable checkpoints and both seeds. The
  late-stage policy consistently defers more when green is leaving than when it is arriving.
- The forecast channel is not inert: its spatial total-variation sensitivity is roughly
  48–64% of the current-green control channel and far above the null channel. The late-stage
  failure is therefore best described as **forecast read strongly but mapped to the wrong
  temporal behavior**, not “forecast never reaches the action.”
- There is no sign flip within ck8–ck10. This rules out a last-two-checkpoint reversal, but
  not an earlier reversal. Recovering the true learning onset requires rerunning training
  with ck1–ck10 retention or saving the temporal probe at every checkpoint.

Raw JSON outputs are under `artifacts/v31/v3_oracle_checkpoint_probe/` in the implementation
worktree.
