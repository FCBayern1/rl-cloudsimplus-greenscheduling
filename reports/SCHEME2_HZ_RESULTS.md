# Scheme 2-HZ formal results: DISCOVERY and CONFIRMATION (2026-09-03)

Prereg `reports/SCHEME2_HZ_PREREG.md` (frozen at 47a025d7, Addendum A on the reader). Scene: accelerated-weather, marginal-carbon mechanism positive control (zero-floor hosts, ×2 scarcity, 32-PE 48-row jobs, six cells). Arms after the blind freeze: frozen blind, truth-informed planner (clean), calibrated_shrink_v1 (primary realistic error), shuffle, anti. Every verdict quantity comes from `hz_verdict.py`; the JSONs, the blind freeze, the phase manifest and a SHA256 list of all 162 run outputs are under `reports/manifests/hz/`.

## 1. Blind freeze (DISCOVERY, before any clean number existed)

| candidate | pooled carbon kg (mean over 18 runs) | contract failures |
|---|---|---|
| load_smoothing | 0.00686 | 0 |
| nowait_planner | 0.00509 | 0 |
| reactive_wait_planner | 0.00466 | 0 |
| reservation_edf | 0.00681 | 0 |

Frozen blind: **reactive_wait_planner**.

## 2. Verdicts

| set | verdict | grids valid | voided grid | G0 | G1 clean vs blind (pooled / median / cells / windows) | G2 shrink vs clean (pooled raise / cells / windows) | R_pool shrink | R_pool shuffle | R_pool anti |
|---|---|---|---|---|---|---|---|---|---|
| discovery | **PASS_HZ_DISCOVERY** | 18/18 | — | True | −28.8% / −34.4% / 6/6 / 3/3 | +101.4% / 6/6 / 3/3 | -1.50 | -0.73 | -1.76 |
| confirmation | **PASS_HZ_CONFIRMATION** | 17/18 | [['s2_r48_w72_c5_n50', 42]] | True | −42.5% / −46.7% / 6/6 / 3/3 | +154.5% / 6/6 / 3/3 | -1.09 | -1.06 | -1.52 |

Pooled carbon intensity relative to the frozen blind:

| set | clean | shrink | shuffle | anti |
|---|---|---|---|---|
| discovery | -28.8% | +43.3% | +20.9% | +50.9% |
| confirmation | -42.5% | +46.2% | +45.0% | +64.5% |

## 3. Per-grid carbon intensity (kg per 1e9 MI), CONFIRMATION then DISCOVERY


### confirmation (windows k=[26, 34, 42])

| cell | k | blind | clean | shrink | shuffle | anti | ontime (blind/clean/shrink/shuffle/anti) |
|---|---|---|---|---|---|---|---|
| _c1_n20 | 26 | 0.026 | 0.008 | 0.051 | 0.048 | 0.059 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n20 | 34 | 0.044 | 0.022 | 0.038 | 0.026 | 0.049 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n20 | 42 | 0.058 | 0.027 | 0.069 | 0.087 | 0.083 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 26 | 0.032 | 0.009 | 0.039 | 0.038 | 0.062 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 34 | 0.049 | 0.018 | 0.073 | 0.057 | 0.059 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 42 | 0.086 | 0.071 | 0.104 | 0.133 | 0.120 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 26 | 0.043 | 0.019 | 0.098 | 0.087 | 0.094 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 34 | 0.025 | 0.011 | 0.024 | 0.008 | 0.018 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 42 | 0.020 | 0.021 | 0.032 | 0.055 | 0.060 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 26 | 0.029 | 0.012 | 0.075 | 0.057 | 0.065 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 34 | 0.056 | 0.032 | 0.061 | 0.057 | 0.065 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 42 | 0.042 | 0.032 | 0.056 | 0.080 | 0.094 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 26 | 0.059 | 0.037 | 0.105 | 0.083 | 0.092 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 34 | 0.028 | 0.018 | 0.030 | 0.016 | 0.028 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 42 | 0.006 | 0.005 | 0.005 | 0.008 | 0.008 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 26 | 0.040 | 0.021 | 0.091 | 0.069 | 0.085 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 34 | 0.018 | 0.010 | 0.014 | 0.028 | 0.035 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 42 | 0.044 | 0.042 | 0.056 | 0.084 | 0.097 | 1.00/1.00/0.98/1.00/1.00 |

### discovery (windows k=[2, 10, 18]; k=2 previously read)

| cell | k | blind | clean | shrink | shuffle | anti | ontime (blind/clean/shrink/shuffle/anti) |
|---|---|---|---|---|---|---|---|
| _c1_n20 | 2 | 0.038 | 0.021 | 0.077 | 0.064 | 0.070 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n20 | 10 | 0.058 | 0.031 | 0.051 | 0.070 | 0.086 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n20 | 18 | 0.086 | 0.039 | 0.124 | 0.087 | 0.100 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 2 | 0.049 | 0.025 | 0.078 | 0.069 | 0.085 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 10 | 0.068 | 0.026 | 0.080 | 0.080 | 0.087 | 1.00/1.00/1.00/1.00/1.00 |
| _c1_n50 | 18 | 0.063 | 0.031 | 0.087 | 0.075 | 0.080 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 2 | 0.023 | 0.022 | 0.097 | 0.036 | 0.083 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 10 | 0.063 | 0.056 | 0.069 | 0.096 | 0.128 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n20 | 18 | 0.108 | 0.089 | 0.154 | 0.122 | 0.144 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 2 | 0.064 | 0.061 | 0.107 | 0.080 | 0.118 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 10 | 0.064 | 0.039 | 0.065 | 0.075 | 0.099 | 1.00/1.00/1.00/1.00/1.00 |
| _c3_n50 | 18 | 0.085 | 0.056 | 0.123 | 0.089 | 0.106 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 2 | 0.038 | 0.023 | 0.077 | 0.058 | 0.104 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 10 | 0.048 | 0.032 | 0.048 | 0.066 | 0.083 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n20 | 18 | 0.097 | 0.090 | 0.161 | 0.130 | 0.135 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 2 | 0.073 | 0.068 | 0.120 | 0.085 | 0.124 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 10 | 0.077 | 0.058 | 0.079 | 0.093 | 0.108 | 1.00/1.00/1.00/1.00/1.00 |
| _c5_n50 | 18 | 0.105 | 0.096 | 0.168 | 0.121 | 0.138 | 1.00/1.00/1.00/1.00/1.00 |

## 4. Reading

- **Both sets PASS all four gates under the registered rule.** On CONFIRMATION (three never-before-read windows) the truth-informed planner cuts pooled carbon intensity by 42.5% against the frozen blind, in 6/6 cells and 3/3 windows; the calibrated TimeCAP-derived amplitude error raises it 154% above clean and 46% above the blind; shuffle and anti are 45% and 65% above the blind. Effects on CONFIRMATION are larger than on DISCOVERY, not smaller.
- **Strict reading.** One CONFIRMATION run, calibrated_shrink_v1 on c5_n50 / k=42, finished all 50 jobs with ontime 0.98 (one job late; the other four arms on that grid were 1.0). Under the registered G0 rule the grid is voided and the verdict rests on 17 grids with thresholds still out of 6 and 3. Under a strict "every run contract-green" reading, which the first build of the reader enforced and the prereg text did not register (Addendum A), CONFIRMATION would fail G0. The late job is itself an effect of the corrupted forecast on the deadline axis, and is reported as such rather than hidden.
- **What the primary error does.** The audited λ lead-curve pulls the forecast toward the site mean at long leads, so the planner books jobs into stretches it believes are windy and that are not; on this scene that is worse than reversing the forecast. This is the frozen primary corruption from the E prereg, unchanged.
- **Scope.** This is a mechanism positive control (1 W floor, accelerated weather). It establishes step 1 of the chain in a controlled scene; it does not by itself support a real-world saving claim. Next: Stage D prereg (vanilla / EU-CRD, clean and corrupt, matched no-forecast), then the spiral's second turn with the real 51.4 W floor.
