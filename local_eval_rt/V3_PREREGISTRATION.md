# v3 pre-registration (frozen 2026-08-10, before any v3 training)

Written before the first v3 run so the verdict cannot be re-negotiated after
seeing the numbers. Six earlier testbeds failed; the two recurring causes were a
weak blind baseline in the pre-check and a metric that inflated the headroom.

## Scenario and why each knob is there

Each lever removes one cushion identified in the 2026-08-10 autopsy of the
v2026-gamble testbed, where a blind policy reached oracle-level carbon.

| Lever | v2026 gamble | v3 | Cushion removed |
|---|---|---|---|
| Wind stretch | 10 (peak ~150 steps) | 6 (peak ~90 steps) | jobs shorter than peaks |
| Job length | median 55 steps | median 149 steps | "any peak fits any job" |
| Green-DC brown intensity | 0.08 (Nordic) | 0.55 | cheap-brown floor made wrong timing nearly free |
| Blind arm reward | window carbon on true future | persistence | training-time future leak |

## Analytic pre-check (already run)

`scan_voi_v3.py`, pooled absolute carbon over the 10 deployment windows,
clairvoyant against the best of {drain, reactive x2 thresholds, climatology
planner, peak-duration-hazard x3 thresholds}: **VoI = 78-84%** on the chosen
cell, against a 30% gate. The historical 73.9% figure that misled the gamble
campaign came from averaging per-window ratios with near-zero denominators, and
is not used here.

## Pre-registered verdict criteria (2 seeds per arm, deterministic decoding)

PASS requires all three:

1. **Effect size.** Median clean carbon of oracle is at least 20% below the
   blind arm, i.e. RL realises at least a quarter of the analytic VoI.
2. **Iso-completion.** Both arms complete at least 99% of the workload on the
   seeds entering the median, so the gap is not a dropped-work artefact.
3. **Not a basin lottery.** Both blind seeds land in the same regime. If one
   blind seed reaches oracle level and the other does not, the result is
   recorded as a basin split, not as forecast value, and two more seeds are
   required before any claim.

FAIL on any of the three means v3 is archived as characterisation and the paper
proceeds on C-regime with the reframed motivation (no clean upside, large
corruption downside).

## What PASS unlocks, in order

1. Out-of-band replication on a 2020 wind band selected by deployment-level
   discriminative power, not by surface statistics.
2. TimeCAP arm (realistic forecaster) and the corruption suite (blend, shuffle,
   inversion, severity sweep) through the timecap path.
3. EU-CRD training arms, making v3 the testbed where forecast trust is
   load-bearing.

## Notes

- Deadline drop-mode is deliberately NOT part of v3. It does not change the
  optima (starting at the latest admissible time equals the forced fallback), so
  it is a training-shaping device to be considered only after the verdict.
- The power divisor is calibrated empirically at the head of the chain from a
  measured-demand smoke, never extrapolated from the gamble configuration.
