# Why the forecast does not move carbon here, and where its value actually lives

Date 2026-09-03. Zero-training analysis on the Scheme 2-H window (k=2, offset 2018, DCs Nordic/Germany/US-East with turbines, brown factor uniform 0.5). Tool: `g1/compressed_timecap_s2/lever_decomp.py` (tests in `test_lever_decomp.py`). Every number below is computed on the green truth alone, with the host floor treated as sunk (marginal accounting) and no competition between jobs (solo bound). Both choices are generous to the forecast: they remove the consolidation confound and the capacity confound. Whatever lever survives here is an upper bound on what any simulator run can show.

## 1. The decomposition

For each job (32 PE, drawing 132.7 W on a woken RS500A) the brown share of its own energy under three policies on the same truth:

- run-now: start at arrival on the greenest DC.
- myopic: start at the first row where some DC covers the job fully, else at the wait cap. Needs only the present.
- oracle: the least-brown start within the wait cap. Needs the forecast.

The forecast's information content is exactly myopic − oracle. Everything else the "defer" lever earns is available to a policy that only reads the current green.

## 2. Result on the H window (runtime 48 rows, wait cap 72 rows, 210 jobs)

| divisor | jobs fully green at arrival | brown share run-now | myopic | oracle | forecast-only lever | wasted waits |
|---|---|---|---|---|---|---|
| ×1 | 81% | 3.6% | 3.8% | 2.0% | 1.8 pp | 9% |
| ×2 | 28% | 17.7% | 20.3% | 15.3% | 5.0 pp | 53% |
| ×4 | 1% | 53.6% | 54.8% | 50.3% | 4.5 pp | 86% |

Wasted wait: myopic waited the full cap and still ran partly brown. The oracle runs those jobs at once.

Counterfactuals on the same wind (job length 48 → 12 → 4 rows, wait cap 72 or 24 rows, three DCs or one DC): the forecast-only lever stays between 0.6 and 5.9 pp in all 18 combinations. Shortening the job does not open it. Removing the spatial hedge (one DC) adds at most 1 pp.

Wind autocorrelation on this window (rows of 10 min): lag 6 → 0.88, lag 24 → 0.53, lag 48 → 0.22, lag 120 → 0.11. The 48-row job already spans the decorrelation time, so its start time barely changes its average coverage. The 4-row job does not, yet the lever still does not open, which points at the structure of the problem rather than at the time scales.

## 3. The structural reason

Waiting is free in every S2 variant: slack is wide, `deadline_forced_count` was 0 in all 54 pilot runs, completion 1.000 everywhere. When waiting is free, the policy "wait until a DC covers the job, else run at the cap" needs only the current green and captures the whole feast/famine gain. The forecast is left with one job: refining the start inside partially covered stretches. That refinement is bounded by the variance of the clipped coverage min(1, G/P) across the wait window, which is small for wind with this DC mix. Hence ≤ 6 pp under conditions more generous than any simulator run. In the simulator the floor is not sunk, so the same refinement also pays a fragmentation tax of +15…+25% energy (PILOT_H_REPORT §1), which is why godeye lost to the blind consolidator at ×2 and ×4.

This is the mechanism behind every earlier stop: TB13 (EVPI p50 5.57%), S2 confirmation (shuffle retains 106%, because lead-0 truth is all the policy ever used), E, F, and H. It is not that defer is a bad lever. It is that under free waiting, defer needs no forecast, and the forecast's residual value on carbon is a few percent of job energy.

## 4. Where the forecast is load-bearing

The table's last column. At ×2, 53% of myopic waits are wasted: the job waited 72 rows (12 h of wind) for green that never came and ran brown anyway. At ×4, 86%. Under a binding deadline each of those is a miss, or a forced brown run at the last moment, which clusters forced runs into the worst rows. The oracle avoids all of them by running at arrival. A wrong forecast produces the mirror error: run now when green was about to arrive, wait when it was not. So the forecast's value is a decision about **whether** to wait, and it exists only when waiting has a price. That price is the completion/SLA axis, with carbon coupled through forced runs.

This is exactly the rwtight finding (memory: blind wait 92.6% missed, anti-forecast −10.9 pp completion and +14% carbon, EU-CRD +1.9 pp), and exactly the axis every S2 variant designed away by setting slack so that forced = 0.

## 5. What to do

1. Stop searching for a carbon-only forecast lever in this simulator family. The zero-training bound is ≤ 6 pp and the simulator adds a fragmentation tax on top.
2. Register the target scenario as "costly waiting": scarcity ×2 (28% of jobs fully green at arrival, 53% wasted-wait rate), jobs shorter than the wind decorrelation time (≤ 12 rows), slack of the order of the decorrelation time (24…72 rows), deadline binding with `latest_start` force and a counted miss. Blind arms: run-now and myopic (lead-0 wait). Forecast arms: godeye, calibrated_shrink, anti, shuffle. Verdict on the joint (deadline miss, carbon) frontier at iso-completion, with forced brown runs reported separately.
3. Gate before any simulator run, with this script extended for per-DC capacity: forecast-only lever on misses ≥ 30 pp of wasted waits recovered, and shuffle/anti worse than myopic by more than the noise floor. The script runs in seconds. No RL until the gate passes.
4. Keep the corrected power model (RS500A 51.4/214 W) and the S2 tooling (perturbed oracle, TIERS_E, blind family, window schedule). This is rwtight rebuilt on the current simulator, not a new family.
5. The paper claim then reads: forecast error makes waiting-type carbon-aware schedulers miss deadlines and cluster forced brown runs; vanilla RL inherits this; EU-CRD resists it. Carbon is reported as a secondary iso-completion metric.
