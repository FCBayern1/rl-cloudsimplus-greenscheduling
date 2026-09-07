# Forecast responsibility as local decision regret: H1–H5 reading (2026-09-07, 21:19)

Preregistration reports/EUCRD_REGRET_SIGNAL_PREREG.md (frozen 1b935ca5, Addendum A 77f476cc, both before any run of this signal). 45 zero-training runs in `stage_a_out/eucrd_regret`: 5 tiers × 3 windows × {regret source, source-only control}, plus 5 tiers × 3 windows replaying the godeye arm's own schedule. Arm `cover_argmax`, `COVER_TIE=index`, `OFFSET_GRID_DENSE=1`. No carbon claim, no policy comparison, nothing tuned. Verdict **REGRET_SIGNAL_PASS**; G5 started automatically at 21:19.

## 1. Gates

| gate | requirement | result |
|---|---|---|
| H1 | godeye max signal ≤ 1e-12 | **pass** — exactly 0 at all 1921 steps of the three windows |
| H2 | shrink75 mean > 0 on every window | **pass** — 1.29e-5, 2.07e-5, 5.04e-5 kg per decision step |
| H3 | min signal ≥ 0 everywhere | **pass** — 0 across all 30 runs, both phases |
| H4 | on the godeye trajectory, every degraded tier > godeye's zero | **pass** — 3.0e-5, 1.27e-4, 1.59e-4, 1.86e-4 kg |
| H5 | policy observation bit-identical to the control, no `crd_*` key in it | **pass** — 0 violations over 15 pairs |

## 2. Signal by tier, kg CO₂ per decision step (105 decision states per tier, 35 per window)

| tier | own trajectory | godeye trajectory (H4) | decisions changed by the forecast | empty-grid cover MAE |
|---|---|---|---|---|
| godeye | 0 | 0 | 0/105 | 0 |
| shrink 0.75 | 2.797e-5 | 2.993e-5 | 59/105 | 0.130 |
| shrink 0.50 | 6.672e-5 | 1.275e-4 | 79/105 | 0.164 |
| shrink 0.25 | 1.060e-4 | 1.592e-4 | 82/105 | 0.169 |
| shrink 0 | 1.715e-4 | 1.859e-4 | 89/105 | 0.169 |

Three things are worth stating precisely.

**The inversion that stopped the old signal is gone.** Both columns rise monotonically with the forecast error, including the last rung where the cover error fell (shrink 0 scored 0.179 against shrink 0.25's 0.207 under the old definition). Monotonicity was deliberately *not* gated here, so this is an observation, not a passed test: pricing the decision that changes hands is not affected by the trajectory compression that shrank the raw cover error, because a compressed reservation grid removes the opportunity from both the forecast's choice and the truth's choice at once.

**The two trajectories disagree in the middle of the ladder** — shrink 0.50 reads 6.7e-5 along its own trajectory and 1.27e-4 along godeye's. The signal is still a function of the state it is measured at, as it must be; what H4 fixes is that the comparison across tiers is now made at the same states.

**The auxiliary quality scale saturates where the regret does not.** Empty-grid cover MAE is identical at shrink 0.25 and shrink 0 (0.1686 both): below a quarter amplitude the candidate-cover error stops growing. The regret still rises 1.06e-4 → 1.72e-4 over the same step, because what changes is not how wrong the curve is but how often it misranks the candidates (82 → 89 of 105 decisions). This is the concrete case for pricing decisions rather than curves.

## 3. What this does and does not license

Established: the repaired source is zero under a correct forecast, positive and non-negative under a degraded one, monotone in the error at fixed states, and leaves the actor's observation untouched in these runs.

Not established: anything about EU-CRD's effect, anything about carbon, and any claim that the whole training procedure uses no future information — the truth curve is privileged supervision on the learner channel, and H5 speaks only to the actor's observation in these runs.

Recorded approximations (prereg Addendum A3): the per-job regrets are summed without modelling competition between same-batch jobs for the same residual green; the empty-grid diagnostic still uses the trajectory's legality mask.

The replay is verified, not assumed: 35 jobs per window with no repeated sighting and no masked start, and the replayed godeye run reproduces the online godeye run's episode carbon exactly in all three windows (6.06e-4, 1.696e-3, 4.913e-3 kg, relative difference 0). H4's "same state" is therefore the same state.

## 4. Next

G5 (wiring in the real learner path, two 40 000-step runs, a wiring budget and not a training budget) started on this source at 21:19. The matched V_err / E_err training stays blocked until it passes.
