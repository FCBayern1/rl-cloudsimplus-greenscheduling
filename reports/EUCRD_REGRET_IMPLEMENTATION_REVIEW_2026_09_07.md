# EU-CRD regret signal: independent implementation review

2026-09-07. Review of the in-progress shared worktree after registration
`1b935ca5`. This note does not amend that registration or authorize an experiment.
The existing implementation is being edited concurrently; no shared implementation
files were changed by this review. No training or simulation results were generated.

## Repair sequence

1. Define the signal on a fixed pre-decision reservation grid and legal set.
   Compare the forecast-greedy choice with the truth-greedy choice, and settle
   both under truth. This is local model regret of the forecast-greedy rule,
   not necessarily regret of the sampled RL action or total trajectory carbon.
2. Validate truth-zero, non-negativity, equal-cost ties, hand-calculated loss,
   empty batches and malformed inputs. Freeze any necessary definition amendments
   before simulation measurements.
3. Carry the scalar only through learner auxiliary data. Explicitly selected
   candidate sources must raise for missing, malformed or nonfinite data instead
   of silently falling back to current-step forecast error.
4. Check the actual loss path: raw signal, anomaly-filtered signal, responsibility
   share, and applied weights after warmup and the guardrail. Inspect positive and
   negative advantages separately: suppressing a negative advantage can suppress
   learning to avoid a bad action. Nonuniform weights alone do not prove benefit.
5. Freeze matched Vanilla/EU-CRD imperfect-forecast training with identical data,
   forecast exposure, prior, action space, budget and paired seeds. Compare clean
   performance and corrupted performance against both the paired learner and the
   zero-parameter rule. The old clean-trained result remains a separate result.

## Concrete issues found before running

### Carbon definition

The current function and registration use `(1-cover) * E * brown_factor`.
That is uncovered dynamic-energy brown carbon. With nonzero green carbon it is
not total dynamic carbon. For the current coverage abstraction, the latter is

`E * (brown_factor * (1-cover) + green_factor * cover)`.

Under common site factors 0.5/0.01, the choice ordering is unchanged, but the
regret uses 0.49 rather than 0.5. For heterogeneous factors, even candidate
ordering can change. Before claiming equivalence to `cover_argmax`, either
validate the homogeneous-factor assumptions and common tie rule or explicitly
describe a different reference policy. Host overhead and interactions between
same-batch jobs remain outside this marginal dynamic-energy abstraction.

The registered hand example was evaluated directly with the current function:

- energy = `64 * 2.02 * 10 / 3.6e6` kWh;
- brown-only regret = `0.00010773333333333334` kg;
- dynamic-carbon regret including green factor 0.01 = `0.00010557866666666665` kg.

The rounded registration literal `1.0774e-4` differs from the brown-only result
by `6.66666666665418e-9`, exceeding its stated `1e-12` assertion tolerance.
Tests should evaluate the chosen formula without rounding the expected value.
Resolve terminology/formula with an append-only amendment before measurement.

### State and batching

Identical job IDs and decision times do not make states identical if reservations
differ. Fixed-state comparisons must freeze reservations and the legal mask too.
Per-job regret summed over a batch does not model competition among those jobs;
record the approximation and valid decision count. An empty-grid diagnostic that
retains a trajectory-derived mask is still partially trajectory-dependent.

### Privileged supervision and training

Hidden future truth used to compute this scalar is privileged training
supervision, even when actor observations are unchanged. Forecast-greedy regret
can be zero despite forecast errors if the selected action stays optimal; neither
strict positivity nor tier monotonicity is a universal mathematical requirement.

The existing G5 budget is two 40k-step training runs, not zero training.
Perfect forecasts imply zero forecast error, not automatically uniform total
credit weights when other responsibility channels vary.

## Verification performed

Read the environment feature builder, the candidate signal functions, the
learner fallback and responsibility paths, and the new registration. Executed
the hand example as a pure function; no carbon experiment, learner update,
checkpoint change or sealed-window access was performed.
