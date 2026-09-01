# Round1-v2 STOP_GENERATOR_EXHAUSTED

> Round1-v2 stopped at the zero-emissions preflight: 96 of 99 workload keys could be
> generated, and three exhausted the frozen budget of 64 retries. A source audit shows the
> generator squeezed the arrival span to a single epoch in order to fit the target
> concurrency, the full slack and a short fixed horizon at the same time, so the stop
> reflects incompatible axis-generation semantics and is not negative evidence about the
> value of a forecast.

The three keys share (horizon 36, pes_per_job 8, n_jobs 12, wait_cap 24) and differ only
in concurrency. In `workload_v2.draw` the arrival span is

    span = min(target_span, horizon - max(runtime) - wait_cap - 1)

which for these keys is 36 - 12 - 24 - 1 = -1, clamped to 1: every job arrives at epoch
zero. The tightest budget then forces them out together and they collide with capacity.

The remaining 96 keys are not a usable basis for v3. Clipping may have occurred for other
keys too and merely happened to survive the reservation gate, so the axis definition is
rebuilt in v3 rather than pruned. Phase A was not run and no carbon was read.
