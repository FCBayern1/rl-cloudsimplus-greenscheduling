# Prior-collapse probe SGD1 — invalidated

The run under `drl-manager/logs/prior_collapse_probe_sgd1/` is not an
interpretable first-iteration probe.  Its environment was constructed with
the default dyadic offset grid (`[0,1,2,4,8,16,32,64,72]`, 9 offsets), so the
global action space was `5 × 9 = 45`.  The frozen prior-gate experiment and
all policy-degradation comparisons use the dense certified grid `0..72`
(73 offsets, 365 actions).

The mismatch was detected before any carbon or prior-collapse readout.  No
number from this run is evidence and it must not be compared with the prior
gate.  The run is retained only as an implementation check.  A replacement
probe must launch with `OFFSET_GRID_DENSE=1`, and its manifest must record the
resolved action-space cardinality and observation shape before training.
