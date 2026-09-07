# EU-CRD repair plan (draft; no carbon run authorised by this file)

Status: implementation preflight only. The frozen RL_V2 verdict remains
`FAIL_DET_SMOKE`; this document does not reinterpret or overwrite it.

## 1. What the failed smoke actually says

The trained forecast policy is useful and forecast degradation hurts it, but
the frozen EU-CRD line did not reduce that damage. Its training diagnostics
showed `R_forecast = rho_forecast = 0` and uniform responsibility weights.

That zero is not, by itself, proof of broken tensor wiring. RL_V2 trained both
forecast lines with `perturb_tier: godeye`, and the historical forecast term
compares realised current wind with predicted current wind. Under God's-eye
training those values are identical, so a zero term is the expected result.
Furthermore the perturbation ladder deliberately preserves lead 0, while the
offset policy acts on future candidate coverage. The old term can therefore
stay zero even when the future forecast driving the action is badly wrong.

There is also a protocol mismatch: EU-CRD only changes training gradients; it
has no inference-time correction. A model trained exclusively on perfect
forecasts cannot learn to quarantine forecast errors it never observes.

## 2. Config-gated implementation repair

Add `crd.forecast.source: candidate_cover_mae` (default remains
`instantaneous_carbon_cf`). For each decision state:

1. Compute the visible `cand_green_cover` from the arm's forecast exactly as
   before.
2. Compute the same candidate values on the simulator's hidden future, using
   the same legal mask, reservation grid, runtime and power model.
3. Put only the mean absolute error over legal candidates in learner-only
   `crd_aux`; do not put the truth curve or the error in the policy observation.
4. Use that scalar as `|R_forecast|` in the responsibility denominator.

This signal is zero for a correctly aligned God's-eye forecast and positive
when later leads alter the action-relevant coverage, even if lead 0 is exact.
The old source and old observation schema remain the default for checkpoint
compatibility.

## 3. Required preflight before training

- God's-eye: candidate-cover error is numerically zero.
- `shrink75`: lead 0 remains exact but candidate-cover error is non-zero.
- Error signal increases across the frozen shrink ladder in aggregate; no
  carbon metric is used to select or scale it.
- Policy inputs and deterministic actions are unchanged when only the
  learner-only source is enabled.
- With a synthetic routing term of equal magnitude, the new source produces
  `rho_forecast = rho_routing = 0.5` and non-uniform reweighting.
- Default configuration passes existing CRD tests unchanged.

Failure is `STOP_EUCRD_ATTRIBUTION_PREFLIGHT`; do not tune rho floors, caps or
scales to manufacture a non-zero share.

## 4. Matched repair smoke (requires a separate frozen addendum)

The fair smoke retrains a matched pair, not just EU-CRD:

- `V_err`: vanilla PPO, fixed cover prior, forecast-error exposure.
- `E_err`: identical inputs, windows, forecast-error exposure, optimiser,
  seed and budget; only EU-CRD is enabled.

Both lines must see the same prospectively fixed error schedule. A minimal
first smoke can use the already frozen mild rung `shrink75`; a stronger test
should alternate clean and `shrink75` windows by a hash rule fixed before any
training result. Evaluation remains deterministic on clean plus the complete
frozen ladder.

This changes the scientific claim from "clean-only training resists unseen
deployment corruption" to the defensible claim "given the same imperfect-
forecast training data, responsibility decomposition learns a policy that
degrades less than vanilla". Keeping the former claim would require a new
inference-time reliability gate, which EU-CRD currently does not implement.

Health gates before any long run:

- all execution/SLA contracts green;
- clean performance preserves at least 80% of matched vanilla's headroom;
- `R_forecast` has positive magnitude and variance on error-exposed batches;
- `rho_forecast` and `rho_routing` are both non-degenerate;
- applied responsibility weights have non-zero variance;
- EU-CRD's degradation at `shrink75` is lower than matched vanilla's on the
  paired read windows. A long-run effect threshold and seed count must be
  frozen separately; the one-seed smoke is only a mechanism gate.

## 5. Explicit non-fixes

- Do not force `rho_forecast` away from zero under a perfect forecast.
- Do not lower `rho_routing`, raise `rho_min`, or tune the shrink/cap values
  merely to make diagnostics look active.
- Do not compare an error-trained EU-CRD line with the existing clean-trained
  vanilla checkpoint as the causal headline comparison.
- Do not read the sealed 2020 confirmation windows during repair development.
