# EU-CRD repaired forecast signal: G1–G4 reading (2026-09-07, 19:26)

Preregistration: reports/EUCRD_SIGNAL_GATE_PREREG.md (G1–G4 criteria fixed in the gate script's header before launch, restated in that file; G5 frozen at 12e5e96e). Run `stage_a_out/eucrd_signal` (30 runs: 5 tiers × 3 windows × {repaired source, control}), arm `cover_argmax`, no training, no carbon claim. Verdict: **STOP_SIGNAL_GATE:G3_monotone_in_error**. G5 did not start; the runner refused it automatically.

## 1. Results

| tier | all-step mean | decision-step mean (primary) | max | windows |
|---|---|---|---|---|
| godeye | 0.000000 | 0.000000 | 0.00000 | 3 |
| shrink 0.75 | 0.007390 | 0.134419 | 0.41336 | 3 |
| shrink 0.5 | 0.010815 | 0.193546 | 0.52176 | 3 |
| shrink 0.25 | 0.011293 | **0.206928** | 0.53562 | 3 |
| shrink 0 | 0.010323 | **0.178736** | 0.53562 | 3 |

105 decision steps per tier (35 per window, identical (step, slot) sets across tiers).

- **G1 pass.** Under a correct forecast the signal is exactly 0 at every one of the 610+ steps of every window (max 0.0, not a tolerance).
- **G2 pass.** Under shrink 0.75 the mean is positive on every window (0.0061, 0.0089, 0.0071 all-step; 0.134 pooled on decision steps).
- **G3 fail.** The ordering breaks at the last rung on both statistics: shrink 0 scores below shrink 0.25 (0.179 vs 0.207 on decision steps, 0.0103 vs 0.0113 over all steps). The forecast with no amplitude at all produces a *smaller* measured error than a partially shrunk one.
- **G4 pass.** Zero observation-leak violations: with the repaired source and with the historical one, every policy-visible observation key is bit-identical on all 15 (tier, window) pairs.

## 2. Why G3 breaks: the signal is trajectory-dependent

Two checks separate the instrument from the comparison.

- On a **fixed** state set (the same decision states, an empty reservation grid, computed offline from the wind files and the ladder's own rung curves) the ordering is monotone: 0.0286, 0.0399, 0.0416, 0.0416 for λ = 0.75, 0.5, 0.25, 0 pooled over the three windows. The instrument itself orders the tiers correctly when the state is held fixed.
- The live runs share the same decision states but not the same **commitments**. The flatter the forecast, the earlier the arm dispatches: mean chosen offset κ is 48.6 steps at λ = 0.75, 34.9 at λ = 0.25 and 6.3 at λ = 0. Early, dense commitment consumes the residual green that later candidates could claim, which compresses both the predicted and the true coverage of those candidates toward zero and shrinks their absolute difference.

So the quantity is not "how wrong is the forecast" but "how wrong is the forecast along the trajectory this forecast induces". For a responsibility magnitude that is the wrong direction: the worst forecast earns the smallest credit exactly because it has already destroyed the opportunity the credit should point at.

## 3. What is and is not established

- The repaired source is correctly wired and leak-free (G1, G2, G4), and on fixed states it is monotone in the forecast error.
- It is not, as measured along the policy's own trajectory, monotone in forecast quality (G3), so it is not yet suitable as the responsibility magnitude without a decision on §4.
- Nothing is established about EU-CRD's effect. G5 (the real-learner wiring gate) never ran.

## 4. Options for a ruling (none taken here; nothing was tuned)

1. Normalise away the compression: divide the mean absolute error by the spread of the *true* coverage over the legal candidates at that state, so a state where nothing is claimable contributes no magnitude either way.
2. Define the signal on a fixed reference trajectory (for example the committed grid of the clean arm, or an empty grid), so the magnitude depends on the forecast alone.
3. Replace the magnitude by the quantity the credit is actually about: the carbon regret at that state of following the forecast's best candidate instead of the truth's best candidate.
4. Keep the signal as is and drop the monotonicity requirement, on the argument that credit assignment only needs a positive signal when the forecast is wrong. This is the weakest option: it accepts that the worst forecasts receive the least forecast credit.

Whichever is chosen must be frozen before it is measured again, and G1, G2, G4 re-run on the new definition; G5 stays queued behind it.

## Addendum A — two corrections to §2 and §3 (2026-09-07, user ruling)

The measurements stand; two statements about them were too strong and are withdrawn.

1. **"The live runs share the same decision states."** What is shared across tiers is the set of decision times and the jobs present at them. The reservation grid differs, so the full scheduling state differs, and a non-monotone comparison across tiers is therefore not evidence that the instrument is wrong. The supported statement is narrower: the old signal is changed by earlier decisions, so it cannot serve on its own as a scale of forecast quality.
2. **"The worst forecast earns the smallest credit."** shrink 0 scores below shrink 0.25 but above shrink 0.75, so the flat forecast is not at the bottom of the ordering. What the inversion shows is the trajectory dependence in point 1, not a reversal of the whole scale.

Ruling: option 3 becomes the responsibility signal, option 2 stays as an auxiliary quality scale, and no training runs until the new definition passes its own gates. The new signal is a **local decision regret** at the current state, not the causal carbon loss of a whole trajectory: opportunity already destroyed by earlier decisions cannot be recovered by a score at the current step. It is frozen in reports/EUCRD_REGRET_SIGNAL_PREREG.md.
