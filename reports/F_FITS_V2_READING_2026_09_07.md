# F_FITS_V2: reading (2026-09-07, 02:15)

Preregistration: reports/F_FITS_V2_PREREG.md, frozen at 80b693fb; Addendum A (footprint 1200, before the draw), Addendum B (the sentinel fired; two one-row defects fixed append-only; rerun), Addendum C (model freeze 4a552fb4 before the test windows). Run `stage_a_out/f_v2`, archive `reports/manifests/f_v2/{freeze,run1}`. Verdict: **STOP_F2_LEARNER**.

## 1. What ran

- Sentinel: SENTINEL_PASS on all 36 (twin, window) pairs after the fixes of Addendum B: forecast series index h equals the simulator's row t + h to 3e-5 W, the candidate key recomputed from the dumped series and committed grid equals the published key exactly, every observation row equals the wind-file curve.
- Labels: 18 windows × 35 = 630 decisions of the causal expert with the true cost of every legal candidate (fix-and-resolve on joint states); 0 states excluded, the expert's action in the minimum-cost set 630/630.
- Corpora: the expert's schedule replayed on the truth-key twin (F2) and the TimeCAP-key twin (F3) reproduces its carbon bit for bit on all 18 windows.
- Fits: candidate-shared residual on 420 training decisions, selected on the 70 validation decisions (F2: 200 epochs, weight decay 1e-4, validation set loss 1.151; F3: same grid point, 3.270). Frozen before the four test windows were read.

## 2. Test windows (read once; carbon in kg, simulator-settled; validity by the ladder's L1 rule)

| window | offline truth | offline flat | causal expert | headroom rel | valid |
|---|---|---|---|---|---|
| 24859 | 0.004336 | 0.007730 | 0.004926 | 0.44 | yes |
| 28745 | 0.000368 | 0.003617 | 0.000401 | 0.90 | yes |
| 41897 | 0.000306 | 0.001335 | 0.000306 | 0.77 | yes |
| 7934 | 0.001426 | 0.003766 | 0.001716 | 0.62 | yes |

Offline truth and flat closed on every window (carbon relative error ≤ 2e-7); every contract green on every run (expert, cover_argmax, fitted arms, both twins).

Capture of the causal expert's headroom, (C_flat − C) / (C_flat − C_causal):

| window | F2 cover_argmax (truth key) | F2 fitted residual | F3 cover_argmax (TimeCAP key) | F3 fitted residual |
|---|---|---|---|---|
| 24859 | 1.005 | 0.962 | −0.087 | −0.303 |
| 28745 | 1.007 | 0.980 | 0.635 | −0.548 |
| 41897 | 1.000 | 0.910 | 0.246 | 0.266 |
| 7934 | 1.016 | 0.988 | 0.034 | −0.826 |
| pooled | 1.006 | 0.968 | 0.20 | −0.44 |

Gate F2: pooled capture 0.968 ≥ 0.50 passes; the per-window condition (≥ cover_argmax − 0.02) fails on all four windows (−0.043, −0.027, −0.090, −0.028). Verdict STOP_F2_LEARNER as preregistered.

## 3. Reading

- With the aligned candidate key, the zero-parameter rule `cover_argmax` reproduces the causal expert on four sealed windows it never saw (capture 1.000–1.016): on the truth key the interface plus the exact greedy rule IS the online expert. There is nothing for a learned residual to add on the truth key; the residual trained on exact costs and selected on validation loss loses 1 to 9 points of capture at execution, because its own trajectories leave the expert's states (the labels were produced on the expert's committed grid) and the cost ties it reorders are only ties in the expert's states.
- The learner is therefore not "the failure" in the sense of version 1 any more. What can be said: the version-1 evidence is void because of the misalignment; after the alignment a zero-parameter rule reaches the expert on the truth key, so no learner is needed there. What cannot be said: that misalignment was the only cause of the version-1 failure, because version 2 also changed the labels, the loss, the model and the amount of data.
- F3 (diagnostic, TimeCAP key): cover_argmax keeps 0.20 pooled on these four windows (window by window −0.09, 0.64, 0.25, 0.03), against 0.82 on the six development windows; the fitted residual is worse (−0.44). The deployed forecast's error is severe on these windows; this one small residual model trained on 420 decisions did not correct it. Nothing more is claimed about learnability. Consistent with the A2 STOP; does not override it.

## 4. Consequences (decisions for the user)

- The RL question no longer needs a behaviour-cloned policy: the interface with the cover rule is the expert. The four-line RL design (vanilla / EU-CRD × with / without forecast) should start every line from the cover rule (a fixed or initialised cover term) and ask only whether training preserves it on the clean channel and how it degrades along the shrink ladder; cover_argmax is the mandatory reference on every window and tier.
- A further learner on the truth key is not warranted; on the TimeCAP key this residual with 420 decisions did not correct the error, which is all that is established.
- The 2020 windows stay sealed; the GPU stays parked until an RL preregistration is frozen.
