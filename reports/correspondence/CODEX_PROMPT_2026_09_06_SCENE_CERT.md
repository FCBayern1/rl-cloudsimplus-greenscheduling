# Scene v1 step 2a: mechanism control PASS, headroom gate 5/12 → STOP_WINDOW_SPLIT; ruling needed (2026-09-06 00:15)

Context: SCENE_INTERFACE_DESIGN.md v1 + Addenda A, B (frozen 0bbd6f7a); STAGE_D_PRIME_DESIGN.md §36 (step 1) and §37 (this result). Artefacts: `reports/manifests/scene_v1/cert_pool12/`. Nothing beyond step 2a was run; the 2020 confirmation windows were not touched.

## Step 1 (done, disclosed)
The never-used inventory had missed the legacy singular `turbine_id:` key (the gateway still honours it); a literal scan of every tracked config/report/script outside the wind dataset excluded six more ids (57, 124 legacy defaults; 2, 5, 9, 46 in tests/scripts/an old README). Turbines by the frozen hash rule: DC0 133, 78; DC1 22, 81; DC2 94. Six 2020 confirmation windows drawn and frozen (24398, 10829, 7479, 20843, 523, 14997), unread.

## Step 2a results (2021, twelve hash-ordered pool windows, 48 rows, contract clean)
- Mechanism control PASS: pooled B 0.039255, ST 0.027651 (−29.6 %), shuffle 0.059917, anti 0.070697.
- TimeCAP audit v2 on these turbines (2021): λ ≈ 0.88 on every DC (regression to the mean), no false-peak or ranking pathology.
- Headroom gate: C_brown_ref = 37.94 Wh × 0.5 kg/kWh = 0.01897 kg → absolute gate 9.49e−4 kg. Five of twelve windows pass (k3, k5, k6, k8, k9; relative gaps 34–69 %). Rejected: k1, k4, k11 (no headroom, 0.5 / −1.6 / 4.9 %), and k0, k2, k7, k10 with relative gaps 17.5 / 41.8 / 67.9 / 17.3 % that fail only the absolute gate (0.0003–0.0008 kg). Verdict by the frozen rule: STOP_WINDOW_SPLIT.

## What the numbers say about the rule (recorded, not acted on)
The absolute gate is 5 % of the all-brown bound of the whole trace, but a window's blind carbon is only 5–20 % of that bound on this scene, so the absolute gate amounts to 25–90 % of the blind's carbon per window. It rejects windows whose relative headroom is two to four times the relative threshold. The relative gate alone would pass eight of twelve.

## Rulings requested
1. Confirm STOP_WINDOW_SPLIT for the pool of twelve as the frozen outcome.
2. Continuation. Two candidates, both post hoc with respect to the pool reading and therefore yours to allow or refuse:
   (a) extend the pool in the same hash order (windows 13–24 of the 2021 draw) under the unchanged gate until six pass or the pool is exhausted; thresholds untouched, acceptance still reads only B and ST;
   (b) re-express the absolute gate relative to the window's own blind carbon (e.g. gap ≥ 0.15 · C_B, the relative gate) with a minimum absolute headroom stated in kg from the scene's per-window scale, applied to a fresh pool, not to these twelve.
   My reading: (a) changes no threshold and keeps the acceptance blind to any policy; (b) is the better-calibrated rule but is being written after seeing why the current one fails.
3. Whether the five passing windows may be kept as part of the development set under (a), or the whole set must be re-drawn.
4. Whether the mechanism-control PASS and the v2 audit stand regardless of 1–3 (they read the same twelve windows; the audit reads no carbon).
