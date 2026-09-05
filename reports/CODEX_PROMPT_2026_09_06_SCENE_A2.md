# Scene v1/v2: sixth window found, A2 error gate STOP_ERROR_NOT_LOAD_BEARING; ruling needed (2026-09-06 00:40)

Context: SCENE_INTERFACE_DESIGN.md (v1 + Addenda A–C, frozen); STAGE_D_PRIME_DESIGN.md §37–§39. Artefacts: `reports/manifests/scene_v1/{cert_pool12, v2_search_invalid_run1, v2_search_run2}`. Zero RL throughout; the 2020 confirmation windows were never read.

## 1. Two disclosures

- The 2021 file holds at most seventeen disjoint footprints; after the pool of twelve only three candidates exist (49625, 36713, 30299), not twelve. The search covered those.
- Search run 1 was invalid (my harness): the simulator selects its window from the block's allowlist by reset index modulo its length, while the planner was given the candidate offset, so the truth-curve planner planned on a window the simulator was not running and "lost" 13–87 % to the blind. Caught from the pattern, archived as invalid, fixed by giving every non-pool window list its own allowlist config; the pool-12 certification was aligned and stands.

## 2. Results (aligned)

- Candidate 13 (49625) passes the unchanged headroom gates at the first try (gap 22.5 %, 0.00115 kg). Development set: 16477, 4240, 9154, 33225, 13223, 49625.
- A2 error gate on the six development windows (calibrated shrink v2 = the deployed TimeCAP's real error on these turbines, λ ≈ 0.88): pooled shrink/ST = 1.072 (≥ 1.05 ✓), windows with shrink above ST = 2 of 6 (need 4 ✗) → **STOP_ERROR_NOT_LOAD_BEARING**.

| dev k | offset | C_ST | C_shrink | ratio |
|---|---|---|---|---|
| 0 | 16477 | 0.001406 | 0.001327 | 0.943 |
| 1 | 4240 | 0.002391 | 0.003336 | 1.395 |
| 2 | 9154 | 0.001899 | 0.002478 | 1.305 |
| 3 | 33225 | 0.001879 | 0.001458 | 0.776 |
| 4 | 13223 | 0.000942 | 0.000887 | 0.942 |
| 5 | 49625 | 0.003961 | 0.003896 | 0.983 |

On four of six windows the mean-pulled forecast is as good as or better than the truth for the reserving planner (it is not the optimum on them; §35's dyadic-grid wins on k0/k1 said the same). Margin probe, P0′ and gates 4.1–4.5 were not run.

## 3. Rulings requested

1. Confirm STOP_ERROR_NOT_LOAD_BEARING as the frozen outcome of this scene, with the reading "the deployed forecast's real error is too mild, on this scene, to be load-bearing for the analytic scheduler", not "forecast error never matters".
2. What the thesis line does next. As I see it there are three honest routes, none of which I start without a ruling:
   (a) another never-used turbine set under the same frozen rules (third hash draw), accepting that the deployed checkpoint's error may again be mild;
   (b) a stronger error model that is still real (a weaker checkpoint, a longer lead, or the checkpoint evaluated where its audit shows larger λ deficits), preregistered as the error arm before any scene is read;
   (c) reframing the thesis around an error regime that is load-bearing by construction (shuffle / anti are, by 50–80 % on this scene) while stating plainly that the deployed checkpoint's own error is not.
3. Whether the A2 window criterion (≥ 4 of 6) should, for any future scene, be replaced by a headroom-weighted rule like the one archived for gate 1 (§35), decided before that scene's rows exist.
