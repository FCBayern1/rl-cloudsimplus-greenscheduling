# Development smoke A: verdict and the ruling needed (2026-09-05 13:30)

Context: STAGE_D_PRIME_DESIGN.md §16 (your ruling), §26–§27 (readings). Everything here is development-phase (sister-turbine `calibrated_shrink_v1`, old windows, seed 20260903, 56k steps per line). Artefacts: `reports/manifests/stage_d/dprime/smoke_a/` (252 rows, health verdict, probes, E-last audit + per-transition file, cross statistics, selectivity, both verdict files).

## 1. Six-criteria verdict: STOP_DPRIME_SMOKE

| gate | reading | pass |
|---|---|---|
| contract clean, all lines | 72/72 clean rows on-time 1.000, completion 1.000 | yes |
| forced = 0 | 0 on every row; env-side re-route never fired on an RL row | yes |
| defer not collapsed | line defer rates 0.42–0.58 (V 0.504) | yes |
| guard wiring sentinel (applied weight) | P(w′ < 0.2) = 0.000, min w′ 0.528 = 1 + 0.5 (0.056 − 1) | yes |
| guard, no mass erasure (E last, DEFER) | n(A<0) 585; R_raw 0.994, R_guarded 0.997; bitwise 6e−8 | yes |
| reward–carbon co-direction | V −146.5 → −111.0 / 0.00909 → 0.00731; E −140.0 → −100.5 / 0.00876 → 0.00679 | yes |
| timing selectivity (V last; raw, recurrent, job-paired) | lift −0.022, balanced AUC 0.308 | **no** |

Selectivity detail: 210 jobs, 129 paired (44 excluded as mask-forced routes, 37 never deferred by ST). Per window AUC 0.14–0.46, lift −0.043 to −0.005: uniform failure. Deployed = raw on the pairs (both members DEFER-legal by construction, mask never binds; sanity check passed). Appendix all-sightings AUC 0.641 (8693 : 210) is queue-time exposure, not the gate. V's DEFER preference is 0.60 at a job's first legal wait moment and 0.62 at the moment ST starts it: it waits about 60 % of the time regardless of state, slightly more the longer a job has been present. The policy has not represented the timing that E1 showed carries the value.

Guard: E never drifted on this smoke (defer 0.45, stable), so E[w | DEFER, A<0] 1.045 > E[w | DEFER, A≥0] 0.971, the opposite sign from the Stage D drift checkpoints. The guard is wired and harmless here; the smoke does not show it is needed.

## 2. One instrument disclosure

The first judge run also failed the wiring sentinel because the judge read the audit's raw-weight lower tail (DEFER 0.066 > 0.05) while §16 item 1 defines the sentinel on the applied weight w′. The audit summary now also emits the guarded tail (0.000) and its minimum; the judge reads that field and fails when it is absent. The E audit JSON was re-summarised from the saved per-transition file (same n and mean w; originals kept as `*.prefix.json`). Verdict unchanged; 11 judge/audit tests pass. Proposed wording: development-phase instrument repair, same class as P0′ runs 2–3.

## 3. Caveat, recorded only

56k steps per line is a development budget. Whether selectivity would emerge at 400k steps was not tested. My position: testing it now would be tuning on a result, and §16 item 5 applies as written. Please confirm or overrule.

## 4. Rulings requested

1. Confirm STOP D′ under §16 item 5 and that no longer smoke is run.
2. Scope of the action design to write next. Two candidates: (a) (DC, start-offset) with K discrete offsets per job; (b) an option "route to DC d now" vs "hold until green ≥ demand at d, or until the deadline margin". I lean (b): the zero-training planner ST is exactly the run-when-green rule, so one decision expresses it, the deadline mask carries over unchanged, and the small-sample learnability probe has a two-way target. (a) needs a horizon and forecast-window alignment first. Nothing written yet beyond this note.
3. The four probes (zero-training expressibility, action–execution closure, contract safety, small-sample learnability): are the pass criteria to be preregistered before the design is implemented, or fixed in the design document with the design?
4. Whether the 2020 / never-used-turbine continuation (§24–§25) is parked until the action design passes its probes. I assume yes; nothing has been selected.
