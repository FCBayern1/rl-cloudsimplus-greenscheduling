# Deferrable-Jobs Temporal Lever — Design & Plan

_Status: IN PROGRESS (Phase 1 started 2026-06-18)_

## 0. Why (the core insight)

The 3-arm forecast diagnostic (2026-06-18, 5-DC) proved: **even a PERFECT forecast
(godeye) does not reduce green waste** — waste_ratio pinned at ~0.939 across
none / timecap / godeye. (timecap's marginally lower carbon tracks actor
entropy-drop, NOT forecast quality — godeye, the perfect forecast, is no better
than none.)

**Root cause — a type mismatch between information and action:**
- The forecast carries **temporal** information: *when* green energy arrives.
- Pre-lever, the agent's only action was **spatial**: *which DC* to route to, NOW.
- Green waste = **temporal** misalignment between green supply and load demand
  (green peaks when no load is present). A spatial action cannot fix a temporal
  misalignment. Knowing "DC3 greens at 2pm" is useless if you can't put load at 2pm.

So the forecast was unusable not because it was bad or the actor was dumb, but
because **the action space had no temporal degree of freedom**. The deferrable-jobs
lever adds exactly that missing dimension (move load in time), aligning the
action type with the forecast's information type.

## 1. The lever: a "DEFER" routing option

Extend the per-cloudlet global routing action:
```
MultiDiscrete([num_dc] × batch)  →  MultiDiscrete([num_dc + 1] × batch)
                                                    ↑ value = num_dc means DEFER (hold this step)
```
Action-space cost is marginal: per-cloudlet decision goes N→N+1 options (the
policy is factored / score-based; it just emits one extra logit per cloudlet).

**Mechanics — defer = hold ONE step, re-decide next step (agent controls duration):**
- DEFER → cloudlet enters a hold queue (reuses the existing `requeueCloudletToTail`
  pattern in GlobalBroker).
- Next step the held cloudlet re-enters the routing batch alongside new arrivals;
  agent re-decides route-vs-defer.
- Each task has a `deadline` (max steps it may wait). At deadline, DEFER is masked
  out → forced route. No task is held forever.
- Learned policy target: "no green now + forecast says green soon → defer; green
  arrives (or deadline) → release." This directly attacks waste_ratio (otherwise-
  wasted green now has deferred load to consume it).

## 2. Deadline (two distinct "how long"s)

| | what | who sets |
|---|---|---|
| **deadline (slack cap)** | task property: max steps it may wait | the WORKLOAD. MUST be > forecast horizon (else can't wait for the next green peak). |
| **actual defer duration** | how many steps it was actually held | the AGENT (defers repeatedly until release), capped by deadline. |

## 3. Completion-rate tension → Lagrangian (with a CRITICAL caveat)

If the agent defers everything, carbon drops but completion suffers. This is a
constrained-optimization problem: **minimize carbon s.t. completion ≥ target** —
exactly what the Lagrangian does. The existing two-level SLA cost already has the
machinery (c_step penalizes pending pile-up; c_ep penalizes final shortfall;
λ adapts via dual ascent — see MultiDatacenterSimulationCore ~2028-2036).

**⚠️ CRITICAL: the SLA cost must be made DEADLINE-AWARE.** The current
`pending_ratio` treats all unfinished work the same. A task **deferred within its
deadline** is legitimate; a task **past deadline / backlogged** is a violation. If
the SLA cost penalizes both equally, it FIGHTS the lever — the agent becomes afraid
to defer and the forecast stays unused.
- **Fix:** base the SLA cost on **deadline misses**, not raw pending. Deferral is
  then "free until you blow the deadline" — precisely the incentive that makes the
  forecast valuable.
- Result is a clean story: `minimize carbon s.t. deadline-miss-rate ≤ threshold`,
  with defer = the carbon lever and Lagrangian = the completion guard.
- λ will likely need re-tuning (deferral adds a new carbon↔completion trade-off
  axis); sla_target 0.62 / lambda_max 1.5 from [[project_lagrangian_sla_fix]] are
  starting points, not final.

## 4. Phased plan (each phase independently testable; CLAUDE.md test rule)

**Phase 1 — Java sim (the physical lever; no RL):**
1. `CloudletDescriptor`: add `deferDeadlineSteps` (0 = must route now / not deferrable;
   >0 = may be held up to N steps) + derived `isDeferrable()`.
2. Workload creation: assign deadline/deferrable by config (`deferrable_fraction`,
   `defer_deadline_steps`). Start config-driven; later swap to Alibaba batch/online.
3. `GlobalBroker`: hold queue + per-cloudlet remaining-deadline tracking.
   `getBatchForRouting` returns [non-expired deferred + new arrivals], flags at-deadline.
   New `deferCloudlet()` (requeue + decrement; refuse at deadline).
4. `executeGlobalRouting`: action==num_dc → deferCloudlet instead of route.

**Phase 1.5 — DE-RISK with a scripted oracle (cheap, before any RL):**
- Hand-coded forecast-greedy deferral policy (no green now + forecast green within
  N → defer; green → release). Measure waste_ratio / carbon.
- **Decision gate:** even a PERFECT forecast-using scripted policy can't move
  waste_ratio → the lever mechanism doesn't help → STOP (don't build the RL stack).
  It moves waste_ratio → mechanism works, it's "just" an RL learning problem → proceed.
- (Same philosophy as the godeye diagnostic: test the oracle upper bound before
  investing in learning.)

**Phase 2-4 — RL stack (ONLY if Phase 1.5 passes):**
- P2 env: action space num_dc→num_dc+1; obs adds per-cloudlet deadline/deferrable;
  action masking (defer disabled for non-deferrable / at-deadline).
- P3 RLModule: score-based global module emits one extra defer logit per cloudlet,
  honoring the mask. Unit-test shape + gradient + mask.
- P4 reward + validate: deadline-aware SLA cost; retrain; **re-run godeye 3-arm
  diagnostic** — success = godeye > none on waste_ratio (forecast value real) →
  EU-CRD collapse finally becomes viable.

## 5. Cost / risk
- Multi-day; phased + tested so it's incremental.
- Action-space change invalidates ALL baselines/checkpoints/Pareto — full re-train/
  re-baseline once. Necessary: this is the only path to make forecast + EU-CRD
  meaningful.
- Headroom is real: waste_ratio 0.94 = huge wasted green = lots of temporal
  misalignment for the lever to fix.
- Residual risk: if the workload has no exploitable temporal structure, even the
  lever won't help — Phase 1.5 oracle settles this cheaply before the RL spend.

## 6. PIVOT (2026-06-18): the local NoAssign hold already exists — try Option B first

User observation: the LOCAL action `Discrete(max_vms+1)` already has a **NoAssign**
option (action 0 → targetVmId -1) that HOLDS a cloudlet in the DC's local queue
(verified: task not lost, re-assignable next step — MultiDatacenterSimulationCore
~720). So a temporal hold lever ALREADY EXISTS — but it's crippled 3 ways:
1. **forecast-blind**: `_convert_local_observation` has NO future-green features
   (forecast lives only in the GLOBAL obs) → can't anticipate green.
2. **penalized**: "NoAssign while cloudlets waiting → Invalid action" (SimCore ~726)
   → reward discourages holding.
3. **same-DC only**: a task at DC1 can only wait for DC1 to green, can't redirect to
   DC3's green peak.

### Two options:
- **A (global defer, original plan)**: new route-level DEFER; hold + pick greenest
  DC. Flexible, bigger build (action space +1, RLModule). Fixes all 3 limits.
- **B (revive local hold)**: feed per-DC forecast into local obs + de-penalize the
  NoAssign-hold + add a hold deadline. NO action-space / RLModule change (NoAssign
  exists). Cheaper. Limit: same-DC only (depends on global router pre-positioning
  tasks at soon-green DCs; likely captures imminent green peaks, not far/other-DC ones).

### Plan: B first (cheap), A as fallback.
- **B-oracle (do FIRST, cheapest, decisive — needs NO obs/reward/RL change):** a
  scripted policy that reads the forecast DIRECTLY and uses the EXISTING NoAssign to
  "hold a DC's waiting cloudlets while DC not green + forecast says green soon;
  assign to VM when green." Run an episode, measure waste_ratio/carbon vs a no-hold
  run. waste drops → B mechanism works → build B-RL. waste flat → same-DC limit is
  fatal → go to A.
- **B-RL (only if oracle passes):** (B1) per-DC forecast → local obs; (B2) de-penalize
  NoAssign-hold; (B3) hold deadline + deadline-aware SLA (CloudletDescriptor deadline
  field reused); retrain; re-run godeye 3-arm diagnostic (success = godeye > none).
- **A (global defer):** only if B's same-DC limit proves fatal. Phases 1(#7-9)/2-4.

CloudletDescriptor deadline field (done, #6) is used by BOTH B and A — not wasted.

## 7. STOP — B-oracle finding (2026-06-18): waste is a calibration artifact, not a lever problem

B-oracle (oracle_hold_until_green.py, 5-DC godeye): 7165 hold decisions →
green_used/green_waste EXACTLY unchanged (982/17941 Wh), completion 0.18→0.16.
Total green ~18923 Wh, only 5.2% used. **Green supply ≫ demand → no temporal lever
(B or A) can move waste; time-shifting tiny demand can't absorb a ~20× green surplus.**
The carbon drop (0.436→0.410) is a DID-LESS-WORK artifact (lower completion → less
brown), NOT green capture (green_used identical); carbon/completion is actually worse.

This re-interprets the whole project: waste_ratio pinned ~0.94 across ALL runs is
STRUCTURAL (green over-supply), not a policy/forecast failure. godeye≈none meant
"no policy can move waste under this calibration," not just "lever missing."

**NEW critical-path step (lever plan PAUSED): re-calibrate green/load balance so
load ≈ green supply.** Levers: compressed_power_divisor (now 150; larger = less
wind), more/heavier cloudlets (raise load), or lower host idle-power fraction
(make power load-sensitive). Use the oracle as the fast (~½ day) test loop:
recalibrate → rerun oracle → does hold-until-green move waste? Only then resume B/A.
CloudletDescriptor deadline field (#6) and the oracle stay valid.

## 8. RESOLUTION (2026-06-18): new dispatch-rate local agent (unifies throughput fix + temporal lever)

Root cause found (user insight): local dispatches only 1 cloudlet/DC/step
(assignCloudletToVm = queue.poll() once per DC) → throughput-starved → DCs
under-loaded (13% util) → green wasted, completion ~0.18. A temporal lever is
MEANINGLESS while throttled: the lever's value is "burst-process during green",
but a flat 1/DC/step cap removes the ability to burst.

**Key realization: "how many to dispatch this step" IS the temporal lever.**
- dispatch low = hold/defer (brown periods)
- dispatch high (burst) = release (green periods)
So fixing throughput and adding the lever are the SAME change.

**Also: per-VM placement is green-IRRELEVANT** (all VMs in a DC share the DC's
green; with always-on hosts, total DC power is invariant to which VM runs a
cloudlet). So placement → heuristic (best-fit), RL → green-relevant decisions
only (DC routing + dispatch timing). Cleaner architecture; reuses BestFit.

### New local agent (incremental, OLD path untouched, config-gated)
`local_dispatch_mode: "vm_placement"(old default) | "dispatch_rate"(new)`
- **action**: Discrete(N+1) = how many cloudlets to release this step (0=hold all,
  N=burst-drain). N configurable.
- **placement**: sim-internal best-fit over the N dispatched cloudlets (not RL).
- **obs**: + this DC's green_ratio + forecast (dc_future_*) + queue len + VM
  aggregate free capacity → the lever can learn "hold when no green + green
  coming; burst when green".

### Incremental tasks (each testable; old path bit-identical when flag off)
- NL-1 (#13) Java: gated dispatch-rate path (release N via best-fit). Java test.
- NL-2 (#14) env: Discrete(N+1) action + green/forecast in local obs (gated).
- NL-3 (#15) oracle: scripted hold/burst-by-green → does completion/util jump?
  + green balance → does waste finally move? (gate before RL)
- NL-4 (#16) RL: local module handles Discrete(N+1); forecast→obs; deadline-aware
  SLA; retrain; rerun godeye 3-arm (success = godeye > none).

⚠️ Honest: this is a sim-foundation rebuild. Invalidates all baselines/Pareto.
And earlier carbon wins (−7%, RL +28%) were under the throttled+over-supplied
dynamics → likely static-tilt artifacts → must be re-validated after the rebuild.
