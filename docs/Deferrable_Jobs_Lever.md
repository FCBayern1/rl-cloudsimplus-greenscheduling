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
