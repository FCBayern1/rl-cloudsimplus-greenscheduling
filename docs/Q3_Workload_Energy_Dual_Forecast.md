# Q3 — Dual-Side Forecasting: Workload + Energy Bilateral Alignment

_Status: **FUTURE WORK / PROPOSAL** (not yet implemented or validated)_
_Created: 2026-05-31_

## 0. One-line pitch

The current system forecasts only the **supply side** (renewable energy, via
TimeCAP) and feeds it to the **policy**. We propose a symmetric **demand-side**
forecaster (workload arrivals) feeding the **value function (critic)**, turning
the architecture into a **bilateral forecasting framework**:

```
  Supply forecast (renewable, TimeCAP)  ─────►  Actor  (routing decision)
  Demand forecast (workload, proposed)  ─────►  Critic (value estimation)
```

Each forecast resolves a *different* identifiability gap on a *different*
network head. This is the conceptual core of the proposed Q3 contribution.

---

## ⚠️ 1. Honest premise check (read before building)

This proposal rests on **one empirical assumption that is NOT yet verified**:

> *that the global critic's value function is unlearnable because the dominant
> return variance comes from information absent in the observation, and that a
> workload forecast supplies that information.*

The diagnostic history is nuanced and **partially contradictory**:

1. **Observed (robust):** `global_vf_explained_var ≈ 0` across all runs, while
   `local_vf_explained_var` reaches 0.5–0.98 on the same trace (5-DC config C,
   run 20260523_175433). The two critics differ structurally.
2. **First explanation (RETRACTED):** "return dominated by exogenous random
   arrivals." **Wrong** — the workload trace is *deterministic* (fixed CSV,
   identical every episode). Future arrivals are a deterministic function of
   sim-time, not a random variable. A reviewer will catch this immediately.
3. **Second explanation (current best hypothesis, UNVERIFIED):** the global
   critic's failure is driven by **horizon length**, not missing demand info.
   - `local critic` predicts short-horizon, *local* returns → strongly
     determined by local obs → learnable (explained_var up to 0.98).
   - `global critic` predicts long-horizon returns (γ=0.995 → effective horizon
     ~200 steps) of the whole system → most of that future is not in the obs →
     unlearnable.
   - Direct corollary: lowering γ (shorter horizon) may revive the global
     critic *without any forecast at all*. This is a cheaper, competing fix
     that must be ruled out first.
4. **Critical caveat for THIS proposal:** under a *deterministic* trace, a
   workload forecast is informationally **equivalent to a timestep feature**
   (future load = f(t)). So if the critic problem is "missing t", adding a
   plain timestep would fix it and the forecasting framing collapses.

**Gate experiments that MUST pass before committing to this work** (§6).
Until they pass, treat the whole proposal as conditional.

---

## 2. Background: the supply-side forecaster (already built)

The renewable-aware path is implemented and validated:

- **Module:** TimeCAP green-energy predictor (23.8M params), enabled by
  `green_oracle_mode: timecap`.
- **Output → obs:** four per-DC scalar priors injected into the *global
  observation*:
  - `dc_future_short_mean` — short-horizon mean available green power
  - `dc_future_short_trend` — short-horizon trend
  - `dc_future_long_mean` — long-horizon mean
  - `dc_future_long_peak_timing` — when the next renewable peak arrives
- **Consumer:** both heads see it today; conceptually it serves the **actor**
  (it answers "which DC will be green, route there").

This pipeline (semantic-compression forecasting → low-dim structural priors →
obs augmentation) is the template the demand side mirrors.

---

## 3. Proposed demand-side forecaster (the new work)

A symmetric **workload forecaster** trained on global queue history +
per-datacentre arrival traces, emitting four scalar **demand priors** that
mirror the renewable specification:

| Renewable prior (existing) | Workload prior (proposed) |
| --- | --- |
| `dc_future_short_mean` (green power) | `wl_future_short_mean` (arrival rate) |
| `dc_future_short_trend` | `wl_future_short_trend` |
| `dc_future_long_mean` | `wl_future_long_mean` |
| `dc_future_long_peak_timing` | `wl_future_peak_timing` (next demand surge) |

Justification for a *low-dimensional* prior: the v2 trace has clear **diurnal
structure** (arrival rate cycles, mean ~6/step on 5-DC, ~8/step on 10-DC), so a
4-scalar summary should capture most of the predictable component. (If arrivals
turn out to be bursty/Poisson-superimposed, 4 scalars may be insufficient —
this is a risk to test, §6.)

### Gating: demand priors go to the CRITIC only

The key design choice. The workload priors are **gated to the value head**,
NOT the policy head:

```
  global obs ─┬─ (renewable priors + state) ─► actor  ─► routing logits
              └─ (renewable priors + state
                  + workload priors) ─────────► critic ─► V(s)
```

Rationale:
- The **routing decision** should stay a function of the renewable-aware state
  alone — at inference time the policy does not depend on a demand forecaster,
  keeping deployment simple and the action distribution unchanged.
- The **value estimate** is where the demand information resolves the
  identifiability gap: with future-load priors, `E[G_t | s̃]` can separate from
  `E[G_t]`, giving the TD target discriminative signal.

This actor/critic asymmetry (different conditioning sets per head) is naturally
supported by the existing **dual-trunk critic** (Route 2.5 Phase 2): the actor
and critic already have independent encoders + GTrXL trunks with isolated
gradients, so feeding extra features to only the critic trunk is a localized
change.

---

## 4. Formalization (for the paper)

Let `s^g_t` be the renewable-aware global state and `w_t` the workload prior.
The augmented critic state is `s̃^g_t = (s^g_t, w_t)`.

The return decomposes as
```
G_t = (component shaped by the current routing decision)
    + (component shaped by load over the next ~H steps)        ← dominant
```
Under the current obs, the second component is (near-)independent of `s^g_t`,
so the Bayes-optimal value under MSE collapses:
```
V*(s^g_t) = E[G_t | s^g_t] ≈ E[G_t]      (constant — explained_var ≈ 0)
```
With demand priors,
```
V*(s̃^g_t) = E[G_t | s^g_t, w_t]          (separates from E[G_t], to the extent
                                          the forecast is accurate)
```
Expected magnitude is **conditional on forecast quality**: a perfect forecast
collapses `Var(G_t | s̃)` maximally; a noisy forecast helps proportionally; a
random forecast does nothing. (State this as a proposition, not a claim.)

---

## 5. Relation to EU-CRD (distinct, complementary)

This is conceptually separate from EU-CRD:
- **EU-CRD** reweights the *policy gradient* so the actor is not penalized for
  forecast noise it cannot control (a credit-assignment fix on the actor side).
- **Workload-aware critic** restores *value identifiability* so exogenous
  demand variance is absorbed by the conditioning set rather than appearing as
  irreducible target noise (an estimation fix on the critic side).

Expected to interact constructively: the demand prior gives a sharper baseline
against which EU-CRD can isolate the local-scheduling share of tail risk.

---

## 6. Gate experiments (DO THESE FIRST — they decide if the work is viable)

Ordered; each is ~1 day. Stop early if a gate fails.

### Gate A — Is the critic problem "missing timestep"? (cheapest)
Add a single `episode_progress = t / max_steps` feature to the **critic obs**,
train 25 iters.
- `vf_explained_var` jumps to 0.3+ → the cause is a trivial timestep; **the
  forecasting framing collapses** (under deterministic load, forecast ≡ t).
  → Pivot: either make workload stochastic (Gate D) or drop this proposal.
- still ≈ 0 → timestep is not enough; continue.

### Gate B — Is it "horizon too long"? (competing cheap fix)
Lower **global γ** 0.995 → 0.97 (critic-relevant; keep actor reasoning long if
possible), train 25 iters.
- `vf_explained_var` rises → the cause is horizon, fixable by γ alone; the
  forecast is then a *secondary* improvement, not the headline.
- still ≈ 0 → horizon isn't the (sole) issue; the missing-information
  hypothesis survives → proceed to Gate C.

### Gate C — Oracle-forecast feasibility (decides the whole proposal)
Skip building an ML forecaster. Compute the **true** future-K-step arrival
statistics directly from the trace (a perfect oracle forecast), inject the 4
priors into the **critic obs only**, train 25 iters.
- `vf_explained_var` rises to 0.3+ → demand info genuinely resolves the gap →
  **the proposal is viable**, proceed to build a real forecaster.
- still ≈ 0 → even perfect demand info doesn't help → the return is dominated
  by policy stochasticity or something else; **abandon the forecasting story**,
  keep "critic ceiling" as a negative-but-deep finding.

### Gate D — (Only if A shows timestep suffices) Make workload stochastic
To make the demand-forecast story *non-trivial*, randomize arrivals across
episodes (sample arrival times/sizes per episode). Then future load is genuinely
exogenous, timestep no longer determines it, and a forecast adds real
information. This also matches real datacenters (load is stochastic), making the
premise honest rather than fragile.

---

## 7. Implementation sketch (only after gates pass)

1. **Obs space:** add 4 `wl_future_*` keys to the global obs (Box, per-DC or
   global scalar). Gate them to the critic trunk in
   `GTrXLScoreBasedGlobalRLModule._forward_pass` (the dual-trunk critic branch),
   excluded from the actor's score function.
2. **Forecaster:** reuse the TimeCAP semantic-compression pipeline on the
   arrival series; or a lightweight diurnal-mean + trend predictor first (cheap
   baseline to compare ML forecaster against).
3. **Plumbing:** mirror the `green_oracle_mode` switch with a
   `workload_oracle_mode: {off, oracle, learned}` so Gate C (oracle) and the
   final (learned) version share a code path.
4. **Tests:** (a) workload priors reach the critic but NOT the actor logits
   (gradient-isolation unit test, analogous to the existing dual-trunk tests);
   (b) action distribution unchanged when only workload priors change.

---

## 8. Metrics to report (NOT just c/c)

A revived critic likely won't lower c/c (it's near the workload's carbon floor).
The payoff is elsewhere — report:
- `vf_explained_var` (the direct target; with vs without demand prior)
- advantage variance (lower = better baseline)
- 10-DC best→final **drift** (the dead critic's main symptom; expect reduction)
- convergence speed (iters to reach a c/c threshold)
- forecast-quality → explained-var curve (core figure: identifiability
  recovers as forecast accuracy increases)

---

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
| --- | --- | --- | --- |
| Critic problem is just "missing timestep" | Medium | Collapses framing | Gate A first |
| Horizon (γ) is the real cause | Medium | Demotes forecast to secondary | Gate B |
| Oracle forecast doesn't help (policy noise dominates) | Medium | Kills proposal | Gate C |
| Deterministic trace makes forecast ≡ timestep | High (current setup) | Trivializes story | Gate D: stochastic workload |
| 4 scalars insufficient for bursty arrivals | Low–Med | Weak forecast | richer prior / full TimeCAP |
| Reviving critic doesn't lower c/c | High | "so what" | report variance/drift/convergence, not c/c |

---

## 10. Bottom line

The **bilateral forecasting** framing (supply→actor, demand→critic) is elegant
and, if the gates pass, gives a clean *problem → diagnosis → principled fix →
validation* story that elevates the work above "yet another RL scheduler." But
its premise is currently **unverified and partly contradicted** by the
deterministic-trace observation. **Run Gates A–C (≤3 days total) before
investing in a real forecaster.** If they pass, this is a strong standalone
contribution; if not, fall back to the (still publishable) "critic
identifiability ceiling" finding and the dual-scale robust-dominance result.
