# Post-verdict mechanism diagnostic: spatial–temporal decomposition of the HZ analytic lever — definition FROZEN 2026-09-05 before any S row exists

Approved by Codex on 2026-09-05 (§4 of the STOP ruling). This is a **diagnostic, not a continuation**: it does not alter the Stage D verdict (STOP_STAGE_D_CONTRACT, Addendum G) and produces no pass/fail of its own. Its purpose is to decide, with evidence, whether the next design turn changes the action parameterisation or the credit assignment.

## 1. Question

Of the analytic lever on the HZ ×2 scene (truth-informed planner against the frozen strongest blind), how much is captured by choosing the **site** with true future knowledge while starting every job at its earliest feasible time, and how much needs choosing the **start time** as well?

## 2. Arms (all zero-training, all on the identical scene)

| arm | scheduler | information | deferral | how invoked |
|---|---|---|---|---|
| B | `reactive_wait_planner` | current green only (persistence), no reservations | waits reactively for green | the HZ confirmation's frozen strongest blind; **rows already exist**, reused unchanged |
| S | `perturbed_oracle_planner`, tier `godeye` | true future curve, capacity-aware reservations | **forbidden**: `PLANNER_ALLOW_DEFER=0`, so `latest = t` and the planner optimises the site only, starting now | new rows |
| ST | `perturbed_oracle_planner`, tier `godeye` | true future curve, capacity-aware reservations | allowed up to the latest feasible start | the HZ confirmation's clean arm; **rows already exist**, reused unchanged |

S differs from ST by exactly one environment variable, so the site-selection model, capacity truth, deadline handling, power ledger, static-floor override (`PLANNER_STATIC_TOTAL_W=0`), expected-capacity vector and the fallback path (`_fallback_now`, shared with every arm) are identical. Because S keeps the reservation machinery and the same fallback, any feasible arrangement the blind arm reaches is reachable by S; S cannot be worse than B for lack of information alone.

## 3. Scene, cells, windows

Exactly the HZ confirmation set: config `config_s2hz_m2.yml`, the six pilot cells, the three sealed-then-read confirmation windows k = 26, 34, 42 (`e_data_split.json["confirmation"]`), `HZ_ENV` unchanged, one episode per grid, the same evaluate.py path. 18 new runs (S only).

## 4. Quantities

Pooled carbon intensity C_X = Σ carbon / Σ completed MI over the 18 grids, computed by the same pooling as the HZ verdict, for X ∈ {B, S, ST}. Reported, with per-cell and per-window breakdowns:

- spatial capturable: C_B − C_S
- temporal increment: C_S − C_ST
- total lever: C_B − C_ST
- spatial share: (C_B − C_S) / (C_B − C_ST)

Contract fields (completion, on-time, forced starts) are reported for S alongside carbon; a contract failure on S is reported, not voided, and the shares are then computed both with and without those grids.

## 5. Interpretation, fixed in advance

- **Spatial share ≥ 0.6**: the lever is mostly site choice. The trained V line (which uses the forecast and captured 6.4% against N_V on seed 1) is then near the ceiling of what forecast-driven site choice can buy, and gate 2 was unreachable by construction. Next turn: do not change the action space; find why RL does not capture the rest of the spatial value and whether `calibrated_shrink_v1` preserves site ordering.
- **Spatial share ≤ 0.4**: the lever is mostly start-time choice, which the per-step DEFER action cannot express with tractable credit. Next turn: a one-shot (site, start-offset) or macro-action parameterisation.
- **In between**: both, and the per-action-type credit audit (§6) decides the order.

No threshold here is a gate; they are the pre-committed reading rules for a diagnostic.

## 6. Companion diagnostic, defined now, run second: per-action-type credit audit on E and N_E

Offline, from the archived seed-20260904 checkpoints (every 40k steps, both lines), one rollout per checkpoint on the training scene, batches passed through the frozen EU-CRD pipeline exactly as the learner does, then split by the global action taken at each transition (DEFER vs ROUTE): the mean and spread of ρ_routing, the mean-preserving weight w, the advantage before and after reweighting, ΔQ, Δr, c_t and τ. The hypothesis under test is stated in `CODEX_PROMPT_2026_09_05_SEED1_DIAGNOSIS.md` §3 and §7: DEFER transitions receive systematically different responsibility and weight because their Δr is identically zero. Confirmation requires the DEFER/ROUTE difference in w to be of consistent sign across checkpoints after the reweighting warm-up and absent on N_E; anything else is reported as not confirmed.

## 7. What this diagnostic may not do

It may not be re-run with different arms, windows, or thresholds after its numbers are seen; it may not be used to select a new scene; and it may not be cited as evidence about EU-CRD, which it does not involve.
