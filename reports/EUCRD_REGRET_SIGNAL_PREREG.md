# EU-CRD forecast responsibility as local decision regret: definition and gates (frozen 2026-09-07)

Frozen by the commit that adds this file, before the signal is implemented and before it is measured. It replaces `candidate_cover_mae` as the responsibility magnitude following the ruling on reports/EUCRD_SIGNAL_GATE_READING_2026_09_07.md Addendum A. `STOP_SIGNAL_GATE:G3_monotone_in_error` stands and is not reinterpreted. No carbon claim, no policy comparison and no training happens under this document.

## 1. The quantity

At one decision state, with the current reservation grid, the current job batch and the current legal candidate set all held fixed, for each job j:

| symbol | meaning |
|---|---|
| `cov_pred[j,c]` | share of j's dynamic energy covered by green if placed at candidate c, computed on the arm's forecast |
| `cov_true[j,c]` | the same quantity computed on the simulator's hidden future |
| `E_j` | j's dynamic energy, `pes_j × 2.02 W × runtime_steps_j × timestep_sec`, in kWh |
| `bf[d]` | brown carbon factor of the site of candidate c, kg per kWh |

The cost model is the same for both, and is the one the offset actor already consumes:

    cost_x(j,c) = (1 − cov_x[j,c]) · E_j · bf[site(c)]        x ∈ {pred, true}
    a_pred(j)   = argmin over legal c of cost_pred(j,c)
    a_true(j)   = argmin over legal c of cost_true(j,c)
    R_f(j)      = cost_true(a_pred(j)) − cost_true(a_true(j))   ≥ 0
    signal      = Σ_j R_f(j)                                    kg CO₂

Ties in both argmins break to the smallest candidate index a = site·|K| + κ_index, the order `torch.argmax` uses on equal logits and the order the `cover_argmax` arm uses under `COVER_TIE=index`. Both choices are settled on the truth, so a forecast that is numerically wrong without changing the chosen candidate carries no responsibility, and a forecast that changes it carries exactly the carbon that change costs.

Scope, stated as a limitation and not as a claim: this is the **local** decision regret at the current state under the current reservation grid. It is not the causal carbon loss of the trajectory. Opportunity destroyed by earlier decisions is not recoverable by a score at the current step, and no result under this signal may be read as measuring that.

Source name `candidate_carbon_regret`, selected by `crd.forecast.source`, learner-only, config-gated, default unchanged (`instantaneous_carbon_cf`). Unit kg CO₂; `crd.responsibility.normalize_shares` divides each channel by a running estimate of its own magnitude, so the share is scale-free, which a unit test asserts rather than assumes.

## 2. Unit-level requirements (must hold before any run)

- **U1** truth in, zero out: `cov_pred = cov_true` gives exactly 0.
- **U2** non-negative for every input: randomised forecasts never produce a negative value.
- **U3** tie insensitivity: a forecast that selects a different candidate of equal true cost gives exactly 0.
- **U4** hand-computed micro example: one job, two candidates, covers 0.9/0.1 predicted against 0.2/0.8 true, 64 PEs, 10 runtime steps, 1 s steps, bf 0.5, expected regret `(0.8 − 0.2) · E · 0.5` with `E = 64 · 2.02 · 10 / 3.6e6` kWh = 1.0774e-4 kg, asserted to 1e-12.
- **U5** shape or field mismatch raises rather than returning a default.
- **U6** no silent fallback: when `crd.forecast.source` is an explicitly selected candidate source and the auxiliary field is absent, malformed or misshaped, the learner raises. The historical `instantaneous_carbon_cf` path keeps its existing fallback behaviour.
- **U7** the auxiliary field reaches the learner channel and never the policy observation.
- **U8** rescaling the signal by a constant leaves the responsibility shares unchanged once the scale estimate has converged.

## 3. Instrument gates, zero training

Arm `cover_argmax`, tiers godeye, shrink75, shrink50, shrink25, shrink0, the first three reading windows, one run per (tier, window) with the regret source and one control run identical except the source.

- **H1** godeye: max per-step signal ≤ 1e-12 on every step of every window.
- **H2** shrink75: mean per-step signal > 0 on every window.
- **H3** non-negativity on the runs: min per-step signal ≥ 0 everywhere.
- **H4** fixed-state paired comparison: on the **godeye arm's own trajectory**, with its reservation grid and its job batches held fixed, the signal is recomputed under each tier's forecast. Every degraded tier must give a pooled mean strictly greater than godeye's zero. The ordering across the degraded tiers is reported but **not** gated: different tiers on the same trajectory may reorder without that being a defect, and the ruling explicitly does not require monotonicity.
- **H5** no observation leak: every policy-visible observation key is bit-identical between the regret run and its control on all 15 (tier, window) pairs, and no `crd_*` key appears in the policy observation.

Reported, not gated: the per-tier signal along each tier's own trajectory, and the empty-reservation-grid candidate-cover mean absolute error as an auxiliary forecast-quality scale. That auxiliary quantity is never used as a denominator; near-zero coverage spread would make such a ratio meaningless.

Any H failure is STOP_REGRET_SIGNAL and the source is not trained on. Nothing is tuned to obtain a pass.

## 4. Disclosure

The truth curve enters only the learner-side auxiliary channel, as supervision the simulator can provide and a deployment could not. H5 shows that the actor's observation in the gated runs is unaffected. That is a statement about these runs, not a statement that the training procedure as a whole uses no future information, and the paper must disclose the channel rather than rely on H5 to deny it.

## 5. Order

U1–U8, then H1–H5, then G5 (the real-learner wiring gate, reports/EUCRD_SIGNAL_GATE_PREREG.md §2 as amended by its Addendum A), then and only then the matched V_err / E_err training on an identical imperfect forecast.
