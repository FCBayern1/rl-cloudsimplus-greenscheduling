# F1–F3 on the causal expert's decisions: reading (2026-09-06)

Preregistration: reports/F_FITS_PREREG.md, frozen at f81328ca. Two harness defects were fixed append-only before any fit result existed and are disclosed: the fit subprocess lacked the dense-grid flag (b032b42f), and the score module had no path for the candidate key at all (it was being concatenated into the context and crashed); the key now enters the offset logits as one learnable gain, initialised at 1.0 (c842884b, tested). Run: `stage_a_out/f_fits`, archive `reports/manifests/f_fits/run1`. Verdict: **ALL_FAIL_OPEN**.

## 1. Corpora

The causal expert's 210 decisions (35 per window) replayed on the offset twin (F1) and on the interface twin with the candidate key from truth (F2) and from TimeCAP (F3); every replay reproduced the expert's carbon to the last digit on all six windows, so labels and observations are aligned. Fit on k0–k3 (140 decisions), held out k4–k5 (70).

## 2. Results (held-out windows, executed as the `option_bc` arm on each interface's own twin)

| interface | fit top-1 on the training windows (epoch 200) | held-out exact action | site accuracy | offset MAE (steps) | executed capture k4 | k5 | pooled | gate (≥ 0.50) |
|---|---|---|---|---|---|---|---|---|
| F1 summaries only | 0.44 | 0.00 | 0.30 | 28.7 | −0.87 | 0.01 | −0.19 | fail |
| F2 candidate key from truth | 0.65 | 0.00 | 0.36 | 28.6 | −0.47 | 0.27 | 0.10 | fail |
| F3 candidate key from TimeCAP | 0.79 | 0.01 | 0.39 | 26.9 | −0.12 | 0.19 | 0.12 | fail |

Contracts green on every execution. The classification gate is INVALID on this corpus as preregistered (70 held-out decisions, 3 in the κ = 1 class) and is descriptive only.

References per held-out window: causal expert 0.000545 / 0.002635 kg, offline flat 0.002100 / 0.007981 kg (k4 / k5).

## 3. Reading

All three fail the executed-capture gate; by the preregistered rule the result is "sample or architecture, reported as open", and RL is not started. The fits memorise the training windows (top-1 up to 0.79) and transfer nothing of the exact (site, κ) to new windows (exact action ≈ 0, offset error ≈ 27 steps out of 72). The candidate key does help the fit (F2, F3 above F1 on every reading) but not enough to execute: on k4 all three are worse than the flat schedule.

What is not yet separated: whether the interface key itself carries the expert's value (a zero-parameter arm that takes the best-covered legal candidate answers this without learning) or whether 140 decisions and a 365-way head are simply too little to clone a per-job exact optimiser. Both diagnostics are outside the frozen gates and are reported in §4 as they complete.

## 4. Diagnostics (appended as run; no gate is changed)

- cover_argmax: [pending]
