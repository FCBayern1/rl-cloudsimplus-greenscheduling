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

`cover_argmax`: a zero-parameter arm that at each job's first sighting takes the legal (site, κ) with the largest `cand_green_cover` (ties: smallest κ). It reads only the F2/F3 key. A harness defect found on its first run and fixed append-only: the evaluator did not forward `cand_green_cover` (nor the offset legality mask) to baseline arms, so the first run saw no key and routed everything immediately (capture −0.27 on both twins, identical); with the forward in place:

| twin | k0 | k1 | k2 | k3 | k4 | k5 | pooled capture |
|---|---|---|---|---|---|---|---|
| F2 key from truth | 0.751 | 0.802 | 0.798 | 0.812 | 0.596 | 0.713 | **0.765** |
| F3 key from TimeCAP | 0.501 | 0.775 | 0.785 | 0.718 | 0.233 | 0.532 | **0.637** |

Contracts green on all twelve runs. Reading: the interface key itself carries three quarters of the causal expert's headroom with no learning at all, and the TimeCAP version still carries 0.64 pooled (the natural forecast costs about 13 points of capture, most of it on k4). Both zero-parameter figures are above the 0.50 gate the fits failed. The F1–F3 failure is therefore the learner, not the interface: 140 labelled decisions, a 365-way head and a recurrent trunk memorise the training windows (top-1 0.65–0.79) and transfer nothing, even though the module can represent the cover-argmax rule exactly (cover gain large, everything else silent). This is the "sample or architecture" branch of the preregistered reading, now with the two halves separated.

**Disclosure and correction (22:40–23:30).** Checking why the expert's choice had the key's maximum cover in only 50 of 210 decisions exposed a defect upstream of every fit: the Python forecast providers (perturbed_godeye, TimeCAP) read wind rows 12 rows earlier than the simulator burns (the Java provider's COMPRESSED SPLINE skip), and the env's future series added one more row. The candidate key in the F2/F3 corpora therefore described windows 13 rows before the ones the labels were computed on; the F1–F3 fits were trained on observations inconsistent with their labels. Fixed append-only (`simulator_row_shift`, `future_series_from_raw`, alignment tests, commit 475c36b8; design log §54). The corpora were rebuilt on the aligned key (replays still reproduce the expert's carbon exactly) and the diagnostic rerun:

| twin, aligned key | k0 | k1 | k2 | k3 | k4 | k5 | pooled capture |
|---|---|---|---|---|---|---|---|
| F2 key from truth | 1.008 | 1.004 | 0.995 | 0.996 | 0.986 | 1.046 | **1.009** |
| F3 key from TimeCAP | 0.686 | 0.946 | 0.941 | 0.910 | 0.435 | 0.737 | **0.820** |

With the aligned truth key the zero-parameter rule IS the causal expert (pooled capture 1.009; on single-job decisions the expert's objective reduces to the largest cover). With the TimeCAP key it keeps 0.82 of the expert's headroom (k4 0.44): the deployed forecast's natural error costs about 18 points here. The frozen F1–F3 verdict stays on record as produced (ALL_FAIL_OPEN on the misaligned corpora); its interpretation is now: a learner trained on a key that did not describe the labelled windows could not transfer.

Consequences for the next design (decisions, not taken here): any learned policy on this interface must be compared against `cover_argmax` on its own twin, not only against the flat schedule; a fit or a policy should be able to start from the cover rule (large fixed cover gain, learned residual) rather than from a random trunk; more labelled windows are available cheaply (the expert labels a window in about 40 s; the unread 2021 pool has room for about a dozen more windows under the footprint rule). None of this reopens the frozen gates.
