# Adversarial review of the EU-CRD ICLR 2027 submission

**Reviewer:** Claude (5080 box), 2026-08-22
**Artefact reviewed:** `ICLR_2027_EUCRD.pdf` + `ICLR_2027_EUCRD.zip` (Overleaf export, 2026-08-22 14:24), mirrored at `paper_latest/iclr2027_conference.tex` (byte-identical; note `*.tex` is gitignored, so the tex is NOT in version control).
**Purpose of this document:** a second agent is asked to audit *this review*, not the paper. Every finding below carries its derivation so it can be checked independently. Where I think I could be wrong, I say so under "Challenge this".

---

## 0. What the paper claims

Forecast-conditioned RL schedulers inherit a credit-assignment flaw: the policy gradient charges the acting policy for the forecaster's errors, so training instils unconditional trust in a signal that deployment cannot guarantee. EU-CRD splits each transition's credit between the forecast and the routing action via counterfactual probes, gates how far that split is trusted by critic-ensemble disagreement, and adds a deployment-time correlation auditor.

Headline (Table 1, deterministic decoding, carbon per completed work, completion in parentheses):

| Method | Clean | Blend | Shuffle | FI | C_min |
|---|---|---|---|---|---|
| No-Forecast PPO | 0.196 (99.7) | – | – | – | 99.7 |
| Vanilla PPO | **0.184** (99.7) | **0.184** (100) | 0.269 (100) | 1.46 | 95.6 |
| CVaR-PPO | 0.138† (93.5) | 0.102† (97.7) | 0.051† (48.1) | – | 48.1 |
| Mean-variance | 0.179† (30.6) | 0.154† (18.0) | 0.154† (18.0) | – | 18.0 |
| Risk-sensitive | 0.116† (50.1) | 0.143† (19.8) | 0.177† (19.2) | – | 19.2 |
| Dist-CVaR | 0.130† (91.8) | 0.051† (48.6) | 0.051† (49.6) | – | 48.6 |
| **EU-CRD** | 0.192 (100) | 0.185 (100) | **0.255** (100) | **1.33** | **100** |

† completion below 99.5%, carbon deflated by dropped work.

Structure: 9 pages main text + 11 appendices (A testbed, B training, C algorithm, D per-seed, E sampling decode, F failure modes, G training diagnostics, H component ablation, I runtime auditor, J overhead, K reward).

---

## 1. Verdict

**5 / 10 — marginally below the acceptance threshold. Confidence 4.**

The problem framing is genuinely new and the writing is unusually honest (the paper self-reports its noise floor, an auditor blind spot, and a negative ablation). But the headline evidence is thin by the authors' own stated standard, and **two measurement problems in Table 1 are not confronted anywhere in the paper** (§3 W2, W3 below). Those two are the reason this is a 5 rather than a 6.

---

## 2. How this review was conducted

- Full read of body + all appendices from the compiled PDF and the tex source.
- Cross-checked every number asserted in the body against the table it cites (see §5).
- Recomputed derived quantities (FI, forecast value, ablation deltas) from the tabled values.
- Checked internal consistency: label/reference integrity, environment balance, citation-set integrity, terminology, episode-count claims.
- Checked whether any claim in the paper is contradicted by the paper's own newest evidence (the App H negative ablation).

Not done: reproduction of any experiment; assessment of related-work completeness beyond what is cited.

---

## 3. Findings

Ranked by how much each should move a score.

### W1 (major) — The headline effect is smaller than the paper's own measured noise floor

The 5% claim rests on **4 paired seeds, 3 of 4 favouring EU-CRD**, median +1.7% at the final checkpoint (App D). App D itself states: *"Four pairs give no resolution for an effect of this size."* Baselines use **one seed each**.

Independently, App H states the checkpoint-to-checkpoint spread on this testbed is **10–13%**.

So the paper reports a ~5% headline against a self-reported 10–13% dispersion, with n=4. This is the dominant reason for the score.

*Derivation:* App D per-seed shuffle deltas {+18.8, −5.6, +3.3, +0.0}%; median +1.7%. Table 1 medians give (0.269−0.255)/0.269 = 5.2%.

### W2 (major) — On clean forecasts, EU-CRD captures one third of the forecast value Vanilla captures, and the paper never makes this comparison

All three numbers are in Table 1:

```
No-Forecast PPO  0.196     (reference: no forecast at all)
Vanilla PPO      0.184  ->  forecast value = (0.196-0.184)/0.196 = 6.1%
EU-CRD           0.192  ->  forecast value = (0.196-0.192)/0.196 = 2.0%
```

The paper's §4 says *"the decomposition carries no measurable clean penalty at this sample size"* — but that compares EU-CRD to Vanilla only. **The No-Forecast row sits in the same table and the subtraction is never performed.**

Once performed, the natural reading is that EU-CRD is barely using the forecast on clean inputs. If so, its corruption robustness may come from partial disengagement from the forecast channel — which is precisely what the paper criticises the risk-sensitive baselines for ("make the policy uniformly conservative regardless of forecast quality", Introduction and §2.3). The distinction would be one of degree, not of kind, and the paper's central positioning weakens.

**The paper has a rebuttal available and does not deploy it.** App E (sampling decode, clean): EU-CRD 0.379 < Vanilla 0.397 < CVaR 0.443 < No-Forecast 0.464. At that decode EU-CRD is the best learned method on clean and clearly does use the forecast (14% value vs no-forecast). This belongs in the main text, not buried in an appendix, precisely because it defuses the strongest objection to the paper.

*Why I think this is the single most important finding:* it is the objection a hostile reviewer reaches for, it is derivable entirely from the paper's own headline table, and it is fixable by rewriting rather than by running anything.

### W3 (major) — The fragility index is inflated by a worse denominator

FI = Shuffle / Clean.

```
Vanilla   0.269 / 0.184 = 1.462
EU-CRD    0.255 / 0.192 = 1.328     <- reported advantage
EU-CRD shuffle / Vanilla clean = 0.255 / 0.184 = 1.386
```

EU-CRD's better FI is **partly produced by its worse clean carbon**. Holding the denominator at Vanilla's clean value shrinks the FI gap from 0.134 to 0.076 — roughly half the advantage is denominator-driven.

Structurally, FI rewards a method for being worse on clean inputs. This is in direct tension with the paper's claim that EU-CRD "pays no premium on clean forecasts". Either report numerator and denominator alongside FI, or switch to an absolute degradation measured against the No-Forecast reference.

### W4 (significant) — The runtime auditor, one of three contributions, is never demonstrated on EU-CRD

App I evaluates the auditor **on Vanilla PPO only**. The appendix states the matched EU-CRD checkpoint is excluded because it diverged (training completion fell 100% → 78%, clean deployment completion 16.7%).

Consequence: the paper's "training arm + deployment arm" story is never closed end-to-end on the proposed method. The auditor is presented as a contribution, so demonstrating it on the baseline only is a real gap, not a presentational one.

Cheap to fix: re-run App I's grid on a healthy EU-CRD checkpoint (~2h of evaluation; healthy checkpoints exist — the corruption sweep already uses `creg_eucrd_s2` ck10, whose training completion is 1.0000).

### W5 (significant) — The ablation runs on a different training configuration than the headline

App H base arm: clean 0.189 / shuffle 0.227. Table 1 EU-CRD: clean 0.192 / shuffle 0.255. These are different training campaigns (App H family = `knSV3b`; Table 1 = `eucrd_v4`). The paper now states this explicitly and honestly.

But the consequence stands: **no component of the headline number has been ablated.** A reader cannot attribute Table 1's 5% to the gate, the decomposition, or anything else.

### W6 (moderate) — Internal contradiction on evaluation episode counts

- App B: *"Evaluation replays 10 episodes per checkpoint and condition, and every reported evaluation number is the mean over those 10 episodes."*
- App H text: *"on two seeds at one episode per cell"*.
- App I caption: *"three episodes per cell"*.

Three mutually inconsistent statements. Worse, App H uses 1 episode to measure a 6% difference that the same paragraph says falls inside a 10–13% spread — that cell cannot resolve its own question by construction.

Minimal fix: change App B to say each table states its own episode count in its caption, and add the count to App H's caption.

### W7 (moderate) — The advantage exists only under deterministic decoding

Now stated prominently (abstract, contribution 3, §4) — an improvement. But the substance: under sampling, all forecast-using methods converge under shuffle (Vanilla 0.480, EU-CRD 0.483, risk objectives 0.50–0.51, App E). The operational justification for argmax (capacity planning and SLA certification need the same placement for the same state) is a reasonable deployment convention, not a necessity. The contribution is narrower than the title suggests.

### W8 (minor) — Narrow generality

One testbed (5 DCs, 8,000 jobs, 525 VMs), one forecaster (TimeCAP), two corruption operators evaluated only at full strength. No corruption-severity sweep, no second forecaster, no scale variation.

### W9 (minor) — Terminology leftover

Line 582 (App J) still says `routing window`; §3.1, §3.2 and App I use `control interval`. One-word fix.

---

## 4. Questions for the authors

1. What is EU-CRD's clean-forecast value relative to No-Forecast PPO, and why is it a third of Vanilla's? If EU-CRD genuinely uses the forecast less on clean inputs, what distinguishes it mechanistically from the uniform conservatism attributed to the risk baselines?
2. What does the FI advantage become after controlling for the clean denominator?
3. Can the auditor grid be re-run on a healthy EU-CRD checkpoint?
4. Can the ablation be re-run on the configuration that produced Table 1?
5. Please reconcile the three episode-count statements.
6. Eq. 4 holds the routing action fixed and is now correctly downgraded to a controlled direct effect. How large is the action-mediated remainder under corruption — i.e. how much does a corrupted forecast change the routing action itself?

---

## 5. Consistency checks that PASSED

Recorded so the audit is symmetric — these were checked and found correct.

- No undefined references; all LaTeX environments balanced; 44 unique citation keys, none orphaned.
- Abstract / §4 / Conclusion "5%" ↔ Table 1: (0.269−0.255)/0.269 = 5.2%. ✓
- §4 ablation figures 0.184 / 0.189 / 0.279 ↔ App H table. ✓
- App H "23%" ↔ (0.279−0.227)/0.227 = 22.9%. ✓
- App H per-seed "0.283 and 0.275 against 0.205 and 0.248" ↔ raw eval logs. ✓
- §4 auditor "0.178 against 0.185 clean" ↔ App I table. ✓ (this was stale at 0.182/0.184 earlier today and was corrected)
- **No claim anywhere in the paper asserts that the advantage reweighting is empirically load-bearing**, so App H's negative ablation contradicts nothing. Every mention of reweighting is mechanistic. App G ("the reweighting is active but bounded", σ(w)≈0.33) and App H ("not separately resolvable") are compatible: active ≠ measurably beneficial at n=2.
- Main text is exactly 9 pages (Conclusion ends on p9; p10 begins the AI-use statement, which is not counted).

---

## 6. Recommended actions, by cost/benefit

| Action | Fixes | Cost | Priority |
|---|---|---|---|
| Add the EU-CRD vs No-Forecast clean comparison to §4 and promote App E's sampling result into the main text | W2 | writing only | **highest** — kills the strongest objection with data already in hand |
| Report FI numerator/denominator, or replace FI with degradation vs No-Forecast | W3 | writing only | **highest** |
| Reconcile episode counts (amend App B; add count to App H caption) | W6 | one sentence | do now |
| Fix `routing window` → `control interval` | W9 | one word | do now |
| Re-run the auditor grid on a healthy EU-CRD checkpoint | W4 | ~2h eval | high — converts a contribution from "undemonstrated" to "demonstrated" |
| Re-run the ablation on the Table 1 configuration | W5 | ~11h train+eval | medium |
| Raise ablation to 10 episodes per cell | W6 | ~16h eval | medium |
| Expand main-arm seeds to 8–10 | W1 | very expensive | **do not** — power analysis on the observed deltas {+19.0, −5.4, +3.0, +0.0} (mean 4.16%, sd 10.46%) needs ~50 seeds for 80% power; 8 seeds carry a 63% chance of a worse-looking result. Keep as a stated limitation. |

---

## 7. Challenge this

Points where a second reviewer should push back on *me*:

1. **W2 may be over-read.** EU-CRD clean 0.192 vs Vanilla 0.184 is a 4% gap on n=2–3 iso-eligible seeds, well inside the 10–13% checkpoint spread the paper reports. If that gap is noise, the "one third of the forecast value" framing collapses — the honest statement would be "clean forecast value is not resolved at this sample size for either arm." **I believe the finding still holds as a presentational problem** (a reviewer will do this subtraction whether or not it is significant, so the paper must pre-empt it), but the substantive version of the claim may not survive. Someone should decide which version to put in the rebuttal.
2. **W3's severity is arguable.** FI is a ratio and every ratio has this property. The question is whether the paper leans on FI. It appears in Table 1 as a column and is bolded for EU-CRD, but the prose argument rests on absolute shuffle carbon, not on FI. A reviewer might rate this minor rather than major.
3. **W1's severity vs the paper's honesty.** The paper declines to claim significance, which some reviewers reward and others treat as an admission of insufficiency. I scored it as the latter. Reasonable reviewers differ.
4. **I did not verify the related-work coverage.** If a closer prior work exists on channel-level credit assignment under exogenous-signal corruption, the novelty claim (contribution 1) would need revisiting, and that would matter more than anything above.
5. **I did not check the appendices' numbers against raw logs** except for App H (which I generated). App D, E, F, G, J numbers are taken on trust.

---

## 8. Provenance

- Paper: `ICLR_2027_EUCRD.pdf`, `paper_latest/iclr2027_conference.tex` (untracked; `*.tex` is gitignored).
- Ablation raw data: `local_eval_rt/local_rt_summary.txt`, lines matching `[abl ablBase_*|ablG_*|ablW_*]`; driver `local_eval_rt/run_ablation_base.sh` (untracked, `local_eval_rt` is gitignored).
- Auditor grid raw data: `local_eval_rt/auditor_grid.txt`, `local_eval_rt/auditor_calibrated.txt`.
- LaTeX blocks for the newest additions: `reports/PAPER_ADDITIONS_2026-08-21.md`.
- Reviewer comments this submission responds to: `Comments_ICLR.pdf`.
