# Paper plan, assuming Stage D comes out well (2026-09-04)

The paper already exists as a complete ICLR-format draft (`paper_latest/iclr2027_conference.tex`, 622 lines, method and appendices written). What changes is not the structure but the evidence underneath it: the draft is anchored on the old five-datacentre C-regime testbed, and every number there predates the power-semantics fix, the phantom planner static, the ledger-aligned reward and the preregistered HZ scene. This plan is therefore a re-anchoring plan, not a writing-from-scratch plan.

## 1. The claim

One sentence: **a realistic, systematic forecast error makes a carbon-aware scheduler worse than using no forecast at all; a trained policy inherits that harm; and crediting each decision only for the part of the outcome it controlled removes at least half of it, at no cost when the forecast is clean.**

Three steps, each with its own preregistered gate:

| step | claim | evidence | status |
|---|---|---|---|
| 1 | the error harms an analytic scheduler | HZ discovery + one-shot confirmation | **done, PASS** |
| 2 | a trained policy inherits the harm | Stage D gates 1–2 | running |
| 3 | EU-CRD removes ≥ half without a clean-side cost | Stage D gates 3–5 | running |

Two supporting results that the current draft does not have and that reviewers of a "we added forecasts" paper always want:

- **When is a forecast load-bearing at all.** The zero-training decomposition (run-now / present-only / oracle) bounds the forecast's value before any policy is trained, and identifies the structural conditions under which it is worth anything: contested green, jobs long relative to the wind's decorrelation time, and waiting that costs something. Five documented scenes where those conditions fail, and the sixth where they hold.
- **That the reward can price the objective at all.** The P0 reward truth table: replaying four known behaviours under the training reward and requiring the reward ordering to equal the physical-carbon ordering. The inherited reward failed it; the ledger-aligned reward passes 6/6 windows.

## 2. Section-by-section: keep, replace, add

| section | action | notes |
|---|---|---|
| Abstract | **rewrite** | new testbed, new primary corruption, new containment number; drop "five-datacentre testbed", keep the trust-as-liability framing which survives intact |
| 1 Introduction | **light edit** | the motivating failure changes from a mis-mapped site feed (shuffle) to a calibrated amplitude error measured from the deployed forecaster; that is a stronger opening |
| 2 Related work | **add one subsection** | preregistration and positive controls in systems ML; keep the three existing subsections; **restore CCA-PG in the text**, it is currently only in related work |
| 3 Method | **keep almost entirely** | the decomposition, the epistemic gate, the blender and the auditor are unchanged; add the responsibility-share normalisation and anomaly gate (v5 fixes) which the draft predates |
| 3.x **new**: when a forecast is load-bearing | **new section** | the decomposition, the structural sweep, the conditions; ends with the HZ scene as the positive control it is |
| 4 Experiments / Setup | **replace** | HZ scene, six cells, frozen blind, judgement windows, ledger-aligned reward, five paired seeds, two hardware platforms |
| 4.x **new**: protocol | **new subsection** | preregistration, sealed confirmation, blind freeze before any informed number, the reward truth table, the mechanical readers |
| 4 Main comparison | **replace numbers** | table specs in §4 below |
| 4 Risk-sensitive comparison set | **decide, see §6** | those four baselines were run on the old testbed only |
| Appendix testbed / training | **replace** | new fleet, new power semantics (RS500A 51.4/214 W and the zero-floor twin), new reward |
| Appendix per-seed, stochastic decode, failure modes, diagnostics, ablation, auditor, overhead, reward design | **keep the structure, refill** | the ablation and auditor appendices are still valid method content |
| **new** appendix: negative results | **new** | the five STOPs, honestly reported; this is what makes the positive control credible |

## 3. Evidence inventory

Landed and citable now:

- HZ confirmation: clean −42.5% pooled carbon intensity against the frozen strongest blind, 6/6 cells and 3/3 windows; calibrated shrink +154% against clean and +46% against the blind; shuffle +45%, anti +64.5%. Discovery: −28.8% / +101%.
- The lever decomposition and the 192-configuration structural sweep.
- P0: legacy reward inverts the ordering (truth-informed scores 6.7× worse while emitting 31% less carbon); ledger-aligned passes every ordering check.
- Health smoke PASS: chain end to end, policies alive, forecast channel connected, EU-CRD internals active.
- Cross-platform equivalence: identical initial weights on x86 and aarch64, same judgement window replayed, all gate fields present.

Running: Stage D (four lines × five seeds, two platforms) and Stage D-B (CCA-PG, two lines × five seeds).

Missing, and worth deciding on now: the risk-sensitive comparison set on the new scene (§6).

## 4. Tables to produce

1. **Main.** Rows: frozen blind planner, no-forecast RL (N_V), vanilla RL (V), CCA-PG (C), EU-CRD (E). Columns: clean, calibrated shrink, shuffle, anti, each as pooled carbon intensity with completion in parentheses; plus Δ_corr (shrink minus clean) and the containment fraction against vanilla's Δ_corr. Medians over five seeds, per-seed values in an appendix.
2. **Gates.** The five gates with their thresholds, per-seed pass counts, and the direction counts. This table is the preregistration made auditable and belongs in the main text.
3. **Analytic layer.** The HZ confirmation table (blind / clean / shrink / shuffle / anti), which establishes step 1 independently of any policy.
4. **Reward truth table.** Four replayed behaviours × (carbon, reward) under the inherited and the ledger-aligned reward. Small, and it forecloses "your reward was broken".
5. **Overhead.** Decision latency and memory, EU-CRD against vanilla, as the project's own rule requires: effect and cost in the same report.

## 5. Figures

1. Concept: three sites, wind curves, a job placed under the green curve versus placed where the forecast wrongly promised wind.
2. When a forecast is load-bearing: the structural sweep, forecast-only lever against job length and slack; the six failed scenes marked.
3. Analytic layer: HZ confirmation bars, blind / clean / shrink / shuffle / anti.
4. Method schematic: observation → policy and critic ensemble → ΔQ, Δr, R_forecast → epistemic gate → responsibility shares → advantage reweighting, with the forecast channel marked quarantined.
5. Main result: containment, vanilla's Δ_corr against EU-CRD's, five seeds.
6. Robustness curve: clean → shrink → shuffle → anti for vanilla and EU-CRD, with the no-forecast lines as horizontal references.
7. Mechanism: σ², c(t) and ρ_forecast over training, and ρ_forecast under clean versus corrupted deployment.
8. Two-platform replication: the same gate outcomes on the workstation and on the cluster.

## 6. Decisions needed

1. **Risk-sensitive comparison set.** The draft compares against CVaR, mean-variance, risk-sensitive utility and a distributional CVaR variant, all on the old testbed. On the new scene they do not exist. Options: (a) submit four more lines × five seeds now while the cluster is idle, which is twenty more single-GPU jobs of about twenty hours, and keep the comparison set intact; (b) drop them and lead with no-forecast, vanilla and CCA-PG, arguing that a credit-assignment paper needs a credit-assignment baseline rather than risk objectives; (c) keep them as an appendix on the old testbed, clearly labelled as a different scene, which I would not do because the project's own rule forbids cross-scene tables. My recommendation is (a) if the goal is the strongest submission, (b) if the goal is speed.
2. **Venue and deadline**, which sets everything else. The draft is in ICLR 2027 format.
3. **Whether the negative-results appendix goes in.** I recommend yes: without it the positive control looks cherry-picked, and with it the paper has something most systems-ML papers cannot claim.

## 7. Schedule, assuming gates pass

| when | what | blocked on |
|---|---|---|
| now | protocol and scene-design sections, testbed appendix, tables 3 and 4, figures 1–4 | nothing, all evidence landed |
| Stage D lands (Fri–Sat) | tables 1, 2, 5, figures 5–8, results prose | the verdict |
| +1 day | abstract, introduction, conclusion rewritten around the measured containment | tables |
| +2 days | related work update, negative-results appendix, limitations | nothing |
| +3 days | full pass for the honesty ledger below, internal review | draft complete |

## 8. Honesty ledger, to be written into the paper rather than discovered by a reviewer

- The scene is an accelerated-weather, marginal-carbon **mechanism positive control**: one row of wind per step, a 1 W host floor. It establishes the mechanism, not a deployment saving. The real 51.4 W floor is the next turn and is named as future work.
- The HZ confirmation rests on 17 of 18 grids; one corrupted-arm run finished all jobs with 98% on time. Under the registered rule the grid is voided; under a stricter "every run green" reading the gate fails. Both readings are reported.
- The training reward was changed once, before any training, on the evidence of the reward truth table, and the change zeroes two terms that are not functions of physical carbon rather than tuning any weight.
- Five scenes were stopped before this one. Each stop is reported with its reason.
- The truth-informed planner is called truth-informed, never an oracle or an optimum: shuffle beat it in an earlier scene.
- The first Isambard submission and the first health smoke were voided for wiring, with the data archived and unread.
