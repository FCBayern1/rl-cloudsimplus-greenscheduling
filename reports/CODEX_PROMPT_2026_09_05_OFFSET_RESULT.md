# Fallback (DC, dispatch-offset) after the placement repair: gate 3 PASS, gate 1 FAIL on the window criterion (2026-09-05 22:50)

Context: OPTION_ACTION_DESIGN.md §8 + Addenda A–D; STAGE_D_PRIME_DESIGN.md §34 (forensic and repair) and §35 (this result). Artefacts: `reports/manifests/stage_d/dprime/option/forensic_k5/{,postfix}`, `offset/run1_prefix/`, `offset/run2/`, `p0_prime/run7/`.

## 1. Forensic and repair (your items 1–3)

The per-VM dispatch snapshot on the failed cell proved the selector violated its own most-free-fitting rule: at clock 176 cloudlet 10 went to VM 4, whose scheduler showed exec 0 / waiting 0 while cloudlet 11 (submitted the step before) was still in flight to it, with 15 idle suitable VMs available. The drift 160 PE is the planner's sentinel against the never-recovering allocation counter and carries no occupancy information. Repair confined to the placement ledger (`PlacementLedger.java`: in-flight submissions count as committed; most-free fitting VM, lowest id on ties; none fits → queued as before), five JUnit tests for the four ruled properties, full Java suite green; the k5 replay now dispatches to VM 5 with route→start 0. Run-1 rows archived as pre-fix.

Whole-chain rerun under the repaired jar (b0b44d1e…): P0′ run 7 PASS and bitwise equal to run 6 on all 30 rows (references B/ST unchanged); gate-3 smoke PASS; blind* frozen before any informed row (persistence_off 0.020400; fixed_off_72 0.020924; fixed_off_0 0.020926; …); gate 3 on all 90 rows PASS.

## 2. Gate 1

| window | C_B | C_ST | C_oracle_off | gap/C_B | capture |
|---|---|---|---|---|---|
| k0 | 0.002634 | 0.001476 | 0.001150 | 44.0 % | 1.281 |
| k1 | 0.005079 | 0.003191 | 0.002836 | 37.2 % | 1.189 |
| k2 | 0.003876 | 0.003870 | 0.003736 | 0.2 % | invalid |
| k3 | 0.002718 | 0.000894 | 0.000933 | 67.1 % | 0.978 |
| k4 | 0.001946 | 0.001600 | 0.001894 | 17.8 % | 0.150 |
| k5 | 0.000422 | 0.000309 | 0.000377 | 26.7 % | 0.397 |
| pooled | 0.016674 | 0.011339 | 0.010926 | 32.0 % | **1.077** |

Pooled capture 1.077 ≥ 0.80. Window criterion (A6, ratified in C): ≥ 0.70 on all but one valid window = four of five; read three of five (k4 0.15, k5 0.40, the two windows whose absolute gaps are 0.000346 and 0.000113 kg). **Verdict by the frozen rule: STOP_GATE1_FAIL_ACTION_SPACE_LINE_ENDS.** Gate 2 not read (no blind-versus-oracle number computed); gate 4 not read.

## 3. What I ask

1. Confirm the verdict stands as written and the action-space direction ends (C6), with the report wording of §35.
2. On the record only, not as a request to reread these rows: the window criterion weights a 0.0001 kg gap like a 0.0018 kg gap, and the two failing windows are the two smallest gaps. If a future preregistration on a new scene uses a gap-weighted or absolute-gap window criterion, it must be frozen before that scene's rows exist. These rows cannot be rejudged.
3. Given the option line (31 % capture) and the offset line (pooled 108 %, windows 3/5), the next design question is the scene and the observation interface (Addendum D1), not a third action. I propose the next Codex note be a scene-and-interface proposal rather than more action-space work; please say whether that matches your reading.
