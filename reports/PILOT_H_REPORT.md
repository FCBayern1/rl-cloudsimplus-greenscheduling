# Scheme 2-H pilot (32-PE jobs on the F variant) — DESIGN_PILOT, not a verdict

Date 2026-09-03. 54 runs: divisor ×{1,2,4} × {reservation_edf (blind), godeye, shuffle} × 6 cells, one discovery window (k=2). Completion 1.000 and forced 0 in all 54 runs. Cloudlet count equals trace rows in every run (splitting disabled, commit dde2f205).

## 1. Registered hypothesis and its check

Registered (S2_E_F_INTERIM §6): with dynamic/floor ≥ 1 (32-PE job 81 W vs 51.4 W idle floor) the fragmentation cost of chasing green becomes negligible, so forecast timing may become load-bearing.

Check: energy of godeye relative to blind (fragmentation tax, floor-seconds).

| divisor | cell | blind Wh | godeye Wh | tax |
|---|---|---|---|---|
| ×1 | _c1_n20 | 31 | 36 | +16% |
| ×1 | _c1_n50 | 79 | 88 | +13% |
| ×1 | _c3_n20 | 32 | 37 | +16% |
| ×1 | _c3_n50 | 79 | 97 | +23% |
| ×1 | _c5_n20 | 31 | 37 | +18% |
| ×1 | _c5_n50 | 78 | 95 | +22% |
| ×2 | _c1_n20 | 31 | 38 | +21% |
| ×2 | _c1_n50 | 79 | 88 | +12% |
| ×2 | _c3_n20 | 32 | 41 | +31% |
| ×2 | _c3_n50 | 79 | 94 | +19% |
| ×2 | _c5_n20 | 31 | 39 | +25% |
| ×2 | _c5_n50 | 78 | 99 | +26% |
| ×4 | _c1_n20 | 31 | 31 | -0% |
| ×4 | _c1_n50 | 79 | 83 | +5% |
| ×4 | _c3_n20 | 32 | 32 | -0% |
| ×4 | _c3_n50 | 79 | 78 | -0% |
| ×4 | _c5_n20 | 31 | 30 | -5% |
| ×4 | _c5_n50 | 78 | 77 | -2% |

The tax is +15…+25% at ×1 and ×2 (absolute +5…+21 Wh, larger than the +6/+14 Wh seen with 2-PE jobs) and vanishes only at ×4 where nothing waits. **The hypothesis is refuted: wide jobs do not make fragmentation free.** What changed is the other side of the ledger: a 32-PE job moved onto green now shifts 81 W, so the green-mix gain can outrun the floor tax where green is plentiful.

## 2. Three-arm carbon triangle (kg CO₂, one window each)

| divisor | cell | blind | godeye | shuffle | godeye vs blind | godeye vs shuffle | shuffle vs blind | shuffle retention |
|---|---|---|---|---|---|---|---|---|
| ×1 | _c1_n20 | 0.00319 | 0.00463 | 0.00602 | +45.2% | -23.1% | +88.9% | 1.97 |
| ×1 | _c1_n50 | 0.01148 | 0.00824 | 0.01576 | -28.3% | -47.7% | +37.3% | -1.32 |
| ×1 | _c3_n20 | 0.00429 | 0.00220 | 0.00620 | -48.8% | -64.6% | +44.5% | -0.91 |
| ×1 | _c3_n50 | 0.01774 | 0.01982 | 0.01932 | +11.7% | +2.6% | +8.9% | 0.76 |
| ×1 | _c5_n20 | 0.00737 | 0.00329 | 0.00749 | -55.3% | -56.0% | +1.6% | -0.03 |
| ×1 | _c5_n50 | 0.02279 | 0.02100 | 0.02011 | -7.9% | +4.4% | -11.8% | 1.50 |
| ×2 | _c1_n20 | 0.00687 | 0.01090 | 0.00849 | +58.6% | +28.4% | +23.5% | 0.40 |
| ×2 | _c1_n50 | 0.01689 | 0.01517 | 0.01692 | -10.2% | -10.4% | +0.2% | -0.02 |
| ×2 | _c3_n20 | 0.00967 | 0.01200 | 0.00858 | +24.0% | +39.8% | -11.3% | -0.47 |
| ×2 | _c3_n50 | 0.02818 | 0.03244 | 0.02760 | +15.1% | +17.5% | -2.0% | -0.14 |
| ×2 | _c5_n20 | 0.01125 | 0.01170 | 0.00945 | +4.0% | +23.9% | -16.0% | -4.00 |
| ×2 | _c5_n50 | 0.03089 | 0.03772 | 0.02811 | +22.1% | +34.2% | -9.0% | -0.41 |
| ×4 | _c1_n20 | 0.01109 | 0.01155 | 0.01181 | +4.2% | -2.1% | +6.5% | 1.55 |
| ×4 | _c1_n50 | 0.02444 | 0.01907 | 0.02435 | -21.9% | -21.7% | -0.3% | 0.02 |
| ×4 | _c3_n20 | 0.01270 | 0.01338 | 0.01216 | +5.4% | +10.0% | -4.3% | -0.80 |
| ×4 | _c3_n50 | 0.03370 | 0.03433 | 0.03373 | +1.9% | +1.8% | +0.1% | 0.04 |
| ×4 | _c5_n20 | 0.01341 | 0.01332 | 0.01282 | -0.7% | +3.8% | -4.4% | 6.38 |
| ×4 | _c5_n50 | 0.03505 | 0.03501 | 0.03457 | -0.1% | +1.3% | -1.4% | 12.45 |

| divisor | godeye beats blind | median godeye vs blind | godeye beats shuffle | median godeye vs shuffle | median shuffle vs blind | median shuffle retention |
|---|---|---|---|---|---|---|
| ×1 | 4/6 | -18.1% | 4/6 | -35.4% | +23.1% | 0.36 |
| ×2 | 1/6 | +18.6% | 1/6 | +26.1% | -5.5% | -0.27 |
| ×4 | 3/6 | +0.9% | 2/6 | +1.5% | -0.8% | 0.79 |

## 3. Reading

- **×1 (green oversupplied, F brown factor uniform 0.5):** godeye beats blind in 4/6 and beats shuffle in 4/6, with large margins in the three light-load n20 cells (−23…−65% vs shuffle) and in c1_n50. Median shuffle retention 0.36 (S2 confirmation had 1.06). A wrong forecast (shuffle) is worse than no forecast in 5/6 cells. This is the first cell set in the S2 family where the forecast's timing content, not the act of waiting, carries the gain.
- **×2 (moderate scarcity):** godeye loses to blind in 5/6 (median +18.6%) and to shuffle in 5/6. Chasing green when green is short fragments more than it captures. The consolidation mechanism of §6 still rules here.
- **×4 (deep scarcity):** all three arms within ±5%. No lever.
- Per-cell scatter is wide (godeye vs blind at ×1 ranges +45% to −55%) and every number is one window. Under the eval noise-floor record (cross-window 10–16%) none of the per-cell numbers is individually trustworthy. Only the ×1 pattern across 6 cells and 2 comparators is a signal.

## 4. Classification under the pilot stop rule

The rule was: sweet spot found → register a full Scheme 2-H; still all-lose → close the temporal lever for this simulator family. The outcome is neither. ×2 and ×4 are all-lose or null. ×1 is a candidate sweet spot with the right shape (godeye < blind < shuffle in most cells) but one window per cell and a refuted mechanism story. Pilot results are design-only and enter no table.

Recommendation: register Scheme 2-H restricted to ×1 (and ×1.5 as a scarcity edge), the E protocol unchanged (discovery windows k=2/10/18 on the discovery cells, TIERS_E ladder incl. shuffle and anti, blind = reservation_edf, gates 1–5 frozen before data, confirmation k=26/34/42 one-shot and unseen). Gate 1 (godeye beats blind, median across discovery cells×windows, ≥ 2/3 of cells) decides whether the family stays alive. Launch only after the Codex ruling on this report and on §6 of the interim report.

## Addendum A (2026-09-03, Codex R-e / R-h)

The truth-informed arm in this pilot priced green net of the planner's hard-coded 332 W fleet static (see `S2_E_F_INTERIM_FOR_CODEX.md` Addendum B). The H-×1 formal gate proposed in §4 was withdrawn by Codex ruling R-h; the zero-floor scene (`PILOT_HZ_REPORT.md`, `SCHEME2_HZ_PREREG.md`) replaces it. The 32-PE job draw on this legacy fleet was 51.4 W floor + 0.4 × 162.6 W = 116.4 W at the simulator's 0.4 host utilisation, not the 132.7 W (0.5 utilisation) written in §1.
