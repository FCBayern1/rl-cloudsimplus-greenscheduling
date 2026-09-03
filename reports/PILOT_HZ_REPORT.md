# Scheme 2-HZ pilot: zero-floor fleet, six arms, unclaimed windows k=3,4 — DESIGN_PILOT, not a verdict

Date 2026-09-03. 144 runs = divisor ×{1,2} × 6 arms × 6 cells × windows k∈{3,4} (offsets 3027, 4036; unclaimed by any scheme). Fleet: H 32-PE jobs on SPEC_ASUS_RS500A_DYN / RS700A_DYN hosts (1 W floor, 65.6 W per job at the simulator's 0.4 host utilisation), no cloudlet splitting, planner static floor 0 (`PLANNER_STATIC_TOTAL_W=0`), capacity sentinel 640;512;640;512;192. Brown factor uniform 0.5. Every run: completion 1.000, forced 0, cloudlets = trace rows. Energy per run is the same across arms to within 1% (zero floor), so brown share of job energy is the carbon metric.

Toy predictions (`toy_lever.py`, same wind, same arrivals, p = 65.6 W) were written down before the runs (`toy_hz_prediction.txt`). reservation_edf has no toy analogue (capacity-only blind).

## 1. Brown share of job energy, % — simulator (toy in brackets)

| divisor | cell | k | nowait | reactive_wait | reservation_edf | godeye | shuffle | anti | godeye vs best blind | retention shuffle | retention anti |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ×1 | _c1_n20 | 3 | 31.2 (27.5) | 31.2 (27.5) | 38.8 | 7.6 (1.0) | 23.4 (31.7) | 44.5 (46.7) | +23.5 pp | 0.33 | -0.57 |
| ×1 | _c1_n20 | 4 | 18.5 (23.2) | 8.9 (18.6) | 23.5 | 6.8 (0.4) | 29.7 (25.4) | 20.8 (34.9) | +2.1 pp | -9.96 | -5.67 |
| ×1 | _c1_n50 | 3 | 29.4 (32.4) | 22.5 (23.5) | 36.5 | 6.9 (1.9) | 32.2 (32.6) | 39.2 (42.8) | +15.5 pp | -0.63 | -1.08 |
| ×1 | _c1_n50 | 4 | 23.3 (23.4) | 16.8 (18.6) | 32.3 | 4.5 (0.6) | 33.3 (30.5) | 28.6 (55.1) | +12.4 pp | -1.34 | -0.95 |
| ×1 | _c3_n20 | 3 | 36.5 (31.1) | 32.6 (31.1) | 36.3 | 5.9 (0.8) | 14.8 (25.8) | 40.7 (38.6) | +26.6 pp | 0.67 | -0.31 |
| ×1 | _c3_n20 | 4 | 9.9 (14.4) | 10.1 (14.3) | 27.1 | 4.0 (0.0) | 15.5 (16.7) | 39.3 (31.1) | +5.9 pp | -0.94 | -4.97 |
| ×1 | _c3_n50 | 3 | 25.1 (25.4) | 20.0 (25.3) | 33.5 | 7.4 (1.2) | 31.5 (31.7) | 50.1 (47.2) | +12.6 pp | -0.92 | -2.39 |
| ×1 | _c3_n50 | 4 | 26.1 (22.6) | 17.7 (17.8) | 37.4 | 7.0 (2.1) | 33.3 (27.3) | 34.8 (37.4) | +10.7 pp | -1.45 | -1.59 |
| ×1 | _c5_n20 | 3 | 34.1 (37.5) | 34.6 (36.6) | 43.2 | 10.3 (3.7) | 27.0 (32.5) | 56.1 (56.3) | +23.8 pp | 0.30 | -0.92 |
| ×1 | _c5_n20 | 4 | 27.1 (27.5) | 29.4 (30.2) | 55.2 | 6.9 (0.1) | 20.8 (13.7) | 58.0 (41.5) | +20.2 pp | 0.31 | -1.53 |
| ×1 | _c5_n50 | 3 | 33.4 (32.2) | 31.9 (32.2) | 45.5 | 11.7 (3.7) | 31.4 (32.4) | 51.0 (45.8) | +20.2 pp | 0.02 | -0.95 |
| ×1 | _c5_n50 | 4 | 21.3 (23.8) | 19.0 (22.6) | 35.6 | 7.7 (0.7) | 26.2 (21.4) | 34.7 (35.0) | +11.3 pp | -0.63 | -1.39 |
| ×2 | _c1_n20 | 3 | 36.9 (31.4) | 33.3 (31.0) | 42.3 | 9.6 (5.4) | 34.3 (41.0) | 48.7 (52.9) | +23.6 pp | -0.04 | -0.65 |
| ×2 | _c1_n20 | 4 | 24.0 (28.1) | 19.6 (25.0) | 29.9 | 7.8 (2.6) | 30.6 (33.0) | 34.2 (44.1) | +11.8 pp | -0.94 | -1.25 |
| ×2 | _c1_n50 | 3 | 33.3 (36.1) | 26.3 (25.1) | 41.7 | 8.8 (2.7) | 34.2 (34.0) | 46.5 (55.4) | +17.4 pp | -0.46 | -1.16 |
| ×2 | _c1_n50 | 4 | 29.6 (29.0) | 23.8 (24.1) | 37.2 | 7.2 (1.7) | 34.9 (34.9) | 33.8 (64.9) | +16.6 pp | -0.67 | -0.60 |
| ×2 | _c3_n20 | 3 | 40.6 (35.5) | 35.3 (35.8) | 45.0 | 15.5 (5.1) | 28.5 (26.9) | 48.5 (47.5) | +19.9 pp | 0.34 | -0.66 |
| ×2 | _c3_n20 | 4 | 17.3 (20.6) | 16.3 (22.0) | 35.9 | 6.0 (0.5) | 25.1 (24.7) | 42.7 (48.5) | +10.3 pp | -0.85 | -2.57 |
| ×2 | _c3_n50 | 3 | 31.3 (30.5) | 27.4 (28.0) | 43.7 | 12.8 (5.0) | 38.2 (40.8) | 57.1 (58.6) | +14.6 pp | -0.74 | -2.03 |
| ×2 | _c3_n50 | 4 | 33.2 (30.2) | 20.2 (25.1) | 45.8 | 13.3 (6.6) | 35.5 (33.8) | 44.8 (48.4) | +6.9 pp | -2.23 | -3.58 |
| ×2 | _c5_n20 | 3 | 45.1 (43.6) | 41.7 (38.0) | 58.5 | 20.2 (10.1) | 41.3 (37.8) | 60.2 (60.3) | +21.5 pp | 0.02 | -0.86 |
| ×2 | _c5_n20 | 4 | 38.2 (37.9) | 21.2 (27.5) | 68.9 | 10.5 (5.0) | 26.7 (24.7) | 54.5 (50.6) | +10.7 pp | -0.51 | -3.10 |
| ×2 | _c5_n50 | 3 | 40.1 (41.0) | 37.8 (38.1) | 59.8 | 22.7 (15.3) | 39.5 (38.3) | 60.3 (60.0) | +15.1 pp | -0.12 | -1.50 |
| ×2 | _c5_n50 | 4 | 31.3 (31.8) | 20.9 (22.4) | 50.0 | 13.6 (4.5) | 29.9 (30.2) | 44.2 (48.6) | +7.3 pp | -1.23 | -3.18 |

## 2. Summary

| divisor | godeye beats best blind | median lever | smallest lever | median retention shuffle | median retention anti | sim−toy median abs diff |
|---|---|---|---|---|---|---|
| ×1 | 12/12 | +14.1 pp | +2.1 pp | -0.63 | -1.23 | 3.8 pp |
| ×2 | 12/12 | +14.9 pp | +6.9 pp | -0.59 | -1.37 | 3.4 pp |

Pooled over the 12 (cell, window) runs, brown share of energy and total carbon (kg). Correction 2026-09-03 (Codex item 5): the pooled brown-share column was first printed 100x too large (display only, the per-run table and every verdict quantity were unaffected); values below are recomputed from the raw rows.

| divisor | arm | pooled brown share | pooled carbon kg | vs best blind carbon |
|---|---|---|---|---|
| ×1 | nowait | 26.4% | 0.0517 | +17.2% |
| ×1 | reactive_wait | 22.2% | 0.0441 | +0.0% |
| ×1 | reservation_edf | 37.0% | 0.0711 | +61.2% |
| ×1 | godeye | 7.4% | 0.0171 | -61.3% |
| ×1 | shuffle | 28.6% | 0.0557 | +26.3% |
| ×1 | anti | 40.7% | 0.0776 | +76.1% |
| ×2 | nowait | 33.3% | 0.0643 | +23.7% |
| ×2 | reactive_wait | 26.6% | 0.0520 | +0.0% |
| ×2 | reservation_edf | 46.5% | 0.0884 | +70.1% |
| ×2 | godeye | 12.7% | 0.0267 | -48.7% |
| ×2 | shuffle | 34.2% | 0.0658 | +26.7% |
| ×2 | anti | 47.9% | 0.0907 | +74.4% |

## 3. Reading

- **The forecast is load-bearing on the zero-floor fleet, in every one of 24 (cell, window, scarcity) runs.** The truth-informed planner cuts brown share from roughly 20–40% (best blind) to 4–23%. Pooled carbon falls by about half at both scarcity levels.
- **Errors hurt, and both negative controls fall below the best blind.** Shuffle (lead-0 exact, future permuted) retains a median of −0.6 of the clean gain, anti −1.3: a wrong forecast is worse than no forecast. This is the shape the paper needs, and the first time the S2 family has produced it. (Note: S2 confirmation had shuffle retention 1.06 on the 2-PE fleet with the phantom static floor.)
- **The simulator reproduces the simulator-free model** to a median 3.4–3.8 pp of brown share across 120 comparable runs. The action the forecast enables is capacity planning: placing 48-row jobs across three sites and start times so each site's load stays under its green curve for the whole run, which a present-only policy cannot do because green drops or other jobs arrive mid-run.
- **The strongest blind on this fleet is reactive_wait** (lowest pooled brown share among the blinds at both ×1 and ×2), not reservation_edf, which is capacity-only and now the weakest arm. The formal run must re-freeze the blind on its own discovery set, as Codex's ruling requires.
- What was changed relative to H: host floor removed (1 W twins), planner static floor 0 instead of the hard-coded 332 W. The second change alone halved every arm's brown share on the ×2 smoke cell (46% → 23–25%) and let DC1 receive jobs at all. Every S2/E/F/H godeye result to date was priced against that phantom floor.

## 4. What this pilot does not show

- Windows k=3,4 are design windows; nothing here enters a main table. Discovery windows k=10,18 remain unread; confirmation k=26,34,42 sealed.
- The primary realistic error (calibrated shrink, TIERS_E) was not run here; only the negative controls were. The formal run adds it as the primary corruption.
- The floor is at 1 W. The spiral's next turn puts the real 51.4 W floor back with idle power-down and asks whether the lever survives; the planner then needs a packing-aware static term rather than a constant.

## 5. Recommendation

Register Scheme 2-HZ (append-only prereg, gates G0–G4 as in Codex's ruling, blind re-frozen on discovery, primary shrink + shuffle + anti, k=2 declared read, k=10/18 unread until launch, confirmation one-shot). Launch only after the Codex ruling on this report.
