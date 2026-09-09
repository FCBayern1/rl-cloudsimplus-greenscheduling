# Settlement diagnostic done: the 0.47 Wh is fully attributed (2026-09-06)

Read `reports/SETTLEMENT_DIAG_2026_09_06.md` and `reports/manifests/settle_diag/k0_attribution.json`.

| bucket | draw Wh | brown Wh | mechanism, verified how |
|---|---|---|---|
| A extra hosts | +0.206 (740 host-steps) | +0.008 | simulator: one host per job at every k0 step (VmAllocationPolicyCustom spreads VMs, most-free-VM selection); model packs ⌈PEs/64⌉. Probe L2: 131.28 vs 130.28 W |
| B extra job-steps | +0.269 (15 job-steps) | +0.047 | a job starting while its site is mid-cycle is sampled 49 s not 48; rule exact on 35/35 k0 jobs; probes L1 (no), L3a/L3c (yes) |
| C curve tail | 0 | +0.012 | harness: planner truth curve hold-last extrapolated beyond the dump's 599 rows; replay ran to 638 with real wind |
| D DC2 host model | −0.005 | ~0 | DC2 is RS700A (128 PEs, 64.6 W/job); model uses 64-PE hosts everywhere |
| total | +0.470 | +0.068 | equals the ledger |

Per-step power samples sum to the ledger's energy exactly (no integration-phase term). Actual starts are exact; B is end-boundary only. B carries most of the brown because the optimiser ends jobs at the green edge.

Decisions needed (none taken):
1. A: model charges one host per job (matches the simulator) or the simulator consolidates placement. Which side is the physical claim?
2. B: model it (runtime + 1 for busy-site starts, schedule-dependent) or fix the simulator's finish processing; the CloudSim root cause is not yet located. Do you want the source-level trace before deciding?
3. C: harness fix (truth curve from the wind files over the full horizon) is required for any new ladder; confirm.
4. D: per-site host profile in the model; confirm.
5. Only after A–D close on the micro probes: whether to write a new, fully independent preregistration (solver tightening / fixed solver pair, certification workload sized prospectively, closure gates as you specified).
