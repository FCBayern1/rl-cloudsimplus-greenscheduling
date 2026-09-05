# Fallback (DC, dispatch-offset): gate 3 stopped on one row; ruling needed (2026-09-05 22:30)

Context: OPTION_ACTION_DESIGN.md §8 + Addenda A/B/C/D (all committed before any fallback row), STAGE_D_PRIME_DESIGN.md §33. Chain ran once in the C6 order; nothing was tuned.

## What happened

1. Gate 3 smoke (k0, 12 blind arms): PASS. Six-window blind rows (72): all clean. blind* frozen before any informed row: persistence_off (pooled carbon 0.020480; fixed_off_72 0.020924, fixed_off_0 0.020926, …, climatology_off 0.027672), file `stage_a_out/offset_blind_star.json`.
2. Informed rows (oracle_off / shuffle_off / anti_off, 18) generated. Judge 3 → 1 → 2 stopped at **gate 3** on exactly one of 108 rows: persistence_off, window k5, cloudlet 10 (32 PEs, runtime 48, κ = 16): released to DC 0 at step 175, execution start 224.03, route→start 49 steps (gate: ≤ 1). Everything else on that row and on all other rows is clean (completion 1.0, on-time 1.0, forced 0, no refused/masked offsets, ledgers closed). Gates 1, 2 and 4 were not read.

## What the diagnosis established (deterministic replay with per-step observation dump)

- The replay reproduces the 49.03-step delay exactly.
- The executor's reservation grid is not the cause: at step 176 DC 0 (20 VMs × 32 PEs = 640 PEs) carried at most five 32-PE jobs; the grid had room and so did the simulator (its own free-PE reading 512 of 640; DC queue length 0 throughout the wait).
- Queue length 0 during the wait means the cloudlet was assigned to a VM and waited in that VM's space-shared scheduler behind a running cloudlet, i.e. the simulator's VM-level placement in dispatch-rate mode put it on a busy VM although idle VMs existed. The placement code counts committed PEs from each VM's exec + waiting lists and picks the most-free fitting VM; which VM it chose here and why is still being traced (the gateway's per-cloudlet log is off in these runs).
- This placement path is shared by every arm (analytic or RL, option or offset) and by the step-wise references B and ST; it never fired in the P0′ rows, the margin probe (max 1 s) or the 54 option rows, because the option executor checked the simulator's free PEs before releasing. The offset executor, by C1, releases on time only, so it is the first design exposed to it.

## Rulings requested

1. Classification: is this an instrument defect of the shared simulator (VM placement) rather than a flaw of the fallback design? My reading: instrument defect, common to all arms.
2. Does the fallback get its own single gate-3 repair under §6 (the option line used its one repair on the episode-termination hole)? If yes: the repair is confined to the placement logic (a cloudlet must not be queued behind a running cloudlet while an idle VM that fits exists), unit-tested, disclosed, and applied to nothing else.
3. Scope of the rerun after a repair: my position is that the placement change touches every arm, so the whole fallback chain reruns from the gate-3 smoke (blind rows, a fresh blind* freeze, informed rows), and the step-wise references B and ST (P0′ run-6 rows) are regenerated under the repaired jar as well, with the original rows archived. Confirm or restrict.
4. If the answer to 2 is no: the line ends at gate 3 as written, and the report says the fallback was not judged on expressibility or necessity. Confirm the wording.

No number from gates 1, 2 or 4 has been read and none will be until gate 3 passes on every row.
