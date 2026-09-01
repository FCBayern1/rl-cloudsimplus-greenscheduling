# Round 1-v3: STOP_EVPI_GATE_NOT_MET

Phase A froze `immediate_current_only` by pooled carbon over all 1,728 cohort cells.
Phase B solved the exact model on every cell and proved optimality on all of them.

    EVPI >= 15% of total carbon:   0 / 1728
    advancing cells:               0
    advancing blocks:              0
    unresolved cells:              0

The registration is not relaxed and no cell is dropped. The artifacts, the frozen arm and
the cohort stay as they are; v1 and v2 remain untouched and their numbers do not stand in
for this verdict.

A decomposition reported beside the verdict, and not part of it: the schedule-independent
idle floor is about 96% of the carbon ledger in this scenario, so the movable part is a
median 3.8% of the total. Whether a new registration should change the power constants or
the denominator is a scenario-design ruling, not a change to this round.

    commit      0d05214
    cohort      241764512eb5c658591f3a46
    preflight   PASS (11/11)
