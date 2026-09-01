# Round 1-v4: STOP_EVPI_GATE_NOT_MET — TB13 line closed

Read by the verdict reader frozen at commit 16d23ff, before this artifact existed.

    cells / ids               1,728, exact match with cohort b22a5d79d79b66c6c7cac8ca
    proven OPTIMAL            1,550
    UNRESOLVED                178   (never counted as advancing, per the ruling)
    EVPI (total carbon)       p50 5.57%   p75 10.90%   p90 17.23%   max 56.97%
    EVPI >= 15%               235 cells
    advancing (all gates)     149 cells across 48 blocks
    complete 12-cell block    0   -> STOP

Per the registered termination clause the TB13 scenario search ends here. No v5, no
constant tuning. The three prior STOPs and this one stay as they are.

The recalibrated physics moved every quantile exactly as the arithmetic predicted
(p50 1.14% -> 5.57%, p90 4.43% -> 17.23%, max 11.51% -> 56.97%), and 235 cells cleared
the 15% line. What failed is neighbourhood robustness: no anchor held all three divisor
neighbours and all four budgets above every gate at once. The strongest blocks reached
6 of 12, losing the rest mainly to the EVPI gate on their weaker corners.

Post-verdict diagnostics (do not rewrite this verdict): the 178 UNRESOLVED cells sit
mostly at T=144 with 32-PE jobs, where two 30 s CP-SAT stages were not enough under a
2x oversubscribed machine; a diagnostic re-solve with a larger budget may classify them
but cannot change this round's outcome.

    phase B wall              2,656 s
    verdict artifact          round1_v4_verdict.json
