# Platform of record, decided mechanically (2026-09-10 02:0x UTC)

Applied before any metric of the pair was read, on the three facts Addendum C fixes.

**Announced maintenance deadline**: 2026-09-10 05:00 UTC (06:00 BST), Isambard-AI Phase 2,
docs.isambard.ac.uk/service-status/planned_maintenance.

**Exit status of the six Isambard lines** (job time limit 15:00:00 from 12:31 UTC on 2026-09-09):

| line | finished (UTC) | exit | iterations | env steps | last checkpoint | global module state |
|---|---|---|---|---|---|---|
| V_s20260911 | 2026-09-09 20:31:41 | 0 | 15/15 | 120000 | checkpoint_000014 (27M) | present |
| V_s20260912 | 2026-09-09 20:33:19 | 0 | 15/15 | 120000 | checkpoint_000014 (27M) | present |
| V_s20260913 | 2026-09-09 20:35:44 | 0 | 15/15 | 120000 | checkpoint_000014 (27M) | present |
| E_s20260912 | 2026-09-10 00:49:55 | 0 | 15/15 | 120000 | checkpoint_000014 (489M) | present |
| E_s20260911 | 2026-09-10 00:53:54 | 0 | 15/15 | 120000 | checkpoint_000014 (489M) | present |
| E_s20260913 | 2026-09-10 02:00:41 | 0 | 15/15 | 120000 | checkpoint_000014 (489M) | present |

No line reported a traceback or an out-of-memory failure. All six completed **before** the
maintenance deadline, the last with 2 h 59 min to spare, and before the 15 h job limit.

**Decision**: the condition of Addendum B is met, so the **Isambard-AI Phase 2 set is the run of
record**. No result figure entered this decision; only the deadline, the exit statuses and the
checkpoint integrity did.

**Consequence for the workstation run**: its purpose as the hedge is discharged. It is stopped
at this point (V_s20260911 and E_s20260911 complete, V_s20260912 partial) so the workstation can
run the frozen evaluation of the run of record. Its completed pair is kept as an artefact; it is
not a seed set (a seed set is complete only if all six of its lines come from one platform) and
it is not read as an alternative headline. A cross-platform replication, if wanted later, is a
fresh registered run, not this partial one.

**Cost on the platform of record**: E lines 2 972–3 225 s per 8 000 steps, V lines 1 910–1 931 s,
against 1 588 s (E) and 819 s (V) measured on the workstation for the completed pair there.
