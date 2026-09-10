# Invalidated launch

The run launched at 2026-09-10 20:35 Europe/London was stopped at 20:39 before
its first iteration completed.  Only `checkpoint_init` was produced and no
training carbon result was read.

Reason: frozen gate G-c is mathematically unsatisfiable under the generated
configuration's homogeneous carbon factors.  See
`reports/GATED_RESIDUAL_GATE_PREFLIGHT_STOP_2026_09_10.md`.

This is `STOP_GATE_DEFINITION_UNSATISFIABLE`, not a result about the anchored
gated residual mechanism.  Any continuation requires an append-only
registration decision made before a replacement run.
