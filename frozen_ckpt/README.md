# Frozen spatial base (shared across machines)

`v32_nofc600_s1_ck10/` is the V3.2 blind PPO route-only checkpoint that every
temporal arm of the SQT2 / gwo1 ladders shares. It was trained on the 5080
under `drl-manager/logs/` (gitignored), which left the 3060 unable to run the
formal PPO-base verdict at all.

It is tracked here — 13 MB — because "frozen shared base" is a scientific
requirement, not a convenience: both machines must load the *same bytes*, and
a git object plus the `.sha256` manifest makes that checkable rather than
assumed. `drl-manager/tests/test_frozen_ckpt_resolver.py` verifies every file
against the manifest and, on the machine that owns the training copy, that the
two paths are byte-identical.

Provenance: `logs/v32_nofc600_s1/multidc_gtrxl_training/
PPO_multidc_env_70179_00000_0_2026-08-17_01-34-42/checkpoint_000010`
(600k steps, blind/no-forecast recipe, seed 1, final checkpoint).

Resolution order in code (`sqt2_prescreen.resolve_blind_ck`): the training
path when present, else this copy. Never hard-code either.
