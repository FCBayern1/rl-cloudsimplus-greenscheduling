# Where things are

A short map of a repository that is both a development environment and an experiment archive. It says what is *current*, what is *evidence*, and what is *history*. Anything not listed here is history.

## Current entry points

| what | how |
|---|---|
| train a line | `drl-manager/entrypoint_rlmodule_gtrxl.py --config <config> --experiment <block> --total-timesteps N --seed S --num-workers 0 --no-wandb --output-dir <dir>` |
| deploy a checkpoint | `python -m src.baselines.evaluate --experiment <block> --global rllib --new-api --checkpoint <ckpt> --local drain --episodes 1 --seed 42 --reset-skip <window>` (run from `drl-manager/`) |
| run the tests | `drl-manager/.venv/bin/python -m pytest tests/ -q` (from `drl-manager/`) |
| Java gateway | built once with `cloudsimplus-gateway/gradlew installDist`; the runners point `GATEWAY_LIBS` at `build/install/cloudsimplus-gateway/lib` |

Environment every runner sets: `PLANNER_EXPECTED_CAP`, `PLANNER_STATIC_TOTAL_W=0`, `OFFSET_GRID_DENSE=1`, `ORACLE_WIND_DIR`. On a cluster also `RAY_TMPDIR` — short and node-local, because Ray's Unix sockets must stay under 107 bytes.

## Current scene and configs

The live scene is the HZ ×2 zero-floor testbed (turbines 133/78 | 22/81 | 94, wind year 2021, COMPRESSED mode). Configs live in `g1/compressed_timecap_s2/`:

- `config_rl_v2_cpu.yml` — training twin (the V and E blocks the matched pair derives from)
- `config_rl_v2_eval.yml` — evaluation twin (one block per forecast tier)
- generated configs (`config_matched_pair.yml`, `config_small_step.yml`, …) come from the `gen_*.py` generators next to them; each frozen run records its config sha256 in its manifest, so the generators are tracked and the generated files are not

## Current experiment

`reports/EUCRD_MATCHED_PAIR_PREREG.md` (+ Addenda A–C) — the matched imperfect-forecast pair V_err vs E_err. Its protocol, thresholds, seeds and platform rule are frozen there; the evaluator and the reading come after.

## Formal results (evidence)

- **Readings**: `reports/*_READING_*.md` — one per frozen verdict, each naming its preregistration.
- **Preregistrations**: `reports/*_PREREG*.md` — frozen before the run they govern; amended only by append-only addenda.
- **Curated artefacts**: `reports/manifests/` — the run outputs each reading rests on, with the run manifests (config sha, seeds, budget).
- **Design log**: `reports/STAGE_D_PRIME_DESIGN.md` — numbered entries, newest last; §63–§70 cover the current EU-CRD chain.
- **Write-up obligations**: `reports/EUCRD_PAPER_DISCLOSURES.md` — what the paper must disclose, fixed in advance.

Raw run outputs under `g1/*/stage_a_out/` and `drl-manager/logs/` are **not tracked** (see `.gitignore`); they stay on disk and are archived separately with checksums.

## History

`docs/` (the V31/V32-era plans), older `reports/` entries, `isambard/` and `local_eval_rt/` runner collections, and the earlier campaigns' manifests. Useful for provenance, not for running anything today.

## For a paper artefact

This repository keeps the full history and is not the submission artefact. The release repository is built later from a named frozen commit and contains only the runnable code, the frozen configs, the protocols, the judges, the tests, the run instructions and an index of results, with the raw outputs published as a separate checksummed archive.
