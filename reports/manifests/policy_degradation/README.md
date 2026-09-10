# Policy degradation investigation artifacts

Report: `reports/RL_POLICY_DEGRADATION_INVESTIGATION_2026_09_10.md`.
Base repository HEAD: `e67a47137ba2c45a568e1ac752e07cca15e4715a`.
All work here is diagnostic. No training and no 2020 sealed windows were used.

## Main evidence

- `mechanics_v11_120k.json`: pre-repair memory defect, position counterexample,
  actual restored learner GAE connector settings.
- `production_load_v11*.json`: actual production loader, dropout mode, exact actor
  tensor comparison (value-head differences belong to the inference filter).
- `likelihood_*.json`: no-update rollout/loss reconstruction with independently
  controlled positional encoding and dropout.
- `bisect_v11_k0/`: 90 local within-run episodes, 15 checkpoints × 3 tiers × 2
  state modes; `bisect_summary.*` and `checkpoint_carbon_bisect.png` summarize.
- `legacy` in these diagnostic filenames means **stateless with dropout OFF**.
  It does not reproduce all historical production semantics.
- `prior_k0/`: functional zero-residual rule, not a saved matched initialization.
- `drift_local_v11.json`, `fixed_states_v11.json`, `drift_old_clean_v56.json`:
  same-state distribution probes, not on-policy carbon outcomes.
- `training_timeline.*`: available local training metrics and reward/carbon check.
- `prior_trajectory_credit.json`: fresh prior-rollout GAE with final learner critic;
  not historical training advantages or an on-policy performance guarantee.
- `primary_v{11,12,13}_final_six/`: corrected Vanilla final-checkpoint six-window
  shrink75 replays. E is not rejudged. `corrected_primary_vanilla.json` summarizes.
- `corrected_scheduler_repeat{1,2}/`: actual patched production scheduler agrees
  with independent stepwise control at every action; both repeated full trajectories
  and carbon are equal to the reference replay.
- `bisect_primary_v11_k0/AVAILABILITY.md`: primary intermediate checkpoints are
  absent locally. Loading failures are not experiment failures. The final three
  tier rows ran successfully; do not infer a primary intermediate curve.
- `provenance.json`: hashes and versions; created after all outputs/report updates.

## Reproduction (from `drl-manager/`)

```bash
.venv/bin/python ../g1/compressed_timecap_s2/diagnostics/bisect_replays.py \
  --root logs/matched_pair/V_s20260911/multidc_gtrxl_training/PPO_multidc_env_db752_00000_0_2026-09-09_13-06-15 \
  --out ../reports/manifests/policy_degradation/new_replication
```

The runner validates selected checkpoints, fixes two replay workers and single
BLAS threads, uses existing read-window configs, and never launches training.
Use a new output directory; existing result JSONs are not overwritten by the runner.
`--verify-scheduler --modes recurrent` additionally checks the production adapter.

## Diagnostic development notes

The first primary-intermediate inventory incorrectly inferred availability from
checkpoint paths; file loading exposed that only final primary checkpoints were
present. The launcher was hardened with a complete-file preflight, and failures
were retained. No simulator was started for those absent checkpoints.

Per-step reward/value recording was added during the local replay series without
changing the actor or environment return values. Early JSONs therefore lack full
trajectory records. Carbon/actions are available for all 90 rows. No missing
historical advantages or trajectories are fabricated.

Only deployment correctness was patched in production. Training position semantics
still have an explicitly expected-failing regression. A passing targeted suite
must not be described as proof of correct PPO training or carbon improvement.
