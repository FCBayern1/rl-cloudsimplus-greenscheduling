#!/bin/bash
# Submit the whole Stage D long run on Isambard: twenty single-GPU training jobs (five
# seeds x four lines) plus one dependent evaluation job per seed. Idempotent per seed only
# in the sense that re-running resubmits; check the queue first.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs_slurm
SEEDS=${SEEDS:-"20260904 20260905 20260906 20260907 20260908"}
LINES="NV V NE E"
for S in $SEEDS; do
  IDS=""
  for L in $LINES; do
    J=$(sbatch --parsable --export=ALL,SEED=$S,LINE=$L isambard/stage_d_train_line.sbatch)
    IDS="${IDS:+$IDS:}$J"
    echo "seed $S line $L -> job $J"
  done
  E=$(sbatch --parsable --dependency=afterok:$IDS --export=ALL,SEED=$S isambard/stage_d_eval_seed.sbatch)
  echo "seed $S eval -> job $E (after $IDS)"
done
echo "submitted; squeue -u $USER"
