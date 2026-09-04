#!/bin/bash
# Stage D-B, the CCA-PG credit-assignment baseline (STAGE_D_CCA_PREREG): ten single-GPU
# training jobs (five seeds x two lines) plus one dependent evaluation job per seed.
# Separate config, separate output roots, so the frozen four-line run is untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs_slurm
SEEDS=${SEEDS:-"20260904 20260905 20260906 20260907 20260908"}
LINES="NC C"
CFG=$PWD/g1/compressed_timecap_s2/config_stage_d_cca.yml
for S in $SEEDS; do
  IDS=""
  for L in $LINES; do
    J=$(sbatch --parsable --job-name=sdcca \
        --export=ALL,SEED=$S,LINE=$L,STAGE_D_CONFIG=$CFG,STAGE_D_SUFFIX=_cca,STAGE_D_LINES=NC,C \
        isambard/stage_d_train_line.sbatch)
    IDS="${IDS:+$IDS:}$J"
    echo "cca seed $S line $L -> job $J"
  done
  E=$(sbatch --parsable --job-name=sdccaev --dependency=afterok:$IDS \
      --export=ALL,SEED=$S,STAGE_D_CONFIG=$CFG,STAGE_D_SUFFIX=_cca,STAGE_D_LINES=NC,C \
      isambard/stage_d_eval_seed.sbatch)
  echo "cca seed $S eval -> job $E (after $IDS)"
done
echo "submitted"
