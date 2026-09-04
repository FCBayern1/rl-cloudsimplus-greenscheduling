#!/bin/bash
# Stage D-B / D-C submitter: a side preregistration with its own config, line set and
# output roots, so the frozen four-line run is untouched.
#   BASELINE=cca   ten jobs, lines NC C   (STAGE_D_CCA_PREREG)
#   BASELINE=risk  twenty jobs, lines RCV RRS RMV RDC   (STAGE_D_RISK_PREREG)
set -euo pipefail
cd "$(dirname "$0")/.."
mkdir -p logs_slurm
SEEDS=${SEEDS:-"20260904 20260905 20260906 20260907 20260908"}
BASELINE=${BASELINE:-cca}
if [ "$BASELINE" = risk ]; then LINES="RCV RRS RMV RDC"; else LINES="NC C"; fi
LINESCSV=$(echo $LINES | tr " " ",")
CFG=$PWD/g1/compressed_timecap_s2/config_stage_d_$BASELINE.yml
[ -f "$CFG" ] || { echo "missing $CFG"; exit 1; }
for S in $SEEDS; do
  IDS=""
  for L in $LINES; do
    J=$(sbatch --parsable --job-name=sd$BASELINE \
        --export=ALL,SEED=$S,LINE=$L,STAGE_D_CONFIG=$CFG,STAGE_D_SUFFIX=_$BASELINE,STAGE_D_LINES=$LINESCSV \
        isambard/stage_d_train_line.sbatch)
    IDS="${IDS:+$IDS:}$J"
    echo "$BASELINE seed $S line $L -> job $J"
  done
  E=$(sbatch --parsable --job-name=sd${BASELINE}ev --dependency=afterok:$IDS \
      --export=ALL,SEED=$S,STAGE_D_CONFIG=$CFG,STAGE_D_SUFFIX=_$BASELINE,STAGE_D_LINES=$LINESCSV \
      isambard/stage_d_eval_seed.sbatch)
  echo "$BASELINE seed $S eval -> job $E (after $IDS)"
done
echo "submitted"
