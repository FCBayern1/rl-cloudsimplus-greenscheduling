#!/bin/bash
# Multi-seed reproduction of the C-regime forecast ablation on Isambard (4xGH200/node, all parallel).
# Confirms the local seed-42 result (godeye 0.1519/95.3% | noforecast 0.1926/98.2% | timecap 0.1806/99.9%)
# is robust, not single-seed noise. Each arm x seed = one 1-GPU job (train + argmax eval).
#   export PROJECTDIR=/projects/u6kd/rl-cloudsimplus-greenscheduling   # repo parent? no: this IS it
#   SEEDS="1 2 3" bash "$PROJECTDIR/isambard/submit_cregime_seeds.sh"
set -euo pipefail
: "${PROJECTDIR:?}"
cd "$PROJECTDIR/isambard"
declare -A ARMS=(
  [godeye]=experiment_multi_5dc_carbon_v2_deferrable_gdpd
  [noforecast]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_noforecast
  [timecap]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
)
SEEDS="${SEEDS:-1 2 3}"
CFG="${CFG:-$PROJECTDIR/config_C.yml}"
echo "config: $CFG | seeds: $SEEDS"
for arm in godeye noforecast timecap; do
  for s in $SEEDS; do
    sbatch --job-name="c_${arm}_s${s}" \
      --export=ALL,PROJECTDIR="$PROJECTDIR",CFG="$CFG",EXP="${ARMS[$arm]}",OUT="cregime_${arm}",SEED="$s" \
      train_cregime.sbatch
  done
done
echo "submitted $(echo $SEEDS | wc -w) seeds x 3 arms. watch: squeue --me ; results: \$SCRATCHDIR/rl-runs/cregime_seeds_summary.txt"
