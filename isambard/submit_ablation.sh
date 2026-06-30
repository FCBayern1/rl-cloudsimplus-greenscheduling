#!/bin/bash
# THE ISAMBARD PAYOFF: submit all 3 forecast arms x N seeds IN PARALLEL (1 GPU each).
# On your workstation these run sequentially (~3.5h each = ~30h+ for 9). Here they run at once.
#   export PROJECTDIR=/projects/.../rl-cloudsimplus-greenscheduling
#   SEEDS="42 1 2" bash "$PROJECTDIR/isambard/submit_ablation.sh"
set -euo pipefail
: "${PROJECTDIR:?}"
cd "$PROJECTDIR/isambard"
declare -A ARMS=(
  [godeye]=experiment_multi_5dc_carbon_v2_deferrable_gdpd
  [noforecast]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_noforecast
  [timecap]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
)
SEEDS="${SEEDS:-42 1 2}"
for arm in godeye noforecast timecap; do
  for s in $SEEDS; do
    sbatch --job-name="${arm}_s${s}" \
      --export=ALL,PROJECTDIR="$PROJECTDIR",EXP="${ARMS[$arm]}",OUT="iso_${arm}",SEED="$s" \
      train.sbatch
  done
done
echo "submitted. watch: squeue --me"
