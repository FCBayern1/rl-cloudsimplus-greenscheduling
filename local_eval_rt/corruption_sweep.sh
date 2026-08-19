#!/bin/bash
# Corruption-severity sweep (reviewer #4) + auditor quality axis (reviewer #3).
#
# Two graded axes whose eps=1 endpoints reproduce the PUBLISHED Table 1 cells
# byte-for-byte (test-locked): `blend` removes information, `pshuffle` tells a
# coherent lie on a growing fraction of sites. Every cell also runs the
# deployment auditor in LOG mode (behaviour-neutral), so each point carries an
# independently measured forecast-quality statistic (rolling Pearson chi
# between the short-horizon forecast and realised green) - the x-axis the
# reviewer asked for, produced by the paper's own auditor.
#
# Usage: corruption_sweep.sh <arm>     arm in {van, eucrd}
set -uo pipefail
ARM="${1:?arm required: van|eucrd}"
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/corruption_sweep_${ARM}.txt
EPISODES=5
SEED=3
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $REPO/drl-manager
mkdir -p $REPO/local_eval_rt/eval_csv $REPO/local_eval_rt/audit

declare -A CK EXP
CK[van]=${VAN_CK:-logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010}
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
CK[eucrd]=$REPO/isambard_backup/rl-runs-full/creg_eucrd_s2/multidc_gtrxl_training/PPO_multidc_env_dc62c_00000_0_2026-07-15_14-38-35/checkpoint_000010   # Isambard seed-2, training completion 1.0000 (the local s3 is the DIVERGED ckpt, App H)
EXP[eucrd]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4

echo "===== CORRUPTION SWEEP ${ARM} (seed ${SEED}, ep${EPISODES}, argmax) $(date '+%m-%d %H:%M') =====" >>"$OUT"

cell () {  # mode eps
  local PM="$1" EPS="$2"
  local tag="${ARM}_${PM}_${EPS}"
  local lg=$REPO/local_eval_rt/sweep_${tag}.log
  local chi=$REPO/local_eval_rt/audit/chi_${tag}.csv
  rm -f "$chi"
  FORECAST_PERTURB_MODE=$PM FORECAST_PERTURB_EPS=$EPS DECODE_TOPK=0 \
  TRUST_GATE_SOURCE=resid TRUST_GATE_MODE=log TRUST_GATE_LOG=$chi \
  nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "${EXP[$ARM]}" --global rllib --local rllib \
    --checkpoint "${CK[$ARM]}" --new-api --shared-local --global-defer \
    --episodes $EPISODES --seed $SEED \
    --output $REPO/local_eval_rt/eval_csv/sweep_${tag}.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local mchi=$(awk -F, 'NR>1 && $2 ~ /^-?[0-9]+(\.[0-9]+)?$/ {s+=$2; n++} END{if(n) printf "%.4f", s/n; else printf "nan"}' "$chi" 2>/dev/null)
  echo "[sweep ${ARM} ${PM}@${EPS}] carbon=${cc:-?} completion=${cf:-?} chi=${mchi:-nan}" >>"$OUT"
}

cell none 0.0
for E in 0.25 0.50 0.75 1.00; do cell blend $E; done
for E in 0.25 0.50 0.75 1.00; do cell pshuffle $E; done
echo "CORRUPTION SWEEP ${ARM} DONE $(date '+%m-%d %H:%M')" >>"$OUT"
