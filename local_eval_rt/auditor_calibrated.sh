#!/bin/bash
# Auditor coverage fix (2026-08-19): the class-default absolute line chi<0.2
# is calibrated for sign inversion and misses the paper's headline Shuffle
# corruption (chi 0.7085 clean -> 0.2304 shuffled: a 67% drop that never
# crosses 0.2). This validates the calibrated rule "fire when chi < rel *
# chi_clean" with chi_clean measured on the clean run of THIS checkpoint.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/auditor_calibrated.txt
EPISODES=3
SEED=3
CLEAN_CHI=0.7085          # measured, audg clean cell, same ckpt/seed
REL=0.5                   # -> effective threshold 0.354
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $REPO/drl-manager
CK=logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010
EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
echo "===== AUDITOR CALIBRATED (chi_clean=$CLEAN_CHI rel=$REL -> thresh=0.354) $(date '+%m-%d %H:%M') =====" >>"$OUT"
cell () {  # mode eps
  local PM="$1" EPS="$2"
  local tag="cal_${PM}${EPS}"
  local lg=$REPO/local_eval_rt/audc_${tag}.log
  local chi=$REPO/local_eval_rt/audit/chi_c_${tag}.csv
  rm -f "$chi"
  FORECAST_PERTURB_MODE=$PM FORECAST_PERTURB_EPS=$EPS DECODE_TOPK=0 \
  TRUST_GATE_SOURCE=resid TRUST_GATE_MODE=gate TRUST_GATE_LOG=$chi \
  TRUST_GATE_CLEAN_CHI=$CLEAN_CHI TRUST_GATE_REL=$REL \
  nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "$EXP" --global rllib --local rllib --checkpoint "$CK" \
    --new-api --shared-local --global-defer --episodes $EPISODES --seed $SEED \
    --output $REPO/local_eval_rt/eval_csv/audc_${tag}.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local sm=$(grep -a "ForecastResidualMonitor" "$lg"|tail -1|grep -oE "gated=[0-9]+ \([0-9.]+%\)")
  local mchi=$(awk -F, 'NR>1 && $2 ~ /^-?[0-9]+(\.[0-9]+)?$/ {s+=$2; n++} END{if(n) printf "%.4f", s/n; else printf "nan"}' "$chi" 2>/dev/null)
  echo "[audc ${PM}@${EPS}] carbon=${cc:-?} completion=${cf:-?} chi=${mchi:-nan} ${sm}" >>"$OUT"
}
cell pshuffle 1.00      # the blind spot: must now fire
cell none 0.0           # false-positive check: must stay quiet
echo "AUDITOR CALIBRATED DONE $(date '+%m-%d %H:%M')" >>"$OUT"
