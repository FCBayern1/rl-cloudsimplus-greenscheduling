#!/bin/bash
# Auditor expansion (reviewer #3): App H currently reports ONE policy under ONE
# inverted forecast, described as a "functionality check". This runs the auditor
# across the corruption family at matched settings, so the deployment claim rests
# on a grid rather than a single cell: {off, gate, repair} x {clean, blend,
# pshuffle, panti} on the healthy vanilla seed-3 checkpoint (the arm Table 5
# already anchors on; the local EU-CRD ckpt is the diverged one and is excluded).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/auditor_grid.txt
EPISODES=3
SEED=3
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $REPO/drl-manager
mkdir -p $REPO/local_eval_rt/eval_csv $REPO/local_eval_rt/audit
CK=logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010
EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
echo "===== AUDITOR GRID van (seed $SEED, ep$EPISODES, argmax, resid chi) $(date '+%m-%d %H:%M') =====" >>"$OUT"

cell () {  # mode eps auditor
  local PM="$1" EPS="$2" AM="$3"
  local tag="van_${PM}${EPS}_${AM}"
  local lg=$REPO/local_eval_rt/audg_${tag}.log
  local chi=$REPO/local_eval_rt/audit/chi_g_${tag}.csv
  rm -f "$chi"
  local envs=(FORECAST_PERTURB_MODE=$PM FORECAST_PERTURB_EPS=$EPS DECODE_TOPK=0
              TRUST_GATE_SOURCE=resid TRUST_GATE_LOG=$chi)
  case "$AM" in
    off)    envs+=(TRUST_GATE_MODE=log) ;;                       # measure only
    gate)   envs+=(TRUST_GATE_MODE=gate TRUST_GATE_THRESH=0.2) ;;
    repair) envs+=(TRUST_GATE_MODE=repair) ;;
  esac
  env "${envs[@]}" nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "$EXP" --global rllib --local rllib --checkpoint "$CK" \
    --new-api --shared-local --global-defer --episodes $EPISODES --seed $SEED \
    --output $REPO/local_eval_rt/eval_csv/audg_${tag}.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local sm=$(grep -a "ForecastResidualMonitor" "$lg"|tail -1|grep -oE "gated=[0-9]+ \([0-9.]+%\)|repaired=[0-9]+"|tr '\n' ' ')
  local mchi=$(awk -F, 'NR>1 && $2 ~ /^-?[0-9]+(\.[0-9]+)?$/ {s+=$2; n++} END{if(n) printf "%.4f", s/n; else printf "nan"}' "$chi" 2>/dev/null)
  echo "[audg ${PM}@${EPS} ${AM}] carbon=${cc:-?} completion=${cf:-?} chi=${mchi:-nan} ${sm}" >>"$OUT"
}

for AM in off gate repair; do
  cell panti 1.00 $AM        # the published inverted-forecast cell
  cell pshuffle 1.00 $AM     # coherent site lie (Table 1's Shuffle)
  cell blend 1.00 $AM        # information removal (Table 1's Blend)
done
cell none 0.0 off            # clean anchor
echo "AUDITOR GRID DONE $(date '+%m-%d %H:%M')" >>"$OUT"
