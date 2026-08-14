#!/bin/bash
# V3.1 full wave — lit per the pre-registered decision tree (V31_PREREG.md):
# P1 PASS both seeds (s1: +.0065..+.0088 over 4 cks; s2: +.0018..+.0034 over 3)
# x scenario gate PASS (slack oracle -21/-29% @ 100%).
# Order pairs within seed so the first paired verdict lands earliest:
#   oracle_s1 -> nofc_s1 -> oracle_s2 -> nofc_s2, then argmax evals
#   (10 episodes, --local drain per protocol, last-3 checkpoints).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v31_fullwave.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
declare -A EXP
EXP[oracle]=experiment_v3_1_oracle
EXP[nofc]=experiment_v3_1_noforecast
echo "===== V31 FULL WAVE $(date '+%m-%d %H:%M') =====" >>"$OUT"
while pgrep -f "run_v31_stage3[.]sh|entrypoint_rlmodule" >/dev/null 2>&1; do sleep 60; done

train_arm () {  # arm seed
  local A="$1" S="$2"
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  echo "[wave ${A}_s${S}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
    --experiment "${EXP[$A]}" --total-timesteps 600000 --num-workers 6 --seed $S \
    --output-dir logs/v31_${A}_s${S} > $R/v31_${A}_s${S}_train.log 2>&1
  echo "[wave ${A}_s${S}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
}
train_arm oracle 1
train_arm nofc   1
echo "[wave] PAIR s1 trained $(date '+%m-%d %H:%M')" >>"$OUT"
train_arm oracle 2
train_arm nofc   2
echo "[wave] all trained $(date '+%m-%d %H:%M')" >>"$OUT"

# sign curve on the 600k oracle arms (feeds P1 stability + section 7.3)
for A in oracle_s1 oracle_s2; do
  for CK in $(ls -d logs/v31_${A}/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V); do
    ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
    echo "----- probe600 ${A} ${ckn} -----" >>"$OUT"
    .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
      --json-out $R/probe/v31w_${A}_${ckn}.json 2>>"$OUT" | grep -E "difference|fraction" >>"$OUT"
  done
done

eval_ck () {  # arm seed ckpath
  local A="$1" S="$2" CK="$3"
  local ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
  local lg=$R/v31w_${A}_s${S}_${ckn}.log
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  timeout 9000 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "${EXP[$A]}" --global rllib \
    --local drain --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 10 --seed $S --output $R/eval_csv/v31w_${A}_s${S}_${ckn}.csv >"$lg" 2>&1
  if [ $? -eq 124 ]; then
    echo "[v31w ${A}_s${S} ${ckn}] TIMEOUT -- skipped" >>"$OUT"
  else
    local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    local gr=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[v31w ${A}_s${S} ${ckn}@DRAIN] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$OUT"
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
}
for A in oracle nofc; do
  for S in 1 2; do
    RUN=$(ls -d logs/v31_${A}_s${S}/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
    for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
      eval_ck $A $S "$CK"
    done
  done
done
echo "V31 FULL WAVE DONE $(date '+%m-%d %H:%M')" >>"$OUT"
