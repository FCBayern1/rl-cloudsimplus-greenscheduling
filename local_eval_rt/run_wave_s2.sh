#!/bin/bash
# Reference-wave s2 pair, restarted after the 18:22 pipeline death (cause not
# recoverable: no stderr was captured; this script logs stderr to a file so a
# repeat leaves evidence). Needed regardless of the V3.2 Gate-2 question.
# Then: wave probes + final-ck evals (4 cells).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v32_pipeline.txt
ERR=$R/wave_s2_stderr.log
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
echo "===== WAVE S2 RESTART $(date '+%m-%d %H:%M') =====" >>"$OUT"
train () {
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  echo "[wave] train $3 start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
    --experiment "$1" --total-timesteps 600000 --num-workers 6 --seed $2 \
    --output-dir logs/$3 > $R/$3_train.log 2>&1
  echo "[wave] train $3 exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
}
train experiment_v3_1_oracle     2 v31_oracle_s2
train experiment_v3_1_noforecast 2 v31_nofc_s2
echo "[wave] s2 pair trained $(date '+%m-%d %H:%M')" >>"$OUT"
declare -A EXPMAP
EXPMAP[v31_oracle_s1]=experiment_v3_1_oracle;   EXPMAP[v31_oracle_s2]=experiment_v3_1_oracle
EXPMAP[v31_nofc_s1]=experiment_v3_1_noforecast; EXPMAP[v31_nofc_s2]=experiment_v3_1_noforecast
for A in v31_oracle_s1 v31_nofc_s1 v31_oracle_s2 v31_nofc_s2; do
  S=${A##*_s}
  CK=$(ls -d logs/$A/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$CK" ] && { echo "[wave] $A missing" >>"$OUT"; continue; }
  ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
  lg=$R/waveeval_${A}.log
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  timeout 9000 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "${EXPMAP[$A]}" --global rllib \
    --local drain --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 10 --seed $S --output $R/eval_csv/waveeval_${A}.csv >"$lg" 2>&1
  cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  echo "[waveeval $A $ckn@DRAIN] cc=${cc:-?} completion=${cf:-?}" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
done
echo "WAVE S2 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
