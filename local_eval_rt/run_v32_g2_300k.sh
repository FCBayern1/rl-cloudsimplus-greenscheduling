#!/bin/bash
# V3.2 Gate 2 at 300k (PREREG A5: the 100k read measured initialization -
# ck0 and ck1 gave identical deltas). Queued behind the reference wave.
# Same +0.05 threshold; judged against this run's own ck0 as the zero point.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt; OUT=$R/v32_pipeline.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
while pgrep -f "run_wave_s2[.]sh|entrypoint_rlmodule|baselines[.]evaluate" >/dev/null 2>&1; do sleep 120; done
sleep 30
echo "===== V32 GATE2 @300k $(date '+%m-%d %H:%M') =====" >>"$OUT"
pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
  --experiment experiment_v3_2_oracle --total-timesteps 300000 --num-workers 6 --seed 1 \
  --output-dir logs/v32_g2_s1 > $R/v32_g2_s1_train.log 2>&1
echo "[g2-300k] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
for CK in $(ls -d logs/v32_g2_s1/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V); do
  ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
  echo "----- g2-300k probe $ckn -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
    --json-out $R/probe/v32_g2_${ckn}.json 2>>"$OUT" \
    | grep -E "job_temporal_delta|judgeable|by forecast_gain|by time_to" >>"$OUT"
done
echo "V32 GATE2-300K DONE $(date '+%m-%d %H:%M')" >>"$OUT"
