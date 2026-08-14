#!/bin/bash
# Stage-3: oracle s2 300k + per-ck probes -> completes the two-seed P1.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v31_stage2.txt   # same log, same monitor
export EVAL_CONFIG_PATH=$REPO/config_C.yml
cd $REPO/drl-manager
while pgrep -f "run_v31_stage2[.]sh|entrypoint_rlmodule" >/dev/null 2>&1; do sleep 60; done
echo "----- s2 300k train $(date '+%H:%M') -----" >>"$OUT"
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
  --experiment experiment_v3_1_oracle --total-timesteps 300000 --num-workers 6 --seed 2 \
  --output-dir logs/v31_s2_300k > $R/v31_s2_300k_train.log 2>&1
echo "[stage3] train exit rc=$? $(date '+%H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
for CK in $(ls -d logs/v31_s2_300k/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V); do
  ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
  echo "----- probe s2-300k $ckn -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
    --json-out $R/probe/v31_s2_300k_${ckn}.json 2>>"$OUT" | sed -n '/channel/,$p' >>"$OUT"
done
echo "STAGE3 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
