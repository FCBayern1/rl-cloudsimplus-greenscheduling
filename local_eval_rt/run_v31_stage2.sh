#!/bin/bash
# Stage-2 chain (08-14): scenario gate + judgeable P1, strictly sequential in
# ONE script so there are no watcher races.
#   1. slack-aware oracle, theta 0.7 and 0.5  -> scenario upper bound
#   2. oracle s1 retrained to 300k            -> policy mature enough that the
#      control channel responds (the 100k probe found a near-uniform policy:
#      control TV 0.005 - the temporal sign is unjudgeable there)
#   3. probe EVERY checkpoint                 -> P1(s1) + sign-formation curve
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v31_stage2.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
cd $REPO/drl-manager
echo "[stage2] start $(date '+%m-%d %H:%M')" >>"$OUT"
for TH in 0.7 0.5; do
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  echo "----- slack-oracle theta=$TH $(date '+%H:%M') -----" >>"$OUT"
  timeout 2700 .venv/bin/python oracle_slack_planner.py --experiment experiment_v3_1_oracle \
    --seed 1 --theta $TH >> "$OUT" 2>&1 || echo "[slack th=$TH] rc=$?" >>"$OUT"
done
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
echo "----- s1 300k train $(date '+%H:%M') -----" >>"$OUT"
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
  --experiment experiment_v3_1_oracle --total-timesteps 300000 --num-workers 6 --seed 1 \
  --output-dir logs/v31_s1_300k > $R/v31_s1_300k_train.log 2>&1
echo "[stage2] train exit rc=$? $(date '+%H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
for CK in $(ls -d logs/v31_s1_300k/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V); do
  ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
  echo "----- probe s1-300k $ckn -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
    --json-out $R/probe/v31_s1_300k_${ckn}.json 2>>"$OUT" | sed -n '/channel/,$p' >>"$OUT"
done
echo "STAGE2 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
