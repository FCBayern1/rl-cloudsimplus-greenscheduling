#!/bin/bash
# De-confound the v3 verdict: same checkpoints, same offsets, but BOTH arms
# evaluated with `--local drain` instead of each arm's co-learned local policy.
#
# Why: the §2 verdict cells used --local rllib, so每个 global 搭配的是与它共同
# 演化出来的 local(release-rate 各臂不同, 见诊断文档 §2b 第三轮复审)。
# 若 drain 下预报臂仍不赢碳, local co-learning 混杂即被排除, §2 判决站稳;
# 若差距明显变化, 说明原判决部分来自 local 差异。
#
# First cell is a 1-episode WIRING SMOKE for the new local_override path in
# evaluate.py -- aborts loudly unless the log shows "Local override ACTIVE".
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v3_drainfix.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
echo "[drainfix] armed $(date '+%m-%d %H:%M'); queued behind sp/P3/track0" >>"$OUT"
while pgrep -f "run_v3_sp[.]sh|run_v3_sp_eval[.]sh|entrypoint_rlmodule_gtrxl|baselines[.]evaluate|run_v3_track0[.]sh|oracle_hold_until_green" >/dev/null 2>&1; do
  sleep 180
done
sleep 60
cd $REPO/drl-manager
L=$REPO/drl-manager/logs
ck () { ls -d $L/$1/multidc_gtrxl_training/PPO_*/$2 2>/dev/null | head -1; }

# --- wiring smoke: 1 episode, must print the override marker -----------------
SM=$R/drainfix_smoke.log
timeout 1500 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
  .venv/bin/python -m src.baselines.evaluate --experiment experiment_v3_oracle \
  --global rllib --local drain --checkpoint "$(ck v3_oracle_s1 checkpoint_000009)" \
  --new-api --shared-local --global-defer --episodes 1 --seed 1 \
  --output $R/eval_csv/drainfix_smoke.csv >"$SM" 2>&1
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
if ! grep -q "Local override ACTIVE" "$SM"; then
  echo "[drainfix] SMOKE FAILED -- override marker missing, wiring bad, ABORT" >>"$OUT"
  exit 1
fi
echo "[drainfix] wiring smoke OK $(date '+%H:%M')" >>"$OUT"

# --- the six de-confound cells ------------------------------------------------
cell () {  # exp run_dir ckdir tag seed
  local lg=$R/drainfix_${4}.log
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  timeout 9000 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "$1" \
    --global rllib --local drain --checkpoint "$(ck $2 $3)" \
    --new-api --shared-local --global-defer --episodes 10 --seed $5 \
    --output $R/eval_csv/drainfix_${4}.csv >"$lg" 2>&1
  if [ $? -eq 124 ]; then
    echo "[drainfix $4] TIMEOUT -- skipped" >>"$OUT"
  else
    local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    local gr=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[drainfix $4@DRAIN] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$OUT"
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
}
cell experiment_v3_oracle     v3_oracle_s1 checkpoint_000009 oracle_s1_ck9  1  # iso 合格格
cell experiment_v3_noforecast v3_nofc_s1   checkpoint_000009 nofc_s1_ck9    1  # 盲臂最好格
cell experiment_v3_oracle     v3_oracle_s1 checkpoint_000010 oracle_s1_ck10 1
cell experiment_v3_oracle     v3_oracle_s2 checkpoint_000010 oracle_s2_ck10 2
cell experiment_v3_noforecast v3_nofc_s1   checkpoint_000010 nofc_s1_ck10   1
cell experiment_v3_noforecast v3_nofc_s2   checkpoint_000012 nofc_s2_ck12   2
echo "DRAINFIX DONE $(date '+%m-%d %H:%M')" >>"$OUT"
