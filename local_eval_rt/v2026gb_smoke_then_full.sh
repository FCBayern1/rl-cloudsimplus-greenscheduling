#!/bin/bash
# Gate: wait for the 15k training smoke (experiment_v2026_gamble_oracle) to exit.
# If it finished clean (no learner crash / NaN), launch the full godeye-vs-
# noforecast RL comparison. If it crashed, do NOT launch -- log and stop.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
SMOKELOG=/tmp/v2026gb_smoke_train.log
LOG=$REPO/local_eval_rt/v2026gb_gate.log
echo "[gate] armed $(date '+%m-%d %H:%M'); waiting for smoke to exit" >>"$LOG"
while pgrep -f "experiment_v2026_gamble_oracle.*15000" >/dev/null 2>&1; do sleep 30; done
sleep 3
if grep -aiqE "same shape|Traceback|InvalidArgument|RuntimeError|nan_|isnan" "$SMOKELOG" 2>/dev/null; then
  echo "[gate] SMOKE CRASHED -- NOT launching full run. tail:" >>"$LOG"
  tail -20 "$SMOKELOG" >>"$LOG"
  exit 1
fi
CK=$(ls -d $REPO/drl-manager/logs/v2026gb_smoke/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | tail -1)
echo "[gate] smoke clean (checkpoint=${CK:-none}) -> launching full RL run $(date '+%m-%d %H:%M')" >>"$LOG"
pkill -9 -f "ray::[A-Za-z]" 2>/dev/null || true
sleep 5
cd $REPO/drl-manager
nohup bash $REPO/local_eval_rt/run_v2026_gamble_fv.sh 1 >> "$LOG" 2>&1 &
echo "[gate] full run launched pid=$! $(date '+%m-%d %H:%M')" >>"$LOG"
