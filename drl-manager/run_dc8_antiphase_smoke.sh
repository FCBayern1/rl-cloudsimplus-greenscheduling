#!/bin/bash
# GATE before the real anti-phase RL run: does RLlib training run to a checkpoint
# on THIS machine without the learner shape-crash? (GPU box had an unverified
# torch/numpy fix.) 20k steps, 2 workers, oracle arm only -> a few minutes.
# PASS = "SMOKE OK" printed AND a checkpoint_* directory exists.
# Also records the first-step per-DC green so we can eyeball anti-phase.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
cd $REPO/drl-manager
OD=$REPO/drl-manager/logs/dc8ap_smoke
LOG=$OD/train.log
mkdir -p $OD
pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
echo "[smoke] antiphase oracle 20k start $(date '+%m-%d %H:%M')"
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
  --experiment experiment_dc8_antiphase_oracle --total-timesteps 20000 \
  --num-workers 2 --seed 1 --output-dir "$OD" > "$LOG" 2>&1
rc=$?
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null
CK=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
echo "[smoke] train rc=$rc  checkpoint=${CK:-NONE}"
if grep -qiE "same shape|InvalidArgument|Traceback" "$LOG" && [ -z "$CK" ]; then
  echo "SMOKE FAIL — learner error, see $LOG (tail below):"; tail -30 "$LOG"; exit 1
fi
[ -n "$CK" ] && echo "SMOKE OK — learner ran, checkpoint written. Safe to run the full anti-phase comparison." || {
  echo "SMOKE FAIL — no checkpoint. tail:"; tail -30 "$LOG"; exit 1; }
