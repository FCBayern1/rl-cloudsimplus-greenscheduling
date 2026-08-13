#!/bin/bash
# Track 0: the in-simulator heuristic upper bound for v3 (strategy §6e, step 2
# of §6d). One decisive number before any more RL surgery: how much carbon does
# a PERFECT user of the forecast (hold-until-green, godeye) save over a spot-
# price greedy (no-hold) on the v3 testbed, at matched completion?
#
#   gap >= 15%  -> the scenario's carbon lever is real; proceed to V3.1 (track 1)
#   gap <  10%  -> the lever itself is too small; fix scenario params, DON'T
#                  touch PPO. (The script also prints per-DC green/demand
#                  calibration -- ">>1x means structural oversupply, no lever".)
#
# Sweeps hold-thresh because the upper bound is the max over thresholds, not
# the default. Queued behind sp s2 training AND the P3 eval chain -- one job at
# a time on this box (an eval spawns 8 JVMs; overlap slows everything ~4x).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v3_track0.txt
echo "[track0] armed $(date '+%m-%d %H:%M'); waiting for sp train + P3 eval" >>"$OUT"
while pgrep -f "run_v3_sp[.]sh|run_v3_sp_eval[.]sh|entrypoint_rlmodule_gtrxl|baselines[.]evaluate" >/dev/null 2>&1; do
  sleep 180
done
sleep 60
cd $REPO/drl-manager
for SEED in 1 2; do
  for TH in 0.3 0.6; do
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
    echo "----- track0 seed=$SEED thresh=$TH $(date '+%H:%M') -----" >>"$OUT"
    timeout 7200 .venv/bin/python oracle_hold_until_green.py \
      --experiment experiment_v3_oracle --config $REPO/config_C.yml \
      --seed $SEED --hold-thresh $TH \
      > $R/track0_s${SEED}_th${TH}.log 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[track0 s${SEED} th${TH}] rc=$rc (124=timeout) -- see log" >>"$OUT"
    else
      sed -n '/=== RESULT ===/,/no lever helps/p' $R/track0_s${SEED}_th${TH}.log >>"$OUT"
    fi
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  done
done
echo "TRACK0 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
