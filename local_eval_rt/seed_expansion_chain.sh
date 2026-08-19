#!/bin/bash
# Paper seed expansion (reviewer issue #1): main-arm pairs s5..s8 so the
# matched sign test can reach p < 0.01 (4 pairs cap out at p=0.0625).
# Sequential, --num-workers 6, identical recipe to the s1-s4 campaign.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/seed_expansion.txt
cd $REPO/drl-manager
# wait for the SQT2 capacity cells to clear the box
while tmux has-session -t cap_f035b 2>/dev/null || tmux has-session -t cap_f028 2>/dev/null; do sleep 120; done
echo "===== SEED EXPANSION START $(date '+%m-%d %H:%M') =====" >>"$OUT"
declare -A EXP
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
EXP[eucrd]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4
for S in 5 6 7 8; do
  for A in van eucrd; do
    D=creg_${A}_s${S}
    [ -d logs/$D ] && { echo "[seed] $D exists, skip" >>"$OUT"; continue; }
    pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
    echo "[seed] train $D start $(date '+%m-%d %H:%M')" >>"$OUT"
    .venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
      --experiment "${EXP[$A]}" --total-timesteps 600000 --num-workers 6 \
      --seed $S --output-dir logs/$D > $REPO/local_eval_rt/${D}_train.log 2>&1
    echo "[seed] train $D exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null
    pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  done
  echo "[seed] PAIR s${S} COMPLETE $(date '+%m-%d %H:%M')" >>"$OUT"
done
echo "SEED EXPANSION DONE $(date '+%m-%d %H:%M')" >>"$OUT"
