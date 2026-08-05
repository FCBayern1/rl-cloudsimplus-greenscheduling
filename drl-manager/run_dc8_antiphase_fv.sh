#!/bin/bash
# Anti-phase 8DC forecast-value experiment (the load-bearing scenario).
# Same Vanilla-PPO base as dc8_light (crd OFF); ONLY the 4 green-DC
# time_zone_offset_rows changed (Nordic 0 / Germany 0 / US_East 1000 /
# Nordic2 100) so the simulator green is anti-phase -> greedy "use current
# green" captures only ~0.23 of optimal (vs 0.82 on aligned dc8_light), i.e.
# a current-green policy is a poor predictor and the forecast has real carbon
# headroom. Two arms:
#   oracle     = godeye true-future forecast (forecast_mode=full) -> upper bound
#   noforecast = forecast features removed (forecast_mode=none)
# Both arms run on ONE machine (comparison never split). RUN THE SMOKE GATE
# FIRST: ./run_dc8_antiphase_smoke.sh must print "SMOKE OK".
# Usage: ./run_dc8_antiphase_fv.sh [seed]   (default seed 1)
set -uo pipefail
SEED=${1:-1}
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUTDIR=$REPO/local_eval_rt; SUMMARY=$OUTDIR/dc8ap_summary.txt
mkdir -p "$OUTDIR/eval_csv"
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
declare -A EXPS
EXPS[oracle]=experiment_dc8_antiphase_oracle
EXPS[noforecast]=experiment_dc8_antiphase_noforecast

echo "===== DC8-ANTIPHASE-FV seed $SEED start $(date '+%m-%d %H:%M') =====" >>"$SUMMARY"
for A in oracle noforecast; do
  EXP=${EXPS[$A]}; OD=$REPO/drl-manager/logs/dc8ap_${A}_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[dc8ap ${A}_s${SEED}] train start $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$OUTDIR/dc8ap_${A}_s${SEED}_train.log" 2>&1
  echo "[dc8ap ${A}_s${SEED}] train exit rc=$? $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "[dc8ap ${A}_s${SEED}] NO RUN" >>"$SUMMARY"; continue; }
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    lg=$OUTDIR/dc8ap_${A}_s${SEED}_${ckn}_clean.log
    BEFORE=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
    FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
      --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 1 --seed $SEED \
      --output "$OUTDIR/eval_csv/dc8ap_${A}_s${SEED}_${ckn}_clean.csv" >"$lg" 2>&1
    cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[dc8ap ${A}_s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf:-?}" >>"$SUMMARY"
    AFTER=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
    NEW=$(comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER"))
    [ -n "$NEW" ] && kill -9 $NEW 2>/dev/null
    sleep 3
  done
done
echo "DC8-ANTIPHASE-FV seed $SEED DONE $(date '+%m-%d %H:%M')" >>"$SUMMARY"
