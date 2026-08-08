#!/bin/bash
# TimeCAP (realistic-forecaster) arm on the v2026_gamble scenario: how much of
# the godeye 46% does a real forecaster capture? Single arm per invocation;
# compares against the existing noforecast/godeye arms of the SAME seed.
# Clean evals only here (corruption wave comes later with EU-CRD arms).
set -uo pipefail
SEED=${1:-1}
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUTDIR=$REPO/local_eval_rt; SUMMARY=$OUTDIR/v2026_gamble_summary.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
EXP=experiment_v2026_gamble_timecap
OD=$REPO/drl-manager/logs/v2026gb_timecap_s${SEED}
echo "===== V2026-GAMBLE TIMECAP seed $SEED start $(date '+%m-%d %H:%M') =====" >>"$SUMMARY"
pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
echo "[v2026gb timecap_s${SEED}] train start $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
  --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
  --output-dir "$OD" > "$OUTDIR/v2026gb_timecap_s${SEED}_train.log" 2>&1
echo "[v2026gb timecap_s${SEED}] train exit rc=$? $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
[ -z "$RUN" ] && { echo "[v2026gb timecap_s${SEED}] NO RUN" >>"$SUMMARY"; exit 1; }
for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
  ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
  lg=$OUTDIR/v2026gb_timecap_s${SEED}_${ckn}_clean.log
  FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
  .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
    --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 10 --seed $SEED \
    --output "$OUTDIR/eval_csv/v2026gb_timecap_s${SEED}_${ckn}_clean.csv" >"$lg" 2>&1
  cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  gr=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
  echo "[v2026gb timecap_s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$SUMMARY"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
done
echo "V2026-GAMBLE TIMECAP seed $SEED DONE $(date '+%m-%d %H:%M')" >>"$SUMMARY"
