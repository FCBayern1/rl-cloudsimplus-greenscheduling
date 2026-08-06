#!/bin/bash
# THE decisive RL test: does RL (argmax deployment decode) capture the ~40%
# analytic oracle carbon headroom on the v2026 long-job workload? This is the
# only unverified gate after the workload-root-cause finding.
# Scenario: v2026 LP durations (median 55 steps, tail to 1000) + spread3k green
# phases + N=2000 (smoke-calibrated: green_ratio 47.7%, brown burns, ~97% reactive
# completion). Two Vanilla-PPO arms (crd OFF):
#   oracle     = godeye true-future (forecast_mode=full) -> upper bound
#   noforecast = forecast removed (forecast_mode=none)
# Verdict: godeye clearly below noforecast at iso-completion -> RL realises the
# room -> this is the load-bearing testbed. godeye ~= noforecast -> argmax masks it.
set -uo pipefail
SEED=${1:-1}
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUTDIR=$REPO/local_eval_rt; SUMMARY=$OUTDIR/v2026_spread_summary.txt
mkdir -p "$OUTDIR/eval_csv"
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
declare -A EXPS
EXPS[oracle]=experiment_v2026_spread_oracle
EXPS[noforecast]=experiment_v2026_spread_noforecast

echo "===== V2026-SPREAD-FV seed $SEED start $(date '+%m-%d %H:%M') =====" >>"$SUMMARY"
for A in oracle noforecast; do
  EXP=${EXPS[$A]}; OD=$REPO/drl-manager/logs/v2026sp_${A}_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[v2026sp ${A}_s${SEED}] train start $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$OUTDIR/v2026sp_${A}_s${SEED}_train.log" 2>&1
  echo "[v2026sp ${A}_s${SEED}] train exit rc=$? $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "[v2026sp ${A}_s${SEED}] NO RUN" >>"$SUMMARY"; continue; }
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    lg=$OUTDIR/v2026sp_${A}_s${SEED}_${ckn}_clean.log
    BEFORE=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
    FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
      --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 10 --seed $SEED \
      --output "$OUTDIR/eval_csv/v2026sp_${A}_s${SEED}_${ckn}_clean.csv" >"$lg" 2>&1
    cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    gr=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[v2026sp ${A}_s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$SUMMARY"
    AFTER=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
    NEW=$(comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER"))
    [ -n "$NEW" ] && kill -9 $NEW 2>/dev/null
    sleep 3
  done
done
echo "V2026-SPREAD-FV seed $SEED DONE $(date '+%m-%d %H:%M')" >>"$SUMMARY"
