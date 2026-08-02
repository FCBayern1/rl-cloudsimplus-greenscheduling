#!/bin/bash
# Component ablation for tab:ablation (base = knSV3b, single-key diffs):
#   ablG = gate off   (crd.blender.fixed_c=1.0  -> c_t≡1, pure counterfactual)
#   ablW = reweight off (crd.responsibility.reweight_advantages=false -> vanilla gradient + aux ensemble)
# Order: ablG_s1, ablW_s1, ablG_s2, ablW_s2. Eval: last 3 cks x {clean,blend,shuffle} @ argmax.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUTDIR=$REPO/local_eval_rt; SUMMARY=$OUTDIR/local_rt_summary.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
declare -A EXPS
EXPS[ablG]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_ablG
EXPS[ablW]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_ablW

echo "===== ABLATION start $(date '+%m-%d %H:%M') =====" >>"$SUMMARY"
for SEED in 1 2; do
for A in ablG ablW; do
  EXP=${EXPS[$A]}; OD=$REPO/drl-manager/logs/abl_${A}_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[abl ${A}_s${SEED}] train start $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$OUTDIR/abl_${A}_s${SEED}_train.log" 2>&1
  echo "[abl ${A}_s${SEED}] train exit rc=$? $(date '+%m-%d_%H:%M')" >>"$SUMMARY"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  [ -z "$RUN" ] && { echo "[abl ${A}_s${SEED}] NO RUN" >>"$SUMMARY"; continue; }
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    for PERT in none blend shuffle; do
      lab=$PERT; [ "$PERT" = none ] && lab=clean
      lg=$OUTDIR/abl_${A}_s${SEED}_${ckn}_${lab}.log
      BEFORE=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
      FORECAST_PERTURB_MODE=$PERT FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
      .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
        --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
        --episodes 1 --seed $SEED \
        --output "$OUTDIR/eval_csv/abl_${A}_s${SEED}_${ckn}_${lab}.csv" >"$lg" 2>&1
      cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
      cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
      echo "[abl ${A}_s${SEED} ${ckn}@ARGMAX ${lab}] cc=${cc:-?} completion=${cf:-?}" >>"$SUMMARY"
      AFTER=$(pgrep -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null | sort -u)
      NEW=$(comm -13 <(printf '%s\n' "$BEFORE") <(printf '%s\n' "$AFTER"))
      [ -n "$NEW" ] && kill -9 $NEW 2>/dev/null
      sleep 3
    done
  done
done
done
echo "ABLATION DONE $(date '+%m-%d %H:%M')" >>"$SUMMARY"
