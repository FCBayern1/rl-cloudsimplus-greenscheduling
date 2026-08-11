#!/bin/bash
# GPU box: two MORE matched seed pairs for the paper's main arms (Table 1/3).
#
# Why: the main table currently rests on four training seeds of which only two
# to three yield a checkpoint meeting the completion contract, so the clean and
# blend medians run on n=2-3 and the matched-pair test has n=2. Two more pairs
# take the paired comparison to n=4-5, which is where a one-sided sign test can
# cross 0.05 (0.5^5 = 0.031).
#
# Pairing discipline: BOTH arms of a seed must be trained AND evaluated on this
# same machine, because the reported quantity is the within-pair difference and
# the cross-machine noise floor is ~16%. Never mix a Vanilla seed from one box
# with an EU-CRD seed from another.
#
# Each arm: train 600k steps, then evaluate the last three checkpoints under
# clean / blend / shuffle with deterministic decoding, 10 episodes each.
# Runtime estimate: 4 trainings x ~6h + 36 evals x ~0.5h, about 36 hours serial.
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
R=$REPO/local_eval_rt
OUT=$R/mainarm_seeds_gpu.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
mkdir -p $R/eval_csv

declare -A EXP
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecapV3
EXP[knSb]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b

echo "===== MAIN-ARM SEED BACKFILL (s5, s6) $(date '+%m-%d %H:%M') host=$(hostname) =====" >>"$OUT"
run_arm () {   # arm seed
  local A="$1" SEED="$2"
  local OD=$REPO/drl-manager/logs/v3ht_${A}_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[main ${A}_s${SEED}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "${EXP[$A]}" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$R/v3ht_${A}_s${SEED}_train.log" 2>&1
  echo "[main ${A}_s${SEED}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  local RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  # FINAL checkpoint only. The paired analysis in the paper deliberately uses the
  # last checkpoint of every run with no selection, because validation-based
  # selection drops pairs asymmetrically. Sweeping three checkpoints would only
  # serve the cross-method medians, and at 2.6 h per eval cell on this box it
  # would triple the campaign for a quantity the pairing does not use.
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -1); do
    local ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    for MODE in none blend shuffle; do
      local tag=$([ "$MODE" = none ] && echo clean || echo $MODE)
      local lg=$R/v3ht_${A}_s${SEED}_${ckn}_${tag}.log
      pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
      FORECAST_PERTURB_MODE=$MODE FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
      .venv/bin/python -m src.baselines.evaluate --experiment "${EXP[$A]}" --global rllib \
        --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
        --episodes 10 --seed $SEED \
        --output "$R/eval_csv/v3ht_${A}_s${SEED}_${ckn}_${tag}.csv" >"$lg" 2>&1
      local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
      local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
      # SAME line format as local_rt_summary.txt so the aggregation script reads it unchanged
      echo "[v3ht ${A}_s${SEED} ${ckn}@ARGMAX ${tag}] cc=${cc:-?} completion=${cf:-?}" >>"$OUT"
    done
  done
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
}
# pair 5 first, so one complete matched pair lands before pair 6 starts
run_arm van  5
run_arm knSb 5
echo "MAIN-ARM PAIR 5 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
run_arm van  6
run_arm knSb 6
echo "MAIN-ARM SEEDS DONE $(date '+%m-%d %H:%M')" >>"$OUT"
