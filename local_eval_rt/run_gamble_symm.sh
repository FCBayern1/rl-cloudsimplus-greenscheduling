#!/bin/bash
# RESURRECTION EXPERIMENT (fix A, 2026-08-09): retrain the gamble noforecast arm
# with the reward leak closed (window_carbon_source: persistence) and compare
# against the EXISTING oracle arms (unchanged: their reward may legitimately see
# the future they also observe).
#   nofc_symm ~0.06+ (blind level)  => increment RESURRECTED: the tie was our
#        oracle reward teaching everyone; gamble becomes testbed v2 material.
#   nofc_symm still ~0.028          => reactive play needs no future info at
#        all; obs-level increment truly dead here -> structural coupling only.
# Two seeds so the basin lottery is visible (the s1/s2 lesson).
# Serialized: waits for the nofc reference-seed chain (end of current queue).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
GATE=$REPO/local_eval_rt/nofc_ref_seeds.txt
OUT=$REPO/local_eval_rt/gamble_symm.txt
OUTDIR=$REPO/local_eval_rt
EXP=experiment_v2026_gamble_noforecast_symm
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
echo "[symm] armed $(date '+%m-%d %H:%M'); waiting for NOFC REF SEEDS DONE" >>"$OUT"
while ! grep -qa "NOFC REF SEEDS DONE" "$GATE" 2>/dev/null; do sleep 300; done
cd $REPO/drl-manager
for SEED in 1 2; do
  OD=$REPO/drl-manager/logs/v2026gb_nofcsymm_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[symm s${SEED}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$OUTDIR/symm_s${SEED}_train.log" 2>&1
  echo "[symm s${SEED}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    lg=$OUTDIR/symm_s${SEED}_${ckn}.log
    FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
      --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 10 --seed $SEED \
      --output "$OUTDIR/eval_csv/symm_s${SEED}_${ckn}.csv" >"$lg" 2>&1
    cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    gr=$(grep -a "Avg Green" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[symm s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$OUT"
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  done
done
echo "GAMBLE SYMM DONE $(date '+%m-%d %H:%M') -- compare vs oracle_s1 0.0277-0.0370 / oracle_s2 0.0280-0.0319" >>"$OUT"
