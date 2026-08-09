#!/bin/bash
# Protect the paper's 5% clean-forecast-value claim: the tab:main No-Forecast
# reference (0.196) is a SINGLE seed (cregime_noforecast, seed 42). Given the
# nofc basin-lottery evidence (rwtight 0.070-0.144, gamble s1/s2 split), train
# 2 more C-regime noforecast seeds so the reference is a 3-seed median.
# Serialized: waits for plan-B2 to finish (end of tonight's queue).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
GATE=$REPO/local_eval_rt/perseed_planb2.txt
OUT=$REPO/local_eval_rt/nofc_ref_seeds.txt
OUTDIR=$REPO/local_eval_rt
EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd_noforecast
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
echo "[nofcref] armed $(date '+%m-%d %H:%M'); waiting for PERSEED PLAN-B2 DONE" >>"$OUT"
while ! grep -qa "PERSEED PLAN-B2 DONE" "$GATE" 2>/dev/null; do sleep 300; done
cd $REPO/drl-manager
for SEED in 43 44; do
  OD=$REPO/drl-manager/logs/cregime_noforecast_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[nofcref s${SEED}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "$EXP" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$OUTDIR/nofcref_s${SEED}_train.log" 2>&1
  echo "[nofcref s${SEED}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    lg=$OUTDIR/nofcref_s${SEED}_${ckn}.log
    FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "$EXP" --global rllib \
      --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 10 --seed $SEED \
      --output "$OUTDIR/eval_csv/nofcref_s${SEED}_${ckn}.csv" >"$lg" 2>&1
    cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[nofcref s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf:-?}" >>"$OUT"
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  done
done
echo "NOFC REF SEEDS DONE $(date '+%m-%d %H:%M')" >>"$OUT"
