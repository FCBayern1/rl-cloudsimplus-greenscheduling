#!/bin/bash
# P3 for the unified-pricing arm: does experiment_v3_oracle_sp actually differ
# from the blind arm on carbon and completion, or has it merely converged to the
# blind policy?
#
# The P1 probe on seed 1 says the wrong-signed temporal response is gone, but it
# also says forecast sensitivity fell from 0.53 to 0.078 and the defer rate rose
# to 2.93% -- both within reach of the blind arm's own numbers (0.016 and 2.8%).
# So the open question is no longer "did the bug go away" but "did anything
# replace it". Only an evaluation answers that.
#
# Same protocol as the argmax verdict in run_v3_evals_final.sh (10 episodes,
# DECODE_TOPK=0, last three checkpoints, timeout-only screening) so the numbers
# drop straight into the same table.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v3_sp.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
echo "[v3-sp-eval] armed $(date '+%m-%d %H:%M'); waiting for training to finish" >>"$OUT"
while pgrep -f "entrypoint_rlmodule_gtrxl|run_v3_sp.sh" >/dev/null 2>&1; do sleep 180; done
sleep 60
cd $REPO/drl-manager
for SEED in 1 2; do
  RUN=$(ls -d $REPO/drl-manager/logs/v3_oraclesp_s${SEED}/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    lg=$R/final_v3_oraclesp_s${SEED}_${ckn}.log
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
    echo "[final oraclesp_s${SEED} ${ckn}] start $(date '+%H:%M')" >>"$OUT"
    timeout 9000 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
      .venv/bin/python -m src.baselines.evaluate --experiment experiment_v3_oracle_sp \
      --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 10 --seed $SEED --output $R/eval_csv/final_v3_oraclesp_s${SEED}_${ckn}.csv \
      >"$lg" 2>&1
    if [ $? -eq 124 ]; then
      echo "[final oraclesp_s${SEED} ${ckn}] TIMEOUT -- degenerate, skipped" >>"$OUT"
    else
      cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
      cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
      gr=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
      echo "[final oraclesp_s${SEED} ${ckn}] cc=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$OUT"
    fi
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  done
done
echo "V3-SP EVAL DONE $(date '+%m-%d %H:%M')" >>"$OUT"
