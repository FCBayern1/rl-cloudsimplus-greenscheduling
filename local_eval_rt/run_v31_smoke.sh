#!/bin/bash
# V3.1 smoke: 100k oracle-arm training + P1 probe. WRITTEN TONIGHT, ARMED ONLY
# AFTER the morning switch decision (V31_PREREG.md §6) - do NOT nohup this
# until the cert configs are locked and preflight --v31-cert passes.
#
# Sequence (each step gates the next):
#   1. refuse to run while anything is on the machine;
#   2. preflight --v31-cert on the cert pair (hard gate);
#   3. gradlew installDist - the reward surgery lives in Java, and the
#      running-queue ban on refreshing jars only lifts when the queue is empty;
#   4. truth-table JUnit with the artifact's real mu/sigma (second hard gate);
#   5. 100k oracle-arm training (checkpoint EVERY iteration - the ck1-ck7
#      retention gap made the sign-acquisition timeline unrecoverable in v3);
#   6. P1 probe on the last checkpoint -> verdict line.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v31_smoke.txt
ORACLE=experiment_v3_1_oracle
BLIND=experiment_v3_1_noforecast
MU=3.524; SIGMA=2.512   # calib/experiment_v3_1_oracle_carbon_norm.json (mid_util)

if pgrep -f "baselines[.]evaluate|entrypoint_rlmodule|oracle_hold_until_green|run_v3_" >/dev/null 2>&1; then
  echo "[smoke] REFUSING: machine busy" | tee -a "$OUT"; exit 1
fi
cd $REPO/drl-manager
.venv/bin/python preflight_scenario.py $ORACLE $BLIND --v31-cert >>"$OUT" 2>&1 || { echo "[smoke] preflight FAILED" >>"$OUT"; exit 1; }
cd $REPO/cloudsimplus-gateway
./gradlew installDist -q >>"$OUT" 2>&1 || { echo "[smoke] installDist FAILED" >>"$OUT"; exit 1; }
./gradlew test --tests "exe.edu.cspg.multidc.PerActionRewardSurgeryTest" -q \
  -Dv31.mu=$MU -Dv31.sigma=$SIGMA >>"$OUT" 2>&1 || { echo "[smoke] truth-table FAILED with real mu/sigma" >>"$OUT"; exit 1; }
echo "[smoke] gates passed, training starts $(date '+%m-%d %H:%M')" >>"$OUT"
cd $REPO/drl-manager
.venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
  --experiment $ORACLE --total-timesteps 100000 --num-workers 6 --seed 1 \
  --output-dir $REPO/drl-manager/logs/v31_smoke_oracle_s1 > $R/v31_smoke_train.log 2>&1
echo "[smoke] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
CK=$(ls -d $REPO/drl-manager/logs/v31_smoke_oracle_s1/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
if [ -n "$CK" ]; then
  echo "----- P1 probe $(basename $CK) -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$CK" --trials 40 \
    --json-out $R/probe/v31_smoke.json 2>/dev/null | sed -n '/channel/,$p' >>"$OUT"
fi
echo "V31 SMOKE DONE $(date '+%m-%d %H:%M')" >>"$OUT"
