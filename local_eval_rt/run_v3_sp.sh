#!/bin/bash
# v3 oracle re-trained with the blind arm's pricing (window_carbon_source:
# persistence), making forecast_mode the only difference between the two arms.
#
# The hypothesis under test is in docs/V3_FORECAST_DIAGNOSIS.md section 5: the
# actual-window pricing credits "route now" with green that has not arrived, so
# the oracle never learns to wait. Its measured defer rate sits 4-10x below the
# blind arm's and its response to "green is arriving" carries the wrong sign.
#
# Pre-registered criteria, written before the run (section 6 of the doc):
#   P1  P(defer | green arriving) - P(defer | green leaving) > 0, both seeds
#   P2  oracle defer rate >= blind defer rate
#   P3  at >=97% completion, carbon/MI beats the blind arm's best cell by >13%
# The probe below adjudicates P1 as soon as training ends; P3 needs an eval.
#
# The blind arm is NOT retrained: it already uses persistence, so its existing
# checkpoints in logs/v3_nofc_s* are the matched control.
#
# Gated on process state rather than a log marker -- a stale "DONE" string in an
# append-only log once let a queued stage jump ahead and compete with training.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v3_sp.txt
echo "[v3-sp] armed $(date '+%m-%d %H:%M'); waiting for the argmax pass to exit" >>"$OUT"
while pgrep -f "run_v3_evals_final.sh|baselines.evaluate" >/dev/null 2>&1; do sleep 120; done
sleep 30
cd $REPO/drl-manager
for SEED in 1 2; do
  OD=$REPO/drl-manager/logs/v3_oraclesp_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  echo "[v3-sp s${SEED}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment experiment_v3_oracle_sp --total-timesteps 600000 --num-workers 6 \
    --seed $SEED --output-dir "$OD" > "$R/v3_oraclesp_s${SEED}_train.log" 2>&1
  echo "[v3-sp s${SEED}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  CK=$(ls -d $OD/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  if [ -n "$CK" ]; then
    echo "----- P1 probe, s${SEED}, $(basename "$CK") -----" >>"$OUT"
    .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$CK" --trials 40 \
      --json-out $R/probe/oraclesp_s${SEED}.json 2>/dev/null \
      | sed -n '/channel/,$p' >>"$OUT"
  else
    echo "[v3-sp s${SEED}] no checkpoint produced" >>"$OUT"
  fi
done
echo "V3-SP DONE $(date '+%m-%d %H:%M')" >>"$OUT"
