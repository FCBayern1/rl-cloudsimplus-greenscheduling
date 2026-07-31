#!/bin/bash
# Monitor: wait for the local EU-CRD s3 pilot (PID passed as $1) to finish, then
# train + evaluate the vanilla (plain-timecap) C-regime arm at seed 3 on the local
# 5080, so every C-regime method has 3 seeds. Mirrors isambard/train_risk.sbatch
# (train clean forecast -> eval clean + anti x6 + shuffle x3 + blend x3) but local:
# config already carries py4j_port=0 (free port per worker).
set -uo pipefail

REPO="/home/joshua/rl-cloudsimplus-greenscheduling"
PY="$REPO/drl-manager/.venv/bin/python"
CFG="$REPO/config_C.yml"
EXP="experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap"
SEED=3
WORKERS=6
STEPS=600000
RUNDIR="$REPO/drl-manager/logs/creg_van_local_s${SEED}"
SUMMARY="$REPO/drl-manager/logs/cregime_local_summary.txt"
WATCH_PID="${1:?usage: local_vanilla_s3_monitor.sh <eucrd_pid>}"
cd "$REPO/drl-manager"

echo "[mon $(date +%H:%M:%S)] waiting for EU-CRD s3 PID=$WATCH_PID to finish..." | tee -a "$SUMMARY"
# ---- 1) wait for the pilot to exit (poll every 60s) ----
while kill -0 "$WATCH_PID" 2>/dev/null; do sleep 60; done
echo "[mon $(date +%H:%M:%S)] pilot PID=$WATCH_PID gone. GPU free -> launching vanilla s${SEED}." | tee -a "$SUMMARY"
sleep 20  # let ray/java of the pilot fully tear down + free VRAM

# ---- 2) TRAIN vanilla (plain timecap; no crd/risk/cca machinery) ----
"$PY" entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$EXP" \
  --total-timesteps "$STEPS" --num-workers "$WORKERS" --seed "$SEED" \
  --output-dir "$RUNDIR" --no-wandb > "$RUNDIR.train.log" 2>&1
echo "[mon $(date +%H:%M:%S)] training done rc=$?" | tee -a "$SUMMARY"

CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "[van $EXP s$SEED] NO CHECKPOINT" | tee -a "$SUMMARY"; exit 0; fi
echo "[mon] checkpoint=$CK" | tee -a "$SUMMARY"
mkdir -p "$RUNDIR/eval"

grab () {  # $1=logfile -> "carbon completion"
  local cc cf
  cc=$(grep -a "Avg Carbon Emission" "$1" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
  cf=$(grep -a "Avg Finished"        "$1" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
  echo "${cc:-?} ${cf:-?}"
}

# ---- 3a) EVAL clean (own carbon) ----
FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
"$PY" -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" --output "/tmp/van_local_s${SEED}_clean.csv" > "$RUNDIR/eval/clean.log" 2>&1
read cc cf < <(grab "$RUNDIR/eval/clean.log")
echo "[van $EXP s$SEED clean] carbon=$cc completion=$cf" | tee -a "$SUMMARY"

# ---- 3b) EVAL perturbations: anti x6 + shuffle x3 + blend x3 (unified protocol) ----
run_pert () {  # $1=mode $2=reps $3=seedbase
  local mode="$1" reps="$2" base="$3" r ac af
  for r in $(seq 1 "$reps"); do
    FORECAST_PERTURB_MODE="$mode" FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
    "$PY" -m src.baselines.evaluate --experiment "$EXP" \
      --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 1 --seed $((base+r)) --output "/tmp/van_local_s${SEED}_${mode}$r.csv" > "$RUNDIR/eval/${mode}$r.log" 2>&1
    read ac af < <(grab "$RUNDIR/eval/${mode}$r.log")
    echo "[van $EXP s$SEED ${mode}-r$r] carbon=$ac completion=$af" | tee -a "$SUMMARY"
  done
}
run_pert anti 6 3000
run_pert shuffle 3 3100
run_pert blend 3 3200
echo "[mon $(date +%H:%M:%S)] vanilla s${SEED} COMPLETE." | tee -a "$SUMMARY"
