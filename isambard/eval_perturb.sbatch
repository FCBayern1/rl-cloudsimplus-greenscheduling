#!/bin/bash
# Perturbed-forecast eval of a saved checkpoint (Step-2/3 collapse matrix on Isambard).
# Params via --export: EXP, CK, SEED, OUT, MODE (none|blend|shuffle|anti), EPS (default 1.0).
# Writes [OUT sSEED MODE] carbon/completion/GreenRatio to perturb_matrix_summary.txt.
#SBATCH --job-name=pv
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${SEED:?}" "${OUT:?}" "${MODE:?}"
# CK may be a literal checkpoint path, or "auto" to resolve the newest
# checkpoint of $OUTBASE/${OUT}_s${SEED} at RUN time — lets these evals be
# pre-submitted with --dependency on the training job (survives operator
# disconnect; afterany + this guard also covers TIMEOUTed trainings that
# still wrote a checkpoint).
CK="${CK:-auto}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
SUMMARY="$OUTBASE/perturb_matrix_summary.txt"
if [ "$CK" = "auto" ]; then
  CK=$(ls -dt "$OUTBASE/${OUT}_s${SEED}"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
  if [ -z "$CK" ]; then
    echo "[$OUT s$SEED $MODE] NO CHECKPOINT (training failed?)" >> "$SUMMARY"
    exit 0
  fi
fi
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"
# Optional knobs: TAG labels the summary row (e.g. "repair-r1") so special runs
# (sentinel gate/repair, extra reps) never collide with the plain matrix rows;
# EVALSEED decouples the eval RNG seed from the training-seed dir (repetitions).
EVALSEED="${EVALSEED:-$SEED}"
LOG="$OUTBASE/${OUT}_s${SEED}_${MODE}${TAG:+_$TAG}.log"
echo "[perturb-eval] EXP=$EXP MODE=$MODE EPS=${EPS:-1.0} TAG=${TAG:-} CK=$CK"
FORECAST_PERTURB_MODE="$MODE" FORECAST_PERTURB_EPS="${EPS:-1.0}" DECODE_TOPK=0 \
python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$EVALSEED" --output "/tmp/${OUT}_s${SEED}_${MODE}${TAG:+_$TAG}.csv" > "$LOG" 2>&1
rc=$?
c=$(grep -a "Avg Carbon Emission" "$LOG" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
f=$(grep -a "Avg Finished" "$LOG" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
g=$(grep -aoE "GreenRatio=[0-9.]+%" "$LOG" | tail -1)
r=$(grep -a "ForecastResidualMonitor\[" "$LOG" | tail -1 | grep -oE "repaired=[0-9]+")
echo "[$OUT s$SEED $MODE${TAG:+-$TAG}] carbon=${c:-?} completion=${f:-?} ${g:-} ${r:-} rc=$rc" >> "$SUMMARY"
