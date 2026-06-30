#!/bin/bash
# E3 forecast-SHIFT collapse probe. Eval a TIMECAP checkpoint with injected forecast error
# (FORECAST_PERTURB_MODE/EPS, gated in hierarchical_multidc_env._perturb_forecast). Sweeping the
# error reveals whether the forecast-trusting policy degrades GRACEFULLY toward noforecast
# (~0.387 stoch) or COLLAPSES below it (over-trust → EU-CRD has room). Parameterised via
# --export EXP/CK/SEED/PMODE/PEPS. Uses --stochastic (stable completion → clean carbon signal).
#SBATCH --job-name=ec
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${CK:?}" "${SEED:?}" "${PMODE:?}" "${PEPS:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
export FORECAST_PERTURB_MODE="$PMODE"
export FORECAST_PERTURB_EPS="$PEPS"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
# STOCH=1 (default) → stochastic decode; STOCH=0 → argmax (reveals SHARP commitment to the
# perturbed forecast → likely a bigger collapse). Separate summary per decode.
STOCH_FLAG="--stochastic"; DTAG="stoch"
if [ "${STOCH:-1}" = "0" ]; then STOCH_FLAG=""; DTAG="argmax"; fi
SUMMARY="$OUTBASE/collapse_${DTAG}_summary.txt"
TAG="${DTAG}_${PMODE}_${PEPS}"
LOG="$OUTBASE/collapse_s${SEED}_${TAG}.log"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[collapse $DTAG] EXP=$EXP SEED=$SEED PMODE=$PMODE PEPS=$PEPS node=$(hostname)"
DECODE_TOPK=0 python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" $STOCH_FLAG --output "/tmp/collapse_s${SEED}_${TAG}.csv" > "$LOG" 2>&1

c=$(grep -a "Avg Carbon Emission" "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+" | head -1)
f=$(grep -a "Avg Finished"        "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[mode=$PMODE eps=$PEPS seed=$SEED] carbon=${c:-?} completion=${f:-?}" >> "$SUMMARY"
