#!/bin/bash
# ISOLATION TEST: RL global routing + DRAIN local (instead of the learned shared-local).
# Tells us whether the RL's global routing is competitive with gqb when both use drain local
# → isolates "is the weak link the RL's learned local dispatch or its global routing?".
#SBATCH --job-name=evd
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${CK:?}" "${SEED:?}" "${OUT:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
SUMMARY="$OUTBASE/godeye_drain_summary.txt"
STOCH_FLAG=""; TAG="argmax"
[ -n "${STOCHASTIC:-}" ] && { STOCH_FLAG="--stochastic"; TAG="stoch"; }
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"
LOG="$OUTBASE/${OUT}_s${SEED}_${TAG}.log"
echo "[godeye+drain $TAG] EXP=$EXP CK=$CK node=$(hostname)"
# RL global routing (checkpoint) + DRAIN local (dispatch-max heuristic, same as gqb).
DECODE_TOPK=0 python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local drain --checkpoint "$CK" --new-api --global-defer \
  --episodes 1 --seed "$SEED" $STOCH_FLAG --output "/tmp/${OUT}_s${SEED}_${TAG}.csv" > "$LOG" 2>&1
rc=$?
c=$(grep -a "Avg Carbon Emission" "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+" | head -1)
f=$(grep -a "Avg Finished"        "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+%" | head -1)
g=$(grep -aoE "GreenRatio=[0-9.]+%" "$LOG" | tail -1)
echo "[$OUT s$SEED $TAG] carbon=${c:-?} completion=${f:-?} ${g:-} rc=$rc" >> "$SUMMARY"
