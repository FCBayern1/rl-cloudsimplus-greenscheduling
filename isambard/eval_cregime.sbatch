#!/bin/bash
# EVAL-ONLY (no training): load a saved C-regime checkpoint + argmax-eval it, with the
# RAY_LIMIT_CPUS guard so the eval process does NOT hit the 288-worker registration storm
# (which hung the auto-eval inside train_cregime.sbatch). Parameterised via --export EXP/CK/SEED/OUT.
#SBATCH --job-name=ev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=02:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${CK:?}" "${SEED:?}" "${OUT:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
# STOCHASTIC=1 → Gumbel-max sampling (spreads the 128 routing slots; argmax collapses them onto a
# few DCs → completion confound). Writes to a SEPARATE summary so it won't clobber the argmax run.
STOCH_FLAG=""; TAG="argmax"
if [ -n "${STOCHASTIC:-}" ]; then STOCH_FLAG="--stochastic"; TAG="stoch"; fi
SUMMARY="$OUTBASE/cregime_seeds_${TAG}_summary.txt"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[eval-only $TAG] EXP=$EXP SEED=$SEED CK=$CK node=$(hostname)"
DECODE_TOPK=0 python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" $STOCH_FLAG --output "/tmp/eval_${OUT}_s${SEED}_${TAG}.csv" > "$OUTBASE/${OUT}_s${SEED}_${TAG}.log" 2>&1

# Read the SAME log this run wrote (was hardcoded to _evalonly.log → stoch runs harvested the
# stale argmax log → wrong summary values; the per-job logs were always correct). head -1 is
# defensive in case a line ever carries more than one number.
c=$(grep -a "Avg Carbon Emission" "$OUTBASE/${OUT}_s${SEED}_${TAG}.log" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+" | head -1)
f=$(grep -a "Avg Finished"        "$OUTBASE/${OUT}_s${SEED}_${TAG}.log" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[$EXP seed=$SEED] carbon=${c:-?} completion=${f:-?}" >> "$SUMMARY"
