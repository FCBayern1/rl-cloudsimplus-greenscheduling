#!/bin/bash
# C-regime arm (train + argmax eval) on one GH200. Parameterised via --export EXP/OUT/SEED.
# Uses config_C.yml (2x long jobs + obs_cloudlet_mi_high=1e7). Requires the ray zero_padding
# Dict-obs patch (setup_env.sh step 4b) — else the learner crashes on the [B,128]vs[B,T] mismatch.
#SBATCH --job-name=rlc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${OUT:?}" "${SEED:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"                       # so the eval reads the same (C-trace) config
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
RUNDIR="$OUTBASE/${OUT}_s${SEED}"
SUMMARY="$OUTBASE/cregime_seeds_summary.txt"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[sbatch] EXP=$EXP SEED=$SEED CFG=$CFG node=$(hostname)"
python entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$EXP" \
  --total-timesteps "${STEPS:-600000}" --num-workers "${WORKERS:-8}" --seed "$SEED" \
  --output-dir "$RUNDIR" --no-wandb

CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -n "$CK" ]; then
  DECODE_TOPK=0 python -m src.baselines.evaluate --experiment "$EXP" \
    --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 1 --seed "$SEED" --output "/tmp/eval_${OUT}_s${SEED}.csv" > "$RUNDIR/eval.log" 2>&1
  c=$(grep -a "Avg Carbon Emission" "$RUNDIR/eval.log" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+")
  f=$(grep -a "Avg Finished" "$RUNDIR/eval.log" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+%")
  echo "[$EXP seed=$SEED] carbon=${c:-?} completion=${f:-?}" >> "$SUMMARY"
else
  echo "[$EXP seed=$SEED] NO CHECKPOINT" >> "$SUMMARY"
fi
