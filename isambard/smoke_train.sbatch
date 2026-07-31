#!/bin/bash
# SMOKE: single EU-CRD training job, short budget, to measure single-job speed on
# Isambard (no contention) and confirm the pipeline trains + checkpoints. Extrapolate
# steps/hour to size the real walltime. No eval — training throughput only.
#SBATCH --job-name=smoke
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=1:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
PROJECTDIR=/projects/u6kd
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml"
export EVAL_CONFIG_PATH="$CFG"
EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4
STEPS=${STEPS:-24000}
WORKERS=${WORKERS:-6}
RUNDIR="$SCRATCHDIR/rl-runs/smoke_eucrd_$SLURM_JOB_ID"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[smoke] node=$(hostname) EXP=$EXP STEPS=$STEPS WORKERS=$WORKERS start=$(date +%s)"
t0=$(date +%s)
python entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$EXP" \
  --total-timesteps "$STEPS" --num-workers "$WORKERS" --seed 1 \
  --output-dir "$RUNDIR" --no-wandb
rc=$?
t1=$(date +%s)
dt=$((t1 - t0))
CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | tail -1 | grep -oE "checkpoint_[0-9]+")
echo "[smoke] rc=$rc trained ${STEPS} steps in ${dt}s (=$(awk "BEGIN{printf \"%.1f\", $STEPS/($dt+1)}") steps/s)"
echo "[smoke] checkpoint=${CK:-NONE}"
echo "[smoke] EXTRAPOLATION: 600k steps would take ~$(awk "BEGIN{printf \"%.1f\", 600000.0/($STEPS/($dt+1))/3600}") hours (single job, no contention)"
