#!/bin/bash
#SBATCH --job-name=rlv2
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=%x_%j.out
# RL_V2 health smoke (reports/RL_V2_SMOKE_PREREG.md, Addendum A + B) on one GH200 (96 GB HBM):
# four lines ONE AT A TIME with fail-fast, then the init checks. The workstation's 16 GB card
# cannot hold one line (peak 15.33 GiB; Addendum B), and the batch, architecture and budget are
# not changed to make it fit, so the training moves here. The readings are run afterwards on
# whichever machine is free; only the checkpoints are needed for them.
#   sbatch --export=ALL,PROJECTDIR=$PROJECTDIR rl_v2_smoke.sbatch
set -euo pipefail
: "${PROJECTDIR:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
REPO="$PROJECTDIR/rl-cloudsimplus-greenscheduling"
export GATEWAY_LIBS="$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=16
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"
mkdir -p "$RAY_TMPDIR"
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0 OFFSET_GRID_DENSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ORACLE_WIND_DIR="$REPO/cloudsimplus-gateway/src/main/resources/windProduction/simplified"
cd "$REPO/drl-manager"
CFG=../g1/compressed_timecap_s2/config_rl_v2.yml
OUT=../g1/compressed_timecap_s2/stage_a_out/rl_v2
S=20260907; STEPS=56000
log(){ echo "[$(date '+%F %T')] $*"; }
mkdir -p logs/rl_v2
lastck(){ ls -d logs/rl_v2/$1_s$S/*/checkpoint_* 2>/dev/null | grep -v checkpoint_init | sort -V | tail -1; }
for L in NV V NE E; do
  log "training $L"
  python entrypoint_rlmodule_gtrxl.py --config $CFG --experiment rl2_${L}_s2_r48_w72_c3_n35 \
    --total-timesteps $STEPS --num-workers 0 --seed $S --no-wandb --output-dir logs/rl_v2/${L}_s$S \
    > logs/rl_v2/${L}_s$S.log 2>&1
  rc=$?; log "train $L exit=$rc"
  if [ $rc -ne 0 ] || [ -z "$(lastck $L)" ]; then
    R="INVALID_SMOKE_ISAMBARD_TRAIN_${L}_EXIT_${rc}$(grep -qi 'OutOfMemoryError' logs/rl_v2/${L}_s$S.log && echo _OOM)"
    log "ABORT: $R"; echo "$R" > $OUT/INVALID_RUN.txt; exit 1
  fi
done
log "all four lines trained; checkpoints:"
for L in NV V NE E; do log "  $L init=$(ls -d logs/rl_v2/${L}_s$S/*/checkpoint_init 2>/dev/null | head -1) last=$(lastck $L)"; done
log "smoke training finished; run the readings with rl_v2_smoke.sh (readings-only mode) or on the workstation"
