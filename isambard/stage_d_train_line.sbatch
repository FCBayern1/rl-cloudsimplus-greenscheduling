#!/bin/bash
# One Stage D long-run line (400k steps) on one GH200. Twenty of these cover five seeds.
#   sbatch --export=ALL,SEED=20260904,LINE=NV isambard/stage_d_train_line.sbatch
#SBATCH --job-name=sdtrain
#SBATCH --account=brics.u6tx
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=60G
#SBATCH --time=24:00:00
#SBATCH --output=logs_slurm/%x_%j.out
set -uo pipefail
: "${SEED:?}"; : "${LINE:?}"

BASE=$PROJECTDIR/joshua
REPO=$BASE/rl-cloudsimplus-greenscheduling
source $BASE/miniforge3/etc/profile.d/conda.sh
conda activate sd
export TMPDIR=$SCRATCHDIR/tmp RAY_TMPDIR=$SCRATCHDIR/ray
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
export RAY_LIMIT_CPUS=8 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0
export STAGE_D_PYTHON=$(command -v python)
export STAGE_D_DATA_PATH=$SCRATCHDIR
# Optional: a second preregistration (CCA-PG baseline) reuses these scripts with its own
# config, line names and output roots, so the frozen four-line run is untouched.
[ -n "${STAGE_D_CONFIG:-}" ] && export STAGE_D_CONFIG
[ -n "${STAGE_D_SUFFIX:-}" ] && export STAGE_D_SUFFIX
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd "$REPO"
echo "[$(date '+%F %T')] node $(hostname) seed $SEED line $LINE commit $(git rev-parse --short HEAD)"
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" freeze || { echo "freeze FAILED"; exit 1; }
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" train_line "$LINE" || { echo "train FAILED"; exit 1; }
echo "[$(date '+%F %T')] seed $SEED line $LINE done"
