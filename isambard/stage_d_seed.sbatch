#!/bin/bash
# One Stage D long-run seed on one Isambard-AI aip2 node: four lines (N_V, V, N_E, E) in
# parallel, one GH200 each, 400k steps, then the 504 deployment evaluations.
#
#   sbatch --export=ALL,SEED=20260904 isambard/stage_d_seed.sbatch
#
# Migration to this hardware requires a STAGE_D_LONGRUN_PREREG addendum first (§8 R-u fixes
# the GPU model); this script is the mechanism, not the authorisation.
#SBATCH --job-name=stage_d_seed
#SBATCH --account=brics.u6tx
#SBATCH --partition=workq
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --mem=200G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${SEED:?set SEED=<seed> in --export}"

BASE=$PROJECTDIR/joshua
REPO=$BASE/rl-cloudsimplus-greenscheduling
source $BASE/miniforge3/etc/profile.d/conda.sh
conda activate sd

# Bug 1: Slurm points TMPDIR at /run/user/<uid>, which systemd reclaims mid-job.
export TMPDIR=$SCRATCHDIR/tmp RAY_TMPDIR=$SCRATCHDIR/ray
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
# Bug 4: Ray sees all 288 node CPUs through the cgroup and stampedes the NFS import.
export RAY_LIMIT_CPUS=16
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Frozen determinism (long-run prereg §8): the EU-CRD module build is order-sensitive.
export PYTHONHASHSEED=0
# Four lines, one GPU each.
export STAGE_D_PARALLEL_LINES=4 STAGE_D_GPU_MAP="NV:0,V:1,NE:2,E:3"
export STAGE_D_EVAL_WORKERS=12

cd "$REPO"
echo "[$(date '+%F %T')] node $(hostname) seed $SEED commit $(git rev-parse --short HEAD)"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader

export STAGE_D_PYTHON=$(command -v python)

for phase in freeze train check eval crd; do
  python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" "$phase" || { echo "phase $phase FAILED"; exit 1; }
done
echo "[$(date '+%F %T')] seed $SEED done"
