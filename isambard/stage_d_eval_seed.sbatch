#!/bin/bash
# One seed's checkpoint check, paired init-hash gate, 504 deployment evaluations and CRD
# statistics. Submitted with a dependency on that seed's four training jobs.
#   sbatch --dependency=afterok:<4 ids> --export=ALL,SEED=20260904 isambard/stage_d_eval_seed.sbatch
#SBATCH --job-name=sdeval
#SBATCH --account=brics.u6tx
#SBATCH --partition=workq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120G
#SBATCH --time=12:00:00
#SBATCH --output=logs_slurm/%x_%j.out
set -uo pipefail
: "${SEED:?}"

BASE=$PROJECTDIR/joshua
REPO=$BASE/rl-cloudsimplus-greenscheduling
source $BASE/miniforge3/etc/profile.d/conda.sh
conda activate sd
export TMPDIR=$SCRATCHDIR/tmp RAY_TMPDIR=$SCRATCHDIR/ray
mkdir -p "$TMPDIR" "$RAY_TMPDIR"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 TORCH_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONHASHSEED=0 STAGE_D_EVAL_WORKERS=12
export STAGE_D_PYTHON=$(command -v python)
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd "$REPO"
echo "[$(date '+%F %T')] node $(hostname) seed $SEED evaluation"
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" check_init_pairs || exit 1
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" check || exit 1
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" eval || exit 1
python g1/compressed_timecap_s2/stage_d_longrun.py seed "$SEED" crd
echo "[$(date '+%F %T')] seed $SEED evaluation done"
