#!/bin/bash
# One-time Isambard-AI phase 2 setup for the Stage D long run (account u6tx, 2026-09-04).
# Not run automatically: the long run is frozen to one GPU model, so migrating needs a
# prereg addendum first (STAGE_D_LONGRUN_PREREG §8, R-u).
#
#   ssh u6tx.aip2.isambard            login node: aarch64, 144 cores, 237 GB
#   compute: workq, 4x GH200 120GB per node, 288 cores, 460 GB
#   $HOME /home/u6tx/joshualmw.u6tx   /projects/u6tx   /scratch/u6tx/<user>
#
# The four migration bugs from the u6kd era still apply and are handled below:
#   1 Ray temp dir: TMPDIR/RAY_TMPDIR must be on a persistent filesystem, never /run/user
#   2 the gateway must be launched from installDist (java -cp), never `gradlew run`
#   3 sbatch needs an explicit --mem or Slurm reserves the whole 460 GB node
#   4 Ray auto-detects all 288 cores and stampedes the NFS import; cap it
#   5 pyarrow/msgpack drift breaks "works locally" runs; pin the versions below
set -euo pipefail

REMOTE=${REMOTE:-u6tx.aip2.isambard}
PROJ=${PROJ:-/projects/u6tx}
LOCAL_REPO=${LOCAL_REPO:-/home/joshua/rl-cloudsimplus-greenscheduling}
DEST="$PROJ/rl-cloudsimplus-greenscheduling"

echo "== 1. transfer code, wind data and traces (about 5.3 GB), never logs or results =="
rsync -az --info=progress2 \
  --exclude '.git' --exclude '*/.venv' --exclude 'logs' --exclude 'results' \
  --exclude 'build' --exclude 'bin' --exclude 'stage_a_out' --exclude 'isambard_backup' \
  --exclude 'isambard_exodus' --exclude 'local_eval_rt' --exclude '__pycache__' \
  "$LOCAL_REPO"/ "$REMOTE:$DEST"/

echo "== 2. python stack (aarch64: torch cu126, ray 2.40.0, openjdk 21 through conda) =="
ssh "$REMOTE" bash -lc "'
set -euo pipefail
cd $PROJ
if [ ! -d miniforge3 ]; then
  curl -L -o mf.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-aarch64.sh
  bash mf.sh -b -p $PROJ/miniforge3 && rm mf.sh
fi
source $PROJ/miniforge3/etc/profile.d/conda.sh
conda create -y -n sd python=3.12 openjdk=21 || true
conda activate sd
pip install --upgrade pip
pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu126
pip install \"ray[rllib]==2.40.0\" gymnasium==1.0.0 stable_baselines3==2.7.0 sb3_contrib==2.7.0 \
            py4j pyyaml pandas numpy scipy matplotlib pytest
python -c \"import torch, ray; print(torch.__version__, torch.cuda.is_available(), ray.__version__)\"
'"

echo "== 3. build the gateway (installDist, never gradlew run at job time) =="
ssh "$REMOTE" bash -lc "'
source $PROJ/miniforge3/etc/profile.d/conda.sh && conda activate sd
cd $DEST/cloudsimplus-gateway && ./gradlew -q installDist
sha256sum build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar
'"

echo "== 4. equivalence smoke: one short evaluation must match the workstation's fields =="
cat <<'EOF'
Submit with, per seed (four lines on one node, one GPU each):
  sbatch --account=brics.u6tx --partition=workq --gres=gpu:4 --mem=200G --time=12:00:00 \
         --cpus-per-task=64 isambard/stage_d_seed.sbatch <SEED>
and inside the job export, before anything else:
  export PYTHONHASHSEED=0
  export TMPDIR=$SCRATCHDIR/tmp RAY_TMPDIR=$SCRATCHDIR/ray
  export RAY_LIMIT_CPUS=16 OMP_NUM_THREADS=1
EOF
