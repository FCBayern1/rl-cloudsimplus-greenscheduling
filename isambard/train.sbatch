#!/bin/bash
#SBATCH --job-name=rl
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=10:00:00
#SBATCH --output=%x_%j.out
# NOTE: --mem=100G + --cpus-per-task=32 request a 1-GPU SLICE of the node (not the whole
# 460GB/288-core node). Without --mem, Slurm defaults to the full node's memory, so the job
# only schedules onto a COMPLETELY empty node — on a busy cluster (most nodes "mix") it waits
# forever. A sub-node slice lands on any node with one free GPU.
# One training arm on one GH200 (1 H100 + 72 Grace cores). Parameterised via --export env vars.
# Direct use:
#   sbatch --job-name=godeye --export=ALL,PROJECTDIR=$PROJECTDIR,\
#          EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd,OUT=iso_godeye,SEED=42 train.sbatch
set -euo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${OUT:?}"           # $PROJECTDIR is pre-set by Isambard
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"                     # gradle/Py4J use the conda JDK
# HPC: launch the gateway from pre-installed jars (gradlew installDist), NOT `gradlew run`
# (compute nodes have no internet; per-worker cold gradle builds hang past the walltime).
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
# Cap Ray's CPU pool (it otherwise sees all 288 node cores, ignoring the cgroup, and spawns
# ~288 workers that all import from NFS at once → registration timeout → hang). Keep >= workers+driver.
export RAY_LIMIT_CPUS=16
# Ray temp dir: do NOT use $LOCALDIR (=/run/user/<uid>) — systemd-logind reclaims it mid-job in batch,
# so it passes a writability check in the prolog but Ray then hits Permission denied. Use $HOME (shared,
# always writable, short enough for Ray's AF_UNIX socket-path limit). Per-job dir to avoid collisions.
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"
export TMPDIR="$RAY_TMPDIR"
mkdir -p "$RAY_TMPDIR"
echo "[sbatch] RAY_TMPDIR=$RAY_TMPDIR  node=$(hostname)"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"           # checkpoints/outputs -> scratch (5TiB)
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"
python entrypoint_rlmodule_gtrxl.py \
  --experiment "$EXP" \
  --total-timesteps "${STEPS:-600000}" \
  --num-workers "${WORKERS:-16}" \
  --seed "${SEED:-42}" \
  --output-dir "$OUTBASE/${OUT}_s${SEED:-42}" \
  --no-wandb
