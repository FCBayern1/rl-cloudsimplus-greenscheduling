#!/bin/bash
# Re-evaluate an EXISTING checkpoint with --stochastic (Gumbel-max sampling), the
# faithful iso-completion decode. The earlier evals used greedy argmax (default),
# which collapses EU-CRD's 128 same-obs routing slots onto one DC -> false ~9%
# completion. Training was fine (completion 100%); only the eval decode was wrong.
# No re-training: reads the ckpt_10 already on scratch.
#SBATCH --job-name=evs
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=16:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${EXP:?}" "${OUT:?}" "${SEED:?}" "${RUNDIR:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="/projects/u6kd/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24; export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="/projects/u6kd/rl-cloudsimplus-greenscheduling/config_C.yml"; export EVAL_CONFIG_PATH="$CFG"
SUMMARY="$SCRATCHDIR/rl-runs/stoch_summary.txt"
# Checkpoint selection: NEWEST run dir by mtime, then highest ckpt within it.
# (A plain lexicographic sort across PPO_*/checkpoint_* compares the random
# trial hash in the dir name — an aborted earlier run can win and the eval
# silently scores checkpoint_000000. Bit us on eucrdv5 s1/s2, 2026-07-18.)
NEWEST_RUN=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
CK=$(ls -d "${NEWEST_RUN}"checkpoint_* 2>/dev/null | sort | tail -1)
cd /projects/u6kd/rl-cloudsimplus-greenscheduling/drl-manager
echo "[evs] EXP=$EXP SEED=$SEED CK=$CK node=$(hostname)"
[ -z "$CK" ] && { echo "[evs $OUT s$SEED] NO CHECKPOINT" >>"$SUMMARY"; exit 0; }
grab(){ local cc cf; cc=$(grep -a "Avg Carbon Emission" "$1"|grep -av Calling|tail -1|grep -oE "[0-9.]+"|head -1); cf=$(grep -a "Avg Finished" "$1"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1); echo "${cc:-?} ${cf:-?}"; }
ev(){ local mode=$1 sd=$2 tag=$3; local lg="$RUNDIR/stoch_${tag}.log"; FORECAST_PERTURB_MODE=$mode FORECAST_PERTURB_EPS=1.0 \
  python -m src.baselines.evaluate --experiment "$EXP" --global rllib --local rllib --checkpoint "$CK" \
  --new-api --shared-local --global-defer --stochastic --episodes 1 --seed "$sd" --output "/tmp/stoch_${OUT}_s${SEED}_${tag}.csv" >"$lg" 2>&1
  read c f < <(grab "$lg"); echo "[evs $OUT s$SEED ${tag}] carbon=$c completion=$f" >>"$SUMMARY"; }
ev none  "$SEED" clean
for r in 1 2 3; do ev anti $((3000+r)) anti-r$r; done
for r in 1 2; do ev shuffle $((3100+r)) shuffle-r$r; ev blend $((3200+r)) blend-r$r; done
echo "[evs] DONE $OUT s$SEED" >>"$SUMMARY"
