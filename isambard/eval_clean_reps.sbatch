#!/bin/bash
# Extra CLEAN-condition reps (stochastic decode) for an existing checkpoint. The main
# eval_stoch protocol runs clean only ONCE per seed, so a single tail draw (e.g.
# eucrd s2 clean 56.9% with routed=100%, waste=85%) dominates the cell. This adds
# REPS more clean episodes with distinct env seeds to separate tail-draw from systematic.
#SBATCH --job-name=evc
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=8:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${EXP:?}" "${OUT:?}" "${SEED:?}" "${RUNDIR:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="/projects/u6kd/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24; export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="/projects/u6kd/rl-cloudsimplus-greenscheduling/config_C.yml"; export EVAL_CONFIG_PATH="$CFG"
SUMMARY="$SCRATCHDIR/rl-runs/stoch_summary.txt"
CK=$(ls -d "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | grep -oE ".*checkpoint_[0-9]+" | sort | tail -1)
cd /projects/u6kd/rl-cloudsimplus-greenscheduling/drl-manager
[ -z "$CK" ] && { echo "[evc $OUT s$SEED] NO CHECKPOINT" >>"$SUMMARY"; exit 0; }
for r in 2 3 4 5; do
  lg="$RUNDIR/stoch_clean-r$r.log"
  FORECAST_PERTURB_MODE=none \
  python -m src.baselines.evaluate --experiment "$EXP" --global rllib --local rllib --checkpoint "$CK" \
    --new-api --shared-local --global-defer --stochastic --episodes 1 --seed $((SEED*1000+r)) \
    --output "/tmp/evc_${OUT}_s${SEED}_r${r}.csv" >"$lg" 2>&1
  cc=$(grep -a "Avg Carbon Emission" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+"|head -1)
  cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  echo "[evs $OUT s$SEED clean-r$r] carbon=${cc:-?} completion=${cf:-?}" >>"$SUMMARY"
done
echo "[evc $OUT s$SEED done]" >>"$SUMMARY"
