#!/bin/bash
# Carbon-aware scheduling baseline ("Let's Wait Awhile", Wiesner et al. 2021):
# forecast-driven temporal+spatial shifting = global green_forecast routing + DEFER to greener
# windows + drain local. No training (deterministic heuristic). ONE mode x REPS reps, 8h walltime.
# Params: EXP, OUT, SEED, MODE(clean|anti|shuffle|blend), [REPS=3], [SEEDBASE=7000]
#SBATCH --job-name=eh
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=08:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${OUT:?}" "${SEED:?}" "${MODE:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
RUNDIR="$OUTBASE/${OUT}_s${SEED}"; mkdir -p "$RUNDIR"
SUMMARY="$OUTBASE/eval_full_summary.txt"
REPS="${REPS:-3}"; SEEDBASE="${SEEDBASE:-7000}"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"
PMODE="$MODE"; [ "$MODE" = "clean" ] && PMODE="none"
echo "[eh] carbon-aware (green_forecast+defer+drain) EXP=$EXP MODE=$MODE node=$(hostname)"
for r in $(seq 1 "$REPS"); do
  FORECAST_PERTURB_MODE="$PMODE" FORECAST_PERTURB_EPS=1.0 \
  python -m src.baselines.evaluate --experiment "$EXP" \
    --global green_forecast --local drain --global-defer \
    --episodes 1 --seed $((SEEDBASE+r)) --output "/tmp/eh_${OUT}_s${SEED}_${MODE}$r.csv" \
    > "$RUNDIR/eh_${MODE}$r.log" 2>&1
  ac=$(grep -a "Avg Carbon Emission" "$RUNDIR/eh_${MODE}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
  af=$(grep -a "Avg Finished" "$RUNDIR/eh_${MODE}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
  echo "[em $OUT s$SEED ${MODE}-r$r] carbon=${ac:-?} completion=${af:-?}" >> "$SUMMARY"
done
echo "[em $OUT s$SEED $MODE DONE]" >> "$SUMMARY"
