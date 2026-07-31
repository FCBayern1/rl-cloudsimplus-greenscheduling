#!/bin/bash
# Paper-1 baseline consolidation: run a heuristic/metaheuristic GLOBAL scheduler (EVAL-ONLY, no
# training, no checkpoint) on the C-regime (probe_C + real wind, 5-DC) to compare against the RL.
# HiGreen's heuristic baseline was ONLY Round-Robin; this adds the GREEN-AWARE heuristics it missed
# (green_aware/green_queue_balanced/min_brown_power/weighted_score) + GA/PSO. evaluate.py auto-measures
# decision latency (CLAUDE.md overhead requirement). Parameterised via --export SCHED/SEED.
#SBATCH --job-name=bl
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=04:00:00
#SBATCH --output=%x_%j.out
set -euo pipefail
: "${PROJECTDIR:?}" "${SCHED:?}" "${SEED:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
EXP="${EXP:-experiment_multi_5dc_carbon_v2_deferrable_gdpd}"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
SUMMARY="$OUTBASE/baseline_summary.txt"
LOG="$OUTBASE/baseline_${SCHED}_s${SEED}.log"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[baseline] SCHED=$SCHED SEED=$SEED node=$(hostname)"
# Do NOT let an eval crash / empty grep abort before we record a summary line.
# (set -e + pipefail otherwise kills the job silently when a run produces no carbon.)
set +e
# deferrable_gdpd is dispatch_rate mode → local must DRAIN (dispatch max), not place (best_fit).
# DEFER=1 (default) wraps the heuristic with the forecast-driven DEFER lever for action-space
# parity with the arch-B RL (fair comparison). Set DEFER=0 for the no-defer variant.
DEFER_FLAG=""; [ "${DEFER:-1}" = "1" ] && DEFER_FLAG="--global-defer"
python -m src.baselines.evaluate --experiment "$EXP" \
  --global "$SCHED" --local "${LOCAL:-drain}" $DEFER_FLAG \
  --episodes 1 --seed "$SEED" --output "/tmp/baseline_${SCHED}_s${SEED}.csv" > "$LOG" 2>&1
rc=$?

c=$(grep -a "Avg Carbon Emission" "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+" | head -1)
f=$(grep -a "Avg Finished"        "$LOG" | grep -av "Calling" | tail -1 | grep -oE "[0-9.]+%" | head -1)
lat=$(grep -aiE "global.*decision.*mean|decision_us_mean|global_decision_us" "$LOG" | tail -1 | grep -oE "[0-9.]+" | head -1)
echo "[baseline $SCHED seed=$SEED] carbon=${c:-?} completion=${f:-?} decision_us_mean=${lat:-?} rc=$rc" >> "$SUMMARY"
