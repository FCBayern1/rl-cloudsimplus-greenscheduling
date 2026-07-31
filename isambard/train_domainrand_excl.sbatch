#!/bin/bash
# Domain-randomization baseline (cross-comparison for the paper): train the STANDARD
# forecast policy WITH noisy forecasts injected during training (FORECAST_PERTURB_MODE=noise),
# so it learns to discount the forecast. Then eval on CLEAN (own carbon) + ANTI (robustness).
# Expected: clean carbon worse than clean-trained vanilla (learned to distrust), anti more
# stable than vanilla but still worse than EU-CRD -> "the smart not-trust route can't match
# EU-CRD's have-both". Training noise is GENERIC (not the anti test dist) = fair comparison.
#SBATCH --job-name=dr
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=20:00:00
#SBATCH --output=%x_%j.out
set -uo pipefail
: "${PROJECTDIR:?}" "${EXP:?}" "${OUT:?}" "${SEED:?}"
source "$HOME/miniforge3/bin/activate" rl
export JAVA_HOME="$CONDA_PREFIX"
export GATEWAY_LIBS="$PROJECTDIR/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib"
export RAY_LIMIT_CPUS=24
export RAY_TMPDIR="$HOME/rt/$SLURM_JOB_ID"; export TMPDIR="$RAY_TMPDIR"; mkdir -p "$RAY_TMPDIR"
CFG="${CFG:-$PROJECTDIR/rl-cloudsimplus-greenscheduling/config_C.yml}"
export EVAL_CONFIG_PATH="$CFG"
OUTBASE="${OUTBASE:-$SCRATCHDIR/rl-runs}"
RUNDIR="$OUTBASE/${OUT}_s${SEED}"
SUMMARY="$OUTBASE/domainrand_summary.txt"
DR_EPS="${DR_EPS:-0.25}"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[dr] EXP=$EXP SEED=$SEED DR_EPS=$DR_EPS node=$(hostname)"
# ---- TRAIN with noisy forecast (domain randomization) ----
FORECAST_PERTURB_MODE=noise FORECAST_PERTURB_EPS="$DR_EPS" \
python entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$EXP" \
  --total-timesteps "${STEPS:-600000}" --num-workers "${WORKERS:-8}" --seed "$SEED" \
  --output-dir "$RUNDIR" --no-wandb

CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "[dr $OUT s$SEED] NO CHECKPOINT" >> "$SUMMARY"; exit 0; fi

# ---- EVAL clean (own carbon; NO perturbation) ----
FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" --output "/tmp/dr_${OUT}_s${SEED}_clean.csv" > "$RUNDIR/eval_clean.log" 2>&1
cc=$(grep -a "Avg Carbon Emission" "$RUNDIR/eval_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
cf=$(grep -a "Avg Finished" "$RUNDIR/eval_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[dr $OUT s$SEED clean] carbon=${cc:-?} completion=${cf:-?}" >> "$SUMMARY"

# ---- EVAL perturbations: anti x6 (main-table metric) + shuffle x3 + blend x3 ----
run_pert () {  # $1=mode  $2=reps  $3=seedbase
  local mode="$1" reps="$2" base="$3" r
  for r in $(seq 1 "$reps"); do
    FORECAST_PERTURB_MODE="$mode" FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
    python -m src.baselines.evaluate --experiment "$EXP" \
      --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 1 --seed $((base+r)) --output "/tmp/dr_${OUT}_s${SEED}_${mode}$r.csv" > "$RUNDIR/eval_${mode}$r.log" 2>&1
    local ac af
    ac=$(grep -a "Avg Carbon Emission" "$RUNDIR/eval_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
    af=$(grep -a "Avg Finished" "$RUNDIR/eval_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
    echo "[dr $OUT s$SEED ${mode}-r$r] carbon=${ac:-?} completion=${af:-?}" >> "$SUMMARY"
  done
}
run_pert anti 6 3000
run_pert shuffle 3 3100
run_pert blend 3 3200
