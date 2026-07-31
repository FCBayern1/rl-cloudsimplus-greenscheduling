#!/bin/bash
# Risk-averse baseline (cross-comparison): train the standard forecast policy with a
# risk-averse objective (CVaR / risk-sensitive / mean-variance — selected via the
# experiment's crd.risk config), CLEAN forecast during training. Then eval CLEAN (own
# carbon) + ANTI x6 (robustness). Expected: risk objectives make the policy globally
# conservative (some robustness) but can't target forecast-specific risk -> worse
# clean carbon than EU-CRD, robustness between vanilla and EU-CRD.
#SBATCH --job-name=excl
#SBATCH --exclusive
#SBATCH --nodes=1
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
SUMMARY="$OUTBASE/risk_summary.txt"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

echo "[risk] EXP=$EXP SEED=$SEED node=$(hostname)"
# ---- TRAIN (clean forecast; risk objective lives in crd.risk of the experiment) ----
python entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$EXP" \
  --total-timesteps "${STEPS:-600000}" --num-workers "${WORKERS:-8}" --seed "$SEED" \
  --output-dir "$RUNDIR" --no-wandb

CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "[risk $OUT s$SEED] NO CHECKPOINT" >> "$SUMMARY"; exit 0; fi

# ---- EVAL clean ----
FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" --output "/tmp/risk_${OUT}_s${SEED}_clean.csv" > "$RUNDIR/eval_clean.log" 2>&1
cc=$(grep -a "Avg Carbon Emission" "$RUNDIR/eval_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
cf=$(grep -a "Avg Finished" "$RUNDIR/eval_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[risk $OUT s$SEED clean] carbon=${cc:-?} completion=${cf:-?}" >> "$SUMMARY"

# ---- EVAL perturbations: anti x6 (main-table metric) + shuffle x3 + blend x3 ----
run_pert () {  # $1=mode  $2=reps  $3=seedbase
  local mode="$1" reps="$2" base="$3" r
  for r in $(seq 1 "$reps"); do
    FORECAST_PERTURB_MODE="$mode" FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
    python -m src.baselines.evaluate --experiment "$EXP" \
      --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 1 --seed $((base+r)) --output "/tmp/risk_${OUT}_s${SEED}_${mode}$r.csv" > "$RUNDIR/eval_${mode}$r.log" 2>&1
    local ac af
    ac=$(grep -a "Avg Carbon Emission" "$RUNDIR/eval_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
    af=$(grep -a "Avg Finished" "$RUNDIR/eval_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
    echo "[risk $OUT s$SEED ${mode}-r$r] carbon=${ac:-?} completion=${af:-?}" >> "$SUMMARY"
  done
}
run_pert anti 6 3000
run_pert shuffle 3 3100
run_pert blend 3 3200
