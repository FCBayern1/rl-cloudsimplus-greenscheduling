#!/bin/bash
# EVAL-ONLY recovery: the 07-08 cross-comparison jobs trained to 600k but their
# eval suite got truncated by the 20h walltime (all TIMEOUT). Checkpoints survived.
# This re-runs ONLY the eval suite (clean + anti x6 + shuffle x3 + blend x3) on the
# saved checkpoint of $OUT_s$SEED — no 15h retrain. 6h walltime is ample.
#SBATCH --job-name=ev
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=06:00:00
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
SUMMARY="$OUTBASE/eval_recover_summary.txt"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"

CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "[ev $OUT s$SEED] NO CHECKPOINT" >> "$SUMMARY"; exit 0; fi
echo "[ev] EXP=$EXP OUT=$OUT SEED=$SEED CK=$CK node=$(hostname)"

# clean
FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
python -m src.baselines.evaluate --experiment "$EXP" \
  --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
  --episodes 1 --seed "$SEED" --output "/tmp/ev_${OUT}_s${SEED}_clean.csv" > "$RUNDIR/ev_clean.log" 2>&1
cc=$(grep -a "Avg Carbon Emission" "$RUNDIR/ev_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
cf=$(grep -a "Avg Finished" "$RUNDIR/ev_clean.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[ev $OUT s$SEED clean] carbon=${cc:-?} completion=${cf:-?}" >> "$SUMMARY"

run_pert () {  # $1=mode $2=reps $3=seedbase
  local mode="$1" reps="$2" base="$3" r
  for r in $(seq 1 "$reps"); do
    FORECAST_PERTURB_MODE="$mode" FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
    python -m src.baselines.evaluate --experiment "$EXP" \
      --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 1 --seed $((base+r)) --output "/tmp/ev_${OUT}_s${SEED}_${mode}$r.csv" > "$RUNDIR/ev_${mode}$r.log" 2>&1
    local ac af
    ac=$(grep -a "Avg Carbon Emission" "$RUNDIR/ev_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
    af=$(grep -a "Avg Finished" "$RUNDIR/ev_${mode}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
    echo "[ev $OUT s$SEED ${mode}-r$r] carbon=${ac:-?} completion=${af:-?}" >> "$SUMMARY"
  done
}
run_pert anti 6 3000
run_pert shuffle 3 3100
run_pert blend 3 3200
echo "[ev $OUT s$SEED DONE]" >> "$SUMMARY"
