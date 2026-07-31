#!/bin/bash
# Timeout-proof eval: ONE forecast-perturb mode x REPS reps on a saved checkpoint.
# rwtight eval episodes are ~1.5h each, so 3 reps <= 4.5h fits the 8h walltime with margin.
# Optional auditor: set GATE=gate|repair to enable the runtime forecast auditor (EU-CRD).
# Params (--export): EXP, OUT, SEED, MODE(clean|anti|shuffle|blend), [REPS=3], [SEEDBASE=4000], [GATE=]
#SBATCH --job-name=em
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
RUNDIR="$OUTBASE/${OUT}_s${SEED}"
SUMMARY="$OUTBASE/eval_full_summary.txt"
REPS="${REPS:-3}"; SEEDBASE="${SEEDBASE:-4000}"; GATE="${GATE:-}"
cd "$PROJECTDIR/rl-cloudsimplus-greenscheduling/drl-manager"
CK=$(ls -dt "$RUNDIR"/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | head -1)
if [ -z "$CK" ]; then echo "[em $OUT s$SEED $MODE] NO CHECKPOINT" >> "$SUMMARY"; exit 0; fi
export EVAL_CHECKPOINT_PATH="$CK"
if [ -n "$GATE" ]; then export TRUST_GATE_MODE="$GATE" TRUST_GATE_SOURCE=resid TRUST_GATE_THRESH=0.0; fi
PMODE="$MODE"; [ "$MODE" = "clean" ] && PMODE="none"
echo "[em] EXP=$EXP OUT=$OUT s$SEED MODE=$MODE GATE=${GATE:-none} CK=$CK node=$(hostname)"
for r in $(seq 1 "$REPS"); do
  FORECAST_PERTURB_MODE="$PMODE" FORECAST_PERTURB_EPS=1.0 DECODE_TOPK=0 \
  python -m src.baselines.evaluate --experiment "$EXP" \
    --global rllib --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 1 --seed $((SEEDBASE+r)) --output "/tmp/em_${OUT}_s${SEED}_${MODE}${GATE}$r.csv" \
    > "$RUNDIR/em_${MODE}${GATE}$r.log" 2>&1
  ac=$(grep -a "Avg Carbon Emission" "$RUNDIR/em_${MODE}${GATE}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
  af=$(grep -a "Avg Finished" "$RUNDIR/em_${MODE}${GATE}$r.log" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
  echo "[em $OUT s$SEED ${MODE}${GATE:+-$GATE}-r$r] carbon=${ac:-?} completion=${af:-?}" >> "$SUMMARY"
done
echo "[em $OUT s$SEED ${MODE}${GATE:+-$GATE} DONE]" >> "$SUMMARY"
