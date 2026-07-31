#!/bin/bash
# Local ablation chain v2 — fixes the four failures of v1:
#  (1) eval ran in the default SWF env: EVAL_CONFIG_PATH was not exported -> now exported;
#  (2) eval used greedy argmax: --stochastic added (matches eval_stoch.sbatch protocol);
#  (3) OOM cascade: orphaned Java gateways accumulated across runs (77 leaked, 58/60GB)
#      -> cleanup() reaps gateways/ray between runs, using [b]racket patterns so pkill
#      does not match its own command line;
#  (4) resume skipped eval together with training -> now training is skipped when
#      checkpoint_000010 exists but eval always (re)runs.
# Eval protocol matches eval_stoch.sbatch: clean + anti x3 + shuffle x2 + blend x2.
set -uo pipefail
REPO="/home/joshua/rl-cloudsimplus-greenscheduling"
PY="$REPO/drl-manager/.venv/bin/python"
CFG="$REPO/config_C.yml"
export EVAL_CONFIG_PATH="$CFG"
SUMMARY="$REPO/drl-manager/logs/cregime_local_ablation_summary.txt"
WORKERS=6; STEPS=600000
declare -A EXP=(
  [nocalib]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4_nocalib
  [noquar]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4_noquar
  [staticc]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4_staticc
)
cd "$REPO/drl-manager"

cleanup () {  # reap orphaned gateways + ray between runs (OOM guard)
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null
  pkill -9 -f "gradle-wrapper[.]jar" 2>/dev/null
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null
  pkill -9 -f "rayle[t]" 2>/dev/null
  sleep 5
}

grab () { local cc cf
  cc=$(grep -a "Avg Carbon Emission" "$1" | grep -av Calling | tail -1 | grep -oE "[0-9.]+" | head -1)
  cf=$(grep -a "Avg Finished"        "$1" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
  echo "${cc:-?} ${cf:-?}"; }

run_one () {  # $1=arm $2=seed
  local arm="$1" seed="$2" exp="${EXP[$1]}" rundir="$REPO/drl-manager/logs/abl_${1}_s${2}"
  local ck10=$(ls -d "$rundir"/multidc_gtrxl_training/PPO_*/checkpoint_000010 2>/dev/null | head -1)
  if [ -z "$ck10" ]; then
    echo "[chain $(date +%H:%M:%S)] TRAIN $arm s$seed" | tee -a "$SUMMARY"
    cleanup
    "$PY" entrypoint_rlmodule_gtrxl.py --config "$CFG" --experiment "$exp" \
      --total-timesteps "$STEPS" --num-workers "$WORKERS" --seed "$seed" \
      --output-dir "$rundir" --no-wandb > "$rundir.train.log" 2>&1
    cleanup
    ck10=$(ls -d "$rundir"/multidc_gtrxl_training/PPO_*/checkpoint_000010 2>/dev/null | head -1)
    if [ -z "$ck10" ]; then echo "[abl $arm s$seed] TRAIN FAILED (no ckpt_10)" | tee -a "$SUMMARY"; return; fi
  else
    echo "[chain $(date +%H:%M:%S)] $arm s$seed already trained (ckpt_10), eval only" | tee -a "$SUMMARY"
  fi
  mkdir -p "$rundir/eval"
  ev () {  # $1=mode $2=seed $3=tag
    local mode="$1" sd="$2" tag="$3"; local lg="$rundir/eval/${tag}.log"
    FORECAST_PERTURB_MODE="$mode" FORECAST_PERTURB_EPS=1.0 \
    "$PY" -m src.baselines.evaluate --experiment "$exp" \
      --global rllib --local rllib --checkpoint "$ck10" --new-api --shared-local --global-defer \
      --stochastic --episodes 1 --seed "$sd" --output "/tmp/abl_${arm}_s${seed}_${tag}.csv" > "$lg" 2>&1
    local cc cf; read cc cf < <(grab "$lg")
    echo "[abl $exp s$seed ${tag}] carbon=$cc completion=$cf" | tee -a "$SUMMARY"
  }
  ev none "$seed" clean
  for r in 1 2 3; do ev anti $((3000+r)) anti-r$r; done
  for r in 1 2;   do ev shuffle $((3100+r)) shuffle-r$r; ev blend $((3200+r)) blend-r$r; done
  cleanup
  echo "[chain $(date +%H:%M:%S)] DONE $arm s$seed" | tee -a "$SUMMARY"
}

echo "[chain $(date +%H:%M:%S)] v2 start (config-correct + stochastic + gateway reaping)" | tee -a "$SUMMARY"
for seed in 1 2 3; do
  for arm in nocalib noquar staticc; do
    run_one "$arm" "$seed"
  done
done
echo "[chain $(date +%H:%M:%S)] ALL ABLATION ARMS COMPLETE" | tee -a "$SUMMARY"
