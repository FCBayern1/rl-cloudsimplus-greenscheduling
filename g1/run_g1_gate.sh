#!/bin/bash
# G1 futility gate: four pairs, seeds 101-104. Fixed stop rule, not a peek.
# Everything this reads is pinned in reports/G1_FREEZE_MANIFEST.md.
#
# Arm order alternates by seed parity so arm is not confounded with seed:
#   odd  seed -> Vanilla first
#   even seed -> EU-CRD first
# Both members of a pair run back to back on this machine, same jar, same commit.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
PY=$REPO/drl-manager/.venv/bin/python
LOG=$REPO/g1/gate.log

export GATEWAY_LIBS=/home/joshua/frozen/g1_gateway/lib
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

EU=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b
VAN=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_matchedvan

say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }

# Refuse to run against anything but the frozen jar.
want=aba6f0edf473871406e96e9a4f1f2375b5976f3f9be67c4ee3d5fb665962498e
got=$(sha256sum "$GATEWAY_LIBS/cloudsimplus-gateway.jar" | cut -d' ' -f1)
[ "$got" = "$want" ] || { say "ABORT: gateway jar sha mismatch $got"; exit 1; }
[ -w "$GATEWAY_LIBS/cloudsimplus-gateway.jar" ] && { say "ABORT: frozen jar is writable"; exit 1; }
say "frozen jar verified $got"

train(){ # arm_tag experiment seed
  local tag=$1 exp=$2 seed=$3
  local dir=$REPO/drl-manager/logs/g1_${tag}_s${seed}
  if [ -d "$dir/multidc_gtrxl_training" ]; then say "skip g1_${tag}_s${seed} (exists)"; return 0; fi
  say "START g1_${tag}_s${seed}"
  local t0=$(date +%s)
  (cd $REPO/drl-manager && $PY entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
    --experiment "$exp" --total-timesteps 600000 --num-workers 6 --seed "$seed" \
    --output-dir "$dir" --no-wandb) > $REPO/g1/g1_${tag}_s${seed}.log 2>&1
  local rc=$?
  local ck=$(ls -d $dir/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | wc -l)
  say "DONE  g1_${tag}_s${seed}  rc=$rc  ckpts=$ck  $(( ($(date +%s)-t0)/60 )) min"
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null
  sleep 10
}

say "===== G1 FUTILITY GATE START (4 pairs, seeds 101-104) ====="
for seed in 101 102 103 104; do
  if (( seed % 2 == 1 )); then
    train van "$VAN" $seed;  train eucrd "$EU" $seed
  else
    train eucrd "$EU" $seed; train van "$VAN" $seed
  fi
done
say "===== G1 FUTILITY GATE DONE ====="
