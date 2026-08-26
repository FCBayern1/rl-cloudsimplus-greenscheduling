#!/bin/bash
# G1 futility-gate evaluation. Every constant here is frozen in
# reports/G1_FREEZE_MANIFEST.md and none of it is chosen after seeing a result.
#
#   8 checkpoints (4 seeds x 2 arms, always checkpoint_000010 at 600k)
#   x 3 conditions (clean / blend / shuffle)
#   x 3 registered windows (low k=19, mid k=56, high k=34)
#   x 1 episode per cell            <- NOT three: --episodes 3 advances to k+1/k+2
#   = 72 cells, run three-way concurrent
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
PY=$REPO/drl-manager/.venv/bin/python
OUT=$REPO/g1/eval
LOG=$OUT/eval.log

export GATEWAY_LIBS=/home/joshua/frozen/g1_gateway/lib
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

want=aba6f0edf473871406e96e9a4f1f2375b5976f3f9be67c4ee3d5fb665962498e
got=$(sha256sum "$GATEWAY_LIBS/cloudsimplus-gateway.jar" | cut -d' ' -f1)
[ "$got" = "$want" ] || { echo "ABORT: jar sha mismatch $got"; exit 1; }
[ -w "$GATEWAY_LIBS/cloudsimplus-gateway.jar" ] && { echo "ABORT: frozen jar writable"; exit 1; }

mkdir -p $OUT/csv $OUT/logs
say(){ echo "[$(date '+%m-%d %H:%M')] $*" | tee -a "$LOG"; }
say "frozen jar verified $got"
say "===== G1 EVAL START (72 cells, 3-way concurrent) ====="

declare -A EXPKEY=( [eucrd]=experiment_g1eval_knSV3b [van]=experiment_g1eval_matchedvan )
declare -A WIN=( [low]=19 [mid]=56 [high]=34 )
CONDS="none blend shuffle"

ck(){ ls -d $REPO/drl-manager/logs/g1_$1_s$2/multidc_gtrxl_training/PPO_*/checkpoint_000010 2>/dev/null | head -1; }

run_cell(){  # arm seed cond window
  local arm=$1 seed=$2 cond=$3 win=$4 k=${WIN[$4]}
  local tag=${arm}_s${seed}_${cond}_${win}
  local csv=$OUT/csv/${tag}.csv
  [ -s "$csv" ] && { say "  skip $tag (exists)"; return; }
  local c=$(ck $arm $seed)
  [ -n "$c" ] || { say "  MISSING checkpoint $arm s$seed"; return; }
  # Do not build the env prefix with ${x:+...}: bash decides which words are
  # assignments at parse time, so an empty conditional expansion makes the next
  # word the command name. Export inside the subshell instead.
  ( cd $REPO/drl-manager || exit 1
    export FORECAST_PERTURB_MODE=$cond DECODE_TOPK=0
    [ "$cond" = "blend" ] && export FORECAST_PERTURB_EPS=1.0
    unset FORECAST_PERTURB_PROB
    nice -n 5 $PY -m src.baselines.evaluate --experiment "${EXPKEY[$arm]}" \
      --global rllib --local rllib --checkpoint "$c" --new-api --shared-local \
      --global-defer --episodes 1 --seed 20260823 --reset-skip $k \
      --output "$csv" ) > $OUT/logs/${tag}.log 2>&1
  local dc=$(grep -a -oE "Environment: [0-9]+ DCs" $OUT/logs/${tag}.log | head -1)
  say "  done $tag  [$dc]  carbon=$(grep -a 'Avg Carbon/MI' $OUT/logs/${tag}.log|tail -1|grep -oE '[0-9.]+'|head -1) comp=$(grep -a 'Avg Finished' $OUT/logs/${tag}.log|grep -av Calling|tail -1|grep -oE '[0-9.]+%'|head -1)"
}

n=0
for seed in 101 102 103 104; do
  for arm in van eucrd; do
    for cond in $CONDS; do
      for win in low mid high; do
        run_cell $arm $seed $cond $win &
        n=$((n+1)); (( n % 3 == 0 )) && wait
      done
    done
  done
done
wait
say "===== G1 EVAL DONE ($n cells) ====="
