#!/bin/bash
# Stage D health smoke, evaluation part only (checkpoint discovery fixed: Tune trial dir is
# one level deeper; first = checkpoint_000000, last = highest index = checkpoint_at_end).
set -u
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
PY=.venv/bin/python; T=/home/joshua/.claude/jobs/f676ac21/tmp; S=20260903
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export EVAL_CONFIG_PATH=$PWD/../g1/compressed_timecap_s2/config_stage_d_eval.yml
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0
log(){ echo "[$(date '+%F %T')] $*"; }
lastck(){ find logs/stage_d/$1_s$S -type d -name 'checkpoint_*' | sort -V | tail -1; }
firstck(){ find logs/stage_d/$1_s$S -type d -name 'checkpoint_*' | sort -V | head -1; }
CELLS="s2_r48_w72_c1_n20 s2_r48_w72_c1_n50 s2_r48_w72_c3_n20 s2_r48_w72_c3_n50 s2_r48_w72_c5_n20 s2_r48_w72_c5_n50"
evalone(){ L=$1; CK=$2; C=$3; TIER=$4; K=$5; TAG=$6
  OUT=results/stage_d/${L}_${TAG}; mkdir -p $OUT
  [ -s $OUT/${C}_${TIER}_k${K}.csv ] && return
  timeout 3600 .venv/bin/python -m src.baselines.evaluate --experiment sde_${C}_${TIER} --global rllib --new-api --stochastic \
    --checkpoint $CK --local drain --episodes 1 --seed 42 --reset-skip $K \
    --output $OUT/${C}_${TIER}_k${K}.csv > $OUT/${C}_${TIER}_k${K}.log 2>&1 || echo "[$(date '+%F %T')] eval FAILED $L $TAG $C $TIER k$K"; }
export -f evalone
: > $T/stage_d_eval_jobs.txt
for L in NV V NE E; do
  CK=$(lastck $L); CK1=$(firstck $L); log "$L first=$CK1 last=$CK"
  [ -z "$CK" ] && continue
  if [ $L = NV ] || [ $L = NE ]; then TIERS="hollow"; T1=hollow; else TIERS="godeye calibrated_shrink_v1 shuffle anti"; T1=godeye; fi
  for C in $CELLS; do for TIER in $TIERS; do for K in 26 34 42; do echo "$L $CK $C $TIER $K last"; done; done; done >> $T/stage_d_eval_jobs.txt
  for C in $CELLS; do for K in 26 34 42; do echo "$L $CK1 $C $T1 $K first"; done; done >> $T/stage_d_eval_jobs.txt
done
log "eval jobs: $(wc -l < $T/stage_d_eval_jobs.txt)"
xargs -P 4 -L 1 bash -c 'evalone $0 $1 $2 $3 $4 $5' < $T/stage_d_eval_jobs.txt
log "evals done: $(ls results/stage_d/*/*.csv 2>/dev/null | wc -l) csv, failed: $(grep -c FAILED $T/stage_d_eval.log 2>/dev/null)"
log "eval chain done"
