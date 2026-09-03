#!/bin/bash
# Stage D health smoke on the local RTX 5080: four lines, one seed, 56k steps each,
# two lines in parallel, then deployment evaluation of every line's last checkpoint on
# the six HZ cells x certified windows k=26/34/42. HEALTH_SMOKE: no effect claim.
set -u
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
PY=.venv/bin/python; T=/home/joshua/.claude/jobs/f676ac21/tmp
CFG=../g1/compressed_timecap_s2/config_stage_d_physical.yml
EVALCFG=../g1/compressed_timecap_s2/config_stage_d_eval.yml
S=20260903; STEPS=56000
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0
log(){ echo "[$(date '+%F %T')] $*"; }
mkdir -p logs/stage_d results/stage_d
train(){ L=$1; $PY entrypoint_rlmodule_gtrxl.py --config $CFG --experiment sd_${L}_s2_r48_w72_c3_n35 \
  --total-timesteps $STEPS --num-workers 0 --seed $S --no-wandb --output-dir logs/stage_d/${L}_s$S \
  > logs/stage_d/${L}_s$S.log 2>&1; log "train $L exit=$?"; }
log "training NV + V"; train NV & train V & wait
log "training NE + E"; train NE & train E & wait
lastck(){ ls -d logs/stage_d/$1_s$S/*/checkpoint_* 2>/dev/null | sort -V | tail -1; }
firstck(){ ls -d logs/stage_d/$1_s$S/*/checkpoint_* 2>/dev/null | sort -V | head -1; }
for L in NV V NE E; do log "$L checkpoints: first=$(firstck $L) last=$(lastck $L) n=$(ls -d logs/stage_d/${L}_s$S/*/checkpoint_* 2>/dev/null | wc -l)"; done
CELLS="s2_r48_w72_c1_n20 s2_r48_w72_c1_n50 s2_r48_w72_c3_n20 s2_r48_w72_c3_n50 s2_r48_w72_c5_n20 s2_r48_w72_c5_n50"
export EVAL_CONFIG_PATH=$PWD/$EVALCFG
evalone(){ L=$1; CK=$2; C=$3; TIER=$4; K=$5; TAG=$6
  OUT=results/stage_d/${L}_${TAG}; mkdir -p $OUT
  [ -s $OUT/${C}_${TIER}_k${K}.csv ] && return
  timeout 3600 $PY -m src.baselines.evaluate --experiment sde_${C}_${TIER} --global rllib --new-api --stochastic \
    --checkpoint $CK --local drain --episodes 1 --seed 42 --reset-skip $K \
    --output $OUT/${C}_${TIER}_k${K}.csv > $OUT/${C}_${TIER}_k${K}.log 2>&1 || log "eval FAILED $L $TAG $C $TIER k$K"; }
export -f evalone log; export PY OUT S
for L in NV V NE E; do
  CK=$(lastck $L); [ -z "$CK" ] && { log "no checkpoint for $L, skipping eval"; continue; }
  if [ $L = NV ] || [ $L = NE ]; then TIERS="hollow"; else TIERS="godeye calibrated_shrink_v1 shuffle anti"; fi
  log "eval $L last=$CK tiers=[$TIERS]"
  for C in $CELLS; do for TIER in $TIERS; do for K in 26 34 42; do echo "$L $CK $C $TIER $K last"; done; done; done
  CK1=$(firstck $L); T1=$([ $L = NV ] || [ $L = NE ] && echo hollow || echo godeye)
  for C in $CELLS; do for K in 26 34 42; do echo "$L $CK1 $C $T1 $K first"; done; done
done > $T/stage_d_eval_jobs.txt
log "eval jobs: $(wc -l < $T/stage_d_eval_jobs.txt)"
xargs -P 4 -L 1 bash -c 'evalone $0 $1 $2 $3 $4 $5' < $T/stage_d_eval_jobs.txt
log "evals done: $(ls results/stage_d/*/*.csv 2>/dev/null | wc -l) csv"
log "smoke chain done"
