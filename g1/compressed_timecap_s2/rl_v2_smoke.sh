#!/bin/bash
# RL_V2 health smoke (reports/RL_V2_SMOKE_PREREG.md, frozen 96256505 + Addendum A): references,
# four trainings ONE AT A TIME with fail-fast, init checks (deterministic identity and sampled
# prior), last-checkpoint readings, judge. Run 2 after INVALID_SMOKE_RUN1_OOM.
set -u
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
PY=.venv/bin/python; G1=../g1/compressed_timecap_s2; T=/home/joshua/.claude/jobs/f676ac21/tmp
CFG=$G1/config_rl_v2.yml; EVALCFG=$G1/config_rl_v2_eval.yml; OUT=$G1/stage_a_out/rl_v2
S=20260907; STEPS=56000
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0 OFFSET_GRID_DENSE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export ORACLE_WIND_DIR=$PWD/../cloudsimplus-gateway/src/main/resources/windProduction/simplified
log(){ echo "[$(date '+%F %T')] $*"; }
mkdir -p logs/rl_v2 $OUT/init $OUT/last $OUT/ref $OUT/flat
READ=($($PY -c "import json; print(' '.join(str(o) for o in json.load(open('$OUT/manifest.json'))['windows']['read']))"))
log "references (flat on validation windows, cover_argmax on every tier)"
( cd $G1 && $PY rl_v2_refs.py all 2>&1 | grep -v "WARNING\|Missing columns\|SWF" | grep -E "^cover|^flat|Traceback|Error" ) &
REFPID=$!
abort(){ log "ABORT: $1"; echo "$1" > $OUT/INVALID_RUN.txt; kill $REFPID 2>/dev/null; exit 1; }
train(){ L=$1; $PY entrypoint_rlmodule_gtrxl.py --config $CFG --experiment rl2_${L}_s2_r48_w72_c3_n35 \
  --total-timesteps $STEPS --num-workers 0 --seed $S --no-wandb --output-dir logs/rl_v2/${L}_s$S \
  > logs/rl_v2/${L}_s$S.log 2>&1; rc=$?; log "train $L exit=$rc"
  # Addendum A1: fail-fast. A non-zero code, or a missing last checkpoint, aborts the whole smoke.
  [ $rc -eq 0 ] || abort "INVALID_SMOKE_RUN2_TRAIN_${L}_EXIT_${rc}$(grep -qi 'OutOfMemoryError' logs/rl_v2/${L}_s$S.log && echo _OOM)"
  [ -n "$(lastck $L)" ] || abort "INVALID_SMOKE_RUN2_NO_CHECKPOINT_${L}"; }
lastck(){ ls -d logs/rl_v2/$1_s$S/*/checkpoint_* 2>/dev/null | grep -v checkpoint_init | sort -V | tail -1; }
initck(){ ls -d logs/rl_v2/$1_s$S/*/checkpoint_init 2>/dev/null | head -1; }
# One line at a time: the candidate key makes a training batch tensor of 8000 x 128 x 365 x 4 B
# = 1.49 GiB, and two learners do not fit in the 16 GB GPU together (run 1: CUDA OOM on the first
# learner update, Addendum A). The first line is also the memory preflight.
for L in NV V NE E; do log "training $L"; train $L; done
wait $REFPID; log "references done"
for L in NV V NE E; do log "$L init=$(initck $L) last=$(lastck $L) n=$(ls -d logs/rl_v2/${L}_s$S/*/checkpoint_* 2>/dev/null | wc -l)"; done
export EVAL_CONFIG_PATH=$PWD/$EVALCFG
chan(){ case $1 in NV|NE) echo none;; *) echo full;; esac; }
evalone(){ L=$1; CK=$2; TIER=$3; I=$4; KK=$((12+I)); MODE=$5; OUTD=$6
  C=rl2e_$(chan $L)_$TIER; OUTCSV=$OUTD/${L}_${TIER}_k$KK.csv
  case $MODE in init|initstoch) OUTCSV=$OUTD/${L}_k$KK.csv;; esac
  [ -s $OUTCSV ] && return
  STOCH=--stochastic; [ "$MODE" = init ] && STOCH=""
  OBS=0; [ "$MODE" = initstoch ] && OBS=1
  EVAL_DECISION_DUMP=${OUTCSV%.csv}_decisions.csv EVAL_DECISION_DUMP_OBS=$OBS ORACLE_OFFSET_ROWS=${READ[$I]} ORACLE_EXPERIMENT=$C timeout 3600 $PY -m src.baselines.evaluate \
    --experiment $C --global rllib --new-api $STOCH --checkpoint $CK --local drain --episodes 1 --seed 42 --reset-skip $I \
    --output $OUTCSV > ${OUTCSV%.csv}.log 2>&1 || log "eval FAILED $L $TIER k$KK $MODE"; }
log "init checks (deterministic identity, then the sampled prior)"
mkdir -p $OUT/init_stoch
for L in NV V NE E; do CK=$(initck $L)
  [ -n "$CK" ] || abort "INVALID_SMOKE_RUN2_NO_INIT_CHECKPOINT_${L}"
  for I in 0 1 2 3 4 5; do evalone $L $CK godeye $I init $OUT/init; done
  for I in 0 1 2 3 4 5; do evalone $L $CK godeye $I initstoch $OUT/init_stoch; done; done
( cd $G1 && $PY rl_v2_judge.py init 2>&1 | tail -8 ); ( cd $G1 && $PY rl_v2_judge.py prior 2>&1 | tail -30 )
V1=$($PY -c "import json; print(json.load(open('$OUT/init_check.json'))['verdict'])" 2>/dev/null)
V2=$($PY -c "import json; print(json.load(open('$OUT/init_prior_check.json'))['verdict'])" 2>/dev/null)
log "init verdicts: $V1 / $V2"
[ "$V1" = "INIT_OK" ] || abort "$V1"
[ "$V2" = "INIT_PRIOR_CARRIED" ] || abort "$V2"
log "last-checkpoint readings"
for L in NV V NE E; do CK=$(lastck $L); [ -z "$CK" ] && { log "no last checkpoint for $L"; continue; }
  if [ $L = NV ] || [ $L = NE ]; then TIERS="godeye"; else TIERS="godeye shrink75 shrink50 shrink25 shrink0 shuffle anti"; fi
  for TIER in $TIERS; do for I in 0 1 2 3 4 5; do evalone $L $CK $TIER $I last $OUT/last; done; done; done
log "crd statistics from the training logs"
for L in NE E; do grep -oE "(delta_r|delta_r_mean|rho_mean|rho_min|responsibility_gate_active|crd_gate)[^,]*" logs/rl_v2/${L}_s$S.log | tail -12 > $OUT/crd_stats_$L.txt; done
log "judge"; ( cd $G1 && $PY rl_v2_judge.py all 2>&1 | tail -80 )
D=/home/joshua/rl-cloudsimplus-greenscheduling/reports/manifests/rl_v2/smoke; mkdir -p $D
cp $OUT/*.json $OUT/crd_stats_*.txt $D/ 2>/dev/null; for sub in init last ref flat; do mkdir -p $D/$sub; cp $OUT/$sub/*.csv $D/$sub/ 2>/dev/null; rm -f $D/$sub/*_decisions.csv; done
log "smoke finished"
