#!/bin/bash
# Matched-pair deployment (reports/EUCRD_MATCHED_PAIR_PREREG.md §3): deterministic decode, six
# development windows, eight tiers, both lines of every seed plus the cover_argmax reference.
# Decisions are dumped so action divergence can be read without re-running (Addendum D).
# Usage: matched_pair_eval.sh <checkpoint_root>   (dirs <root>/{V,E}_s<seed>/*/PPO_*/checkpoint_*)
set -u
ROOT=${1:?need the checkpoint root}
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
PY=$PWD/.venv/bin/python; G1=$PWD/../g1/compressed_timecap_s2
OUT=$G1/stage_a_out/matched_pair; mkdir -p $OUT/eval
EVALCFG=$G1/config_rl_v2_eval.yml; export EVAL_CONFIG_PATH=$EVALCFG
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0 OFFSET_GRID_DENSE=1
export ORACLE_WIND_DIR=$PWD/../cloudsimplus-gateway/src/main/resources/windProduction/simplified
export COVER_TIE=index
log(){ echo "[$(date '+%F %T')] $*"; }
TIERS="godeye shrink75 shrink50 shrink25 shrink0 shuffle anti calibrated_shrink_v1"
SEEDS="20260911 20260912 20260913"
READ=($($PY -c "import json; print(' '.join(str(o) for o in json.load(open('$G1/stage_a_out/rl_v2/manifest.json'))['windows']['read']))"))

ck(){ ls -d $ROOT/$1/*/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1; }
one(){ # arm cell tier window [extra env]
  local ARM=$1 CELL=$2 TIER=$3 I=$4 CKD=${5:-}
  local CSV=$OUT/eval/${ARM}_${TIER}_k$I.csv
  [ -s "$CSV" ] && return 0
  local ARGS=(--experiment $CELL --local drain --episodes 1 --seed 42 --reset-skip $I --output $CSV)
  if [ -n "$CKD" ]; then ARGS+=(--global rllib --new-api --checkpoint "$CKD"); else ARGS+=(--global cover_argmax); fi
  EVAL_DECISION_DUMP=${CSV%.csv}_decisions.csv ORACLE_OFFSET_ROWS=${READ[$I]} ORACLE_EXPERIMENT=$CELL \
    timeout 3600 $PY -m src.baselines.evaluate "${ARGS[@]}" > ${CSV%.csv}.log 2>&1 \
    || log "FAILED $ARM $TIER k$I"
}
# every line of every seed, then the zero-parameter rule
for S in $SEEDS; do for L in V E; do
  CKD=$(ck ${L}_s$S)
  [ -n "$CKD" ] || { log "ABORT: no checkpoint for ${L}_s$S under $ROOT"; exit 1; }
  log "${L}_s$S <- $CKD"
  for TIER in $TIERS; do for I in 0 1 2 3 4 5; do one ${L}_s$S rl2e_full_$TIER $TIER $I "$CKD"; done; done
  log "evaluated ${L}_s$S"
done; done
for TIER in $TIERS; do for I in 0 1 2 3 4 5; do one cover_argmax rl2e_full_$TIER $TIER $I; done; done
log "evaluated cover_argmax"
( cd $G1 && $PY matched_pair_judge.py )
log "matched-pair evaluation finished"
