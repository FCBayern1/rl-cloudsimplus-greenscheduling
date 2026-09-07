#!/bin/bash
# RL_V2 deterministic-deployment readings (reports/RL_V2_DET_PREREG.md): the four frozen 56k
# checkpoints, deterministic decode (argmax, index ties), six reading windows, every tier of the
# line's channel, then the judge. No training, no reference recomputation, no selection.
set -u
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
PY=$PWD/.venv/bin/python; G1=$PWD/../g1/compressed_timecap_s2
EVALCFG=$G1/config_rl_v2_eval.yml; OUT=$G1/stage_a_out/rl_v2
S=20260907
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export PLANNER_EXPECTED_CAP="640;512;640;512;192" PLANNER_STATIC_TOTAL_W=0 OFFSET_GRID_DENSE=1
export ORACLE_WIND_DIR=$PWD/../cloudsimplus-gateway/src/main/resources/windProduction/simplified
export EVAL_CONFIG_PATH=$EVALCFG
log(){ echo "[$(date '+%F %T')] $*"; }
mkdir -p $OUT/last_det
READ=($($PY -c "import json; print(' '.join(str(o) for o in json.load(open('$OUT/manifest.json'))['windows']['read']))"))
lastck(){ ls -d logs/rl_v2/$1_s$S/*/*/checkpoint_0* 2>/dev/null | sort -V | tail -1; }
chan(){ case $1 in NV|NE) echo none;; *) echo full;; esac; }
# the frozen hashes must still match: the readings are taken on the checkpoints of the record
$PY - <<'EOF' || { log "ABORT: checkpoint hashes changed"; exit 1; }
import hashlib, json, os, glob, sys
S = "20260907"; rec = json.load(open("../g1/compressed_timecap_s2/stage_a_out/rl_v2/checkpoint_hashes.json"))
def dhash(d):
    h = hashlib.sha256()
    for p in sorted(glob.glob(os.path.join(d, "**", "*"), recursive=True)):
        if os.path.isfile(p):
            h.update(os.path.relpath(p, d).encode()); h.update(open(p, "rb").read())
    return h.hexdigest()[:16]
bad = [L for L, v in rec.items() if dhash(v["last_path"]) != v["last_sha256"]]
print("checkpoint hash check:", "ok" if not bad else f"CHANGED {bad}")
sys.exit(1 if bad else 0)
EOF
evalone(){ L=$1; CK=$2; TIER=$3; I=$4; KK=$((12+I))
  C=rl2e_$(chan $L)_$TIER; OUTCSV=$OUT/last_det/${L}_${TIER}_k$KK.csv
  [ -s $OUTCSV ] && return
  EVAL_DECISION_DUMP=${OUTCSV%.csv}_decisions.csv ORACLE_OFFSET_ROWS=${READ[$I]} ORACLE_EXPERIMENT=$C timeout 3600 $PY -m src.baselines.evaluate \
    --experiment $C --global rllib --new-api --checkpoint $CK --local drain --episodes 1 --seed 42 --reset-skip $I \
    --output $OUTCSV > ${OUTCSV%.csv}.log 2>&1 || log "eval FAILED $L $TIER k$KK"; }
for L in NV V NE E; do
  CK=$(lastck $L); [ -n "$CK" ] || { log "ABORT: no last checkpoint for $L"; exit 1; }
  if [ $L = NV ] || [ $L = NE ]; then TIERS="godeye"; else TIERS="godeye shrink75 shrink50 shrink25 shrink0 shuffle anti"; fi
  log "readings $L (deterministic) tiers=[$TIERS]"
  for TIER in $TIERS; do for I in 0 1 2 3 4 5; do evalone $L $CK $TIER $I; done; done
done
log "crd statistics"
for L in NE E; do grep -oE "(delta_r|delta_r_mean|rho_mean|rho_min|responsibility_gate|crd_gate)[^,]*" logs/rl_v2/${L}_s$S.log 2>/dev/null | tail -12 > $OUT/crd_stats_$L.txt; done
log "judge (deterministic)"; ( cd $G1 && RL_V2_LAST_DIR=last_det $PY rl_v2_judge.py all 2>&1 | tail -90 )
D=/home/joshua/rl-cloudsimplus-greenscheduling/reports/manifests/rl_v2/det; mkdir -p $D/last_det
cp $OUT/det_verdict.json $OUT/checkpoint_hashes.json $OUT/crd_stats_*.txt $D/ 2>/dev/null
cp $OUT/last_det/*.csv $D/last_det/ 2>/dev/null; rm -f $D/last_det/*_decisions.csv
log "det readings finished"
