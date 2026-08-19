#!/bin/bash
# Checkpoint qualification for the EU-CRD sweep arm (paper's own rule):
# "each seed contributes the checkpoint with the lowest clean carbon among
# those completing >= 99.5% on clean forecasts". ck10 of creg_eucrd_s2 fails
# it (argmax decode collapse, 13.56% completion) while its training-time
# completion is 1.000 every iteration - the documented 1-in-4 decode fragility.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
B=$REPO/isambard_backup/rl-runs-full
OUT=$REPO/local_eval_rt/eucrd_qualify.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd $REPO/drl-manager
EXP=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4
echo "===== EU-CRD ckpt qualification (clean, ep2, argmax) $(date '+%m-%d %H:%M') =====" >>"$OUT"
try () {  # run_dir ck seed
  local CK=$(find $B/$1 -maxdepth 3 -name "checkpoint_0000$2" -type d | head -1)
  [ -z "$CK" ] && { echo "[qual $1 ck$2] MISSING" >>"$OUT"; return; }
  local lg=$REPO/local_eval_rt/qual_$1_ck$2.log
  DECODE_TOPK=0 nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "$EXP" --global rllib --local rllib --checkpoint "$CK" \
    --new-api --shared-local --global-defer --episodes 2 --seed $3 \
    --output $REPO/local_eval_rt/eval_csv/qual_$1_ck$2.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  echo "[qual $1 ck$2 seed$3] carbon=${cc:-?} completion=${cf:-?}" >>"$OUT"
}
try creg_eucrd_s2 09 2
try creg_eucrd_s1 10 1
try cregime_eucrd_s3 10 3
try cregime_eucrd_s3 09 3
echo "QUALIFY DONE $(date '+%m-%d %H:%M')" >>"$OUT"
