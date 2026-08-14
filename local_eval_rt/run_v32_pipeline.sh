#!/bin/bash
# Unified pipeline (2026-08-14 evening): interleaves the V3.2 gate ladder into
# the reference wave's training gaps, so Gate 2 lands TONIGHT instead of
# tomorrow noon. Replaces run_v31_fullwave.sh's remaining stages (orchestrator
# killed; its nofc_s1 child keeps running and step 1 waits for it).
#
# Jar policy: installDist happens BETWEEN the s1 and s2 pairs. Both arms of a
# pair always share one jar version; V3.2 Java changes are config-gated with
# bit-exact regression locks (65 tests), so v3_1 configs behave identically on
# either jar. Recorded here for the provenance trail.
#
# Reference-wave evals are FINAL-CK-ONLY (4 cells): the wave is a reference,
# not a certification, and the main-arm precedent (c0d52bf) applies.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v32_pipeline.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
echo "===== V32 PIPELINE $(date '+%m-%d %H:%M') =====" >>"$OUT"

# 1. wait for the orphaned nofc_s1 training
while pgrep -f "entrypoint_rlmodule" >/dev/null 2>&1; do sleep 60; done
echo "[pipe] nofc_s1 finished, machine clear $(date '+%H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5

# 2. jar refresh + truth table with real artifact values (hard gates)
cd $REPO/cloudsimplus-gateway
./gradlew installDist -q >>"$OUT" 2>&1 || { echo "[pipe] installDist FAILED" >>"$OUT"; exit 1; }
./gradlew test --tests "exe.edu.cspg.multidc.PerActionRewardSurgeryTest" -q \
  -Dv31.mu=3.524 -Dv31.sigma=2.512 -Dv31.margGreen=0.71 -Dv31.margBrown=2.34 >>"$OUT" 2>&1 \
  || { echo "[pipe] truth-table FAILED" >>"$OUT"; exit 1; }
echo "[pipe] jars refreshed, truth table green $(date '+%H:%M')" >>"$OUT"
cd $REPO/drl-manager

train () {  # exp seed steps outdir
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  echo "[pipe] train $4 start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
    --experiment "$1" --total-timesteps $3 --num-workers 6 --seed $2 \
    --output-dir logs/$4 > $R/$4_train.log 2>&1
  echo "[pipe] train $4 exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
}
probe () {  # outdir jsonname
  local CK=$(ls -d logs/$1/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$CK" ] && { echo "[pipe] $1 no checkpoint" >>"$OUT"; return 1; }
  echo "----- probe $1 $(basename $CK) -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
    --raw-logits --json-out $R/probe/$2.json 2>>"$OUT" | grep -E "difference|fraction|raw defer|route logits|ratio" >>"$OUT"
}
delta () { .venv/bin/python -c "import json;d=json.load(open('$R/probe/$1.json'));print(d['temporal']['delta'])" 2>/dev/null; }

# 3. V3.2 Gate 2 smoke (oracle s1 100k) -- prereg threshold +0.05, frozen
train experiment_v3_2_oracle 1 100000 v32_smoke_s1
probe v32_smoke_s1 v32_smoke_s1
D=$(delta v32_smoke_s1)
G2=FAIL
if [ -n "$D" ] && .venv/bin/python -c "exit(0 if float('$D')>=0.05 else 1)"; then G2=PASS; fi
echo "V32 GATE2 $G2 (delta=$D, threshold=+0.05) $(date '+%H:%M')" >>"$OUT"

# 4. reference wave s2 pair (600k each, new jars, pair-consistent)
train experiment_v3_1_oracle    2 600000 v31_oracle_s2
train experiment_v3_1_noforecast 2 600000 v31_nofc_s2
echo "[pipe] reference wave all trained $(date '+%m-%d %H:%M')" >>"$OUT"

# 5. V3.2 Gate 3 (only if Gate 2 passed): 300k both seeds
if [ "$G2" = "PASS" ]; then
  train experiment_v3_2_oracle 1 300000 v32_g3_s1
  probe v32_g3_s1 v32_g3_s1
  train experiment_v3_2_oracle 2 300000 v32_g3_s2
  probe v32_g3_s2 v32_g3_s2
  D1=$(delta v32_g3_s1); D2=$(delta v32_g3_s2)
  if [ -n "$D1" ] && [ -n "$D2" ] && \
     .venv/bin/python -c "exit(0 if float('$D1')>0 and float('$D2')>0 else 1)"; then
    echo "V32 GATE3 sign-check PASS (s1=$D1 s2=$D2) -> 600k approved pending TD-residual review" >>"$OUT"
  else
    echo "V32 GATE3 FAIL (s1=${D1:-?} s2=${D2:-?}) -> per prereg: check impl / V3.2B" >>"$OUT"
  fi
else
  echo "[pipe] Gate2 FAIL -> skipping Gate3 per prereg (no weight tuning, no extension)" >>"$OUT"
fi

# 6. reference-wave probes (sign curve on oracle arms) + final-ck evals
for A in v31_oracle_s1 v31_oracle_s2; do
  for CK in $(ls -d logs/$A/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V); do
    ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
    echo "----- waveprobe $A $ckn -----" >>"$OUT"
    .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
      --json-out $R/probe/wave_${A}_${ckn}.json 2>>"$OUT" | grep -E "difference|fraction" >>"$OUT"
  done
done
declare -A EXPMAP
EXPMAP[v31_oracle_s1]=experiment_v3_1_oracle;    EXPMAP[v31_oracle_s2]=experiment_v3_1_oracle
EXPMAP[v31_nofc_s1]=experiment_v3_1_noforecast;  EXPMAP[v31_nofc_s2]=experiment_v3_1_noforecast
for A in v31_oracle_s1 v31_nofc_s1 v31_oracle_s2 v31_nofc_s2; do
  S=${A##*_s}
  CK=$(ls -d logs/$A/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$CK" ] && { echo "[pipe] $A missing for eval" >>"$OUT"; continue; }
  ckn=$(basename $CK | sed 's/checkpoint_0*/ck/')
  lg=$R/waveeval_${A}.log
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  timeout 9000 env FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "${EXPMAP[$A]}" --global rllib \
    --local drain --checkpoint "$CK" --new-api --shared-local --global-defer \
    --episodes 10 --seed $S --output $R/eval_csv/waveeval_${A}.csv >"$lg" 2>&1
  if [ $? -eq 124 ]; then echo "[waveeval $A $ckn] TIMEOUT" >>"$OUT"; else
    cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[waveeval $A $ckn@DRAIN] cc=${cc:-?} completion=${cf:-?}" >>"$OUT"
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
done
echo "V32 PIPELINE DONE $(date '+%m-%d %H:%M')" >>"$OUT"
