#!/bin/bash
# V3.1 smoke: 100k oracle-arm training + P1 probe. WRITTEN TONIGHT, ARMED ONLY
# AFTER the morning switch decision (V31_PREREG.md §6) - do NOT nohup this
# until the cert configs are locked and preflight --v31-cert passes.
#
# Sequence (each step gates the next):
#   1. refuse to run while anything is on the machine;
#   2. preflight --v31-cert on the cert pair (hard gate);
#   3. gradlew installDist - the reward surgery lives in Java, and the
#      running-queue ban on refreshing jars only lifts when the queue is empty;
#   4. truth-table JUnit with the artifact's real mu/sigma (second hard gate);
#   5. 100k oracle-arm training (checkpoint EVERY iteration - the ck1-ck7
#      retention gap made the sign-acquisition timeline unrecoverable in v3);
#   6. P1 probe on the last checkpoint -> verdict line.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v31_smoke.txt
ORACLE=experiment_v3_1_oracle
BLIND=experiment_v3_1_noforecast
MU=3.524; SIGMA=2.512   # calib/experiment_v3_1_oracle_carbon_norm.json (mid_util)
# Truth-table marg points MUST come from the same artifact as mu/sigma —
# mixing synthetic margs with real mu/sigma is exactly what aborted the
# 09:39 launch (synthetic brown 16.5 sat 5 sigma out on the real scale).
# margGreen = artifact q1 (best green route); margBrown = median job on a
# 0.55-factor trough DC (4.26 kg * 0.55). Note for the record: this brown
# route prices POSITIVE (+0.24) under the pooled z-score - the wait incentive
# for median jobs is thinner than the synthetic gate suggested.
MG=0.71; MB=2.34

# Bounded wait (<=60 min) for the tail of the verdict queue, then hard refuse.
WAITED=0
while pgrep -f "baselines[.]evaluate|entrypoint_rlmodule|oracle_hold_until_green|oracle_slack_planner|run_v3_track0b[.]sh|run_v3_drainfix[.]sh" >/dev/null 2>&1; do
  [ $WAITED -ge 3600 ] && { echo "[smoke] REFUSING: machine still busy after 60min" >>"$OUT"; exit 1; }
  sleep 60; WAITED=$((WAITED+60))
done
echo "[smoke] machine clear after ${WAITED}s wait $(date '+%m-%d %H:%M')" >>"$OUT"
cd $REPO/drl-manager
.venv/bin/python preflight_scenario.py $ORACLE $BLIND --v31-cert >>"$OUT" 2>&1 || { echo "[smoke] preflight FAILED" >>"$OUT"; exit 1; }
cd $REPO/cloudsimplus-gateway
./gradlew installDist -q >>"$OUT" 2>&1 || { echo "[smoke] installDist FAILED" >>"$OUT"; exit 1; }
./gradlew test --tests "exe.edu.cspg.multidc.PerActionRewardSurgeryTest" -q \
  -Dv31.mu=$MU -Dv31.sigma=$SIGMA -Dv31.margGreen=$MG -Dv31.margBrown=$MB >>"$OUT" 2>&1 || { echo "[smoke] truth-table FAILED with real mu/sigma" >>"$OUT"; exit 1; }
echo "[smoke] gates passed, training starts $(date '+%m-%d %H:%M')" >>"$OUT"
cd $REPO/drl-manager

# P1 is pre-registered as BOTH seeds positive. Staged to fail fast: seed 1
# first; only a positive temporal delta buys seed 2's compute. P2/P3 are NOT
# smoke criteria - they belong to the 600k stage (PREREG paragraph 2/3).
smoke_seed () {  # seed -> writes probe json, echoes delta (empty on failure)
  local S="$1"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config $REPO/config_C.yml \
    --experiment $ORACLE --total-timesteps 100000 --num-workers 6 --seed $S \
    --output-dir $REPO/drl-manager/logs/v31_smoke_oracle_s${S} > $R/v31_smoke_train_s${S}.log 2>&1
  echo "[smoke s${S}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
  local CK=$(ls -d $REPO/drl-manager/logs/v31_smoke_oracle_s${S}/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$CK" ] && { echo "[smoke s${S}] no checkpoint" >>"$OUT"; return 1; }
  echo "----- P1 probe s${S} $(basename $CK) -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$CK" --trials 40 \
    --json-out $R/probe/v31_smoke_s${S}.json 2>/dev/null | sed -n '/channel/,$p' >>"$OUT"
}
delta_of () { .venv/bin/python -c "import json;print(json.load(open('$R/probe/v31_smoke_s$1.json'))['temporal']['delta'])" 2>/dev/null; }

smoke_seed 1
D1=$(delta_of 1)
if [ -z "$D1" ]; then
  echo "V31 SMOKE ABORT: s1 probe missing $(date '+%m-%d %H:%M')" >>"$OUT"; exit 1
fi
if .venv/bin/python -c "exit(0 if float('$D1')>0 else 1)"; then
  echo "[smoke] s1 delta=$D1 POSITIVE -> chaining seed 2 for the formal two-seed P1" >>"$OUT"
  smoke_seed 2
  D2=$(delta_of 2)
  if [ -n "$D2" ] && .venv/bin/python -c "exit(0 if float('$D2')>0 else 1)"; then
    echo "V31 SMOKE P1 PASS (s1=$D1, s2=$D2) -> proceed to 600k full wave (PREREG)" >>"$OUT"
  else
    echo "V31 SMOKE P1 FAIL at s2 (s1=$D1, s2=${D2:-missing}) -> temporal gate, do NOT tune weights" >>"$OUT"
  fi
else
  echo "V31 SMOKE P1 FAIL at s1 (delta=$D1) -> temporal gate, do NOT tune weights" >>"$OUT"
fi
echo "V31 SMOKE DONE $(date '+%m-%d %H:%M')" >>"$OUT"
