#!/bin/bash
# Auditor minimal experiment v2 -- SERIALIZED (v1 died to mutual pkill fratricide:
# each cell's pkill MainMultiDC killed the concurrent gamble eval's gateway and
# vice versa; van_anti_off died at 20:38 = exactly when oracle_s2 evals began).
# This version WAITS for the gamble seed-2 pipeline to fully finish, then owns
# the machine. All 8 cells rerun fresh (cells 1-4 of v1 are garbage: 0-4/10 eps).
# chi auditor via TRUST_GATE_SOURCE=resid (wiring verified in v1 logs:
# "ForecastResidualMonitor[gate]" banner appeared).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/auditor_local.txt
SUMMARY=$REPO/local_eval_rt/v2026_gamble_summary.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager

echo "[aud2] armed $(date '+%m-%d %H:%M'); waiting for gamble seed 2 DONE" >>"$OUT"
while ! grep -qa "V2026-GAMBLE-FV seed 2 DONE" "$SUMMARY" 2>/dev/null; do sleep 60; done
sleep 120   # let the fv script's last eval + gateway exit fully

declare -A CK EXP
CK[van]=logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
CK[eucrd]=logs/creg_eucrd_local_s3/multidc_gtrxl_training/PPO_multidc_env_014b9_00000_0_2026-07-15_14-42-21/checkpoint_000010
EXP[eucrd]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4

echo "===== AUDITOR LOCAL v2 (seed-3 pair, ep10, argmax, resid chi, serialized) $(date '+%m-%d %H:%M') =====" >>"$OUT"
run_cell () { # arm perturb auditor_mode
  local A="$1" PM="$2" AM="$3"
  local tag="${A}_${PM}_${AM}"
  local lg=$REPO/local_eval_rt/aud2_${tag}.log
  local envs=(FORECAST_PERTURB_MODE=$PM DECODE_TOPK=0)
  if [ "$AM" != "off" ]; then
    envs+=(TRUST_GATE_SOURCE=resid TRUST_GATE_MODE=$AM)
    [ "$AM" = "gate" ] && envs+=(TRUST_GATE_THRESH=0.2)
    # chi/trigger CSV: the ONLY reliable trigger evidence -- the monitor never
    # prints to stdout, so grep-based auditor-lines counts are decorative (GPU
    # box finding, 2026-08-09). Count triggers from the gated column of this CSV.
    envs+=(TRUST_GATE_LOG=$REPO/local_eval_rt/aud2_${tag}_chi.csv)
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  echo "[aud2 ${tag}] start $(date '+%m-%d %H:%M')" >>"$OUT"
  env "${envs[@]}" nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "${EXP[$A]}" --global rllib --local rllib --checkpoint "${CK[$A]}" \
    --new-api --shared-local --global-defer --episodes 10 --seed 3 \
    --output $REPO/local_eval_rt/eval_csv/aud2_${tag}.csv >"$lg" 2>&1
  local rc=$?
  local ne=$(grep -ac "^Episode .*/10:" "$lg" 2>/dev/null || true)
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local ng=$(grep -acE "gated|repair" "$lg" 2>/dev/null || true)
  echo "[aud2 ${A} ${PM} ${AM}] rc=${rc} eps=${ne} carbon=${cc:-?} completion=${cf:-?} (auditor-lines=${ng})" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
}
# decisive cells first
run_cell van   anti off
run_cell van   anti gate
run_cell eucrd anti off
run_cell eucrd anti gate
run_cell van   anti repair
run_cell eucrd anti repair
run_cell van   none off
run_cell eucrd none off
echo "AUDITOR LOCAL DONE $(date '+%m-%d %H:%M')" >>"$OUT"
