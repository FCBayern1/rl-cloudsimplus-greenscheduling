#!/bin/bash
# Auditor minimal experiment, LOCAL (GPU box couldn't fetch ckpts; runs here
# nice-d alongside gamble s2 training -- contention slows walltime, not carbon).
# 8 cells: {van, eucrd_v4} x {anti-off, anti-gate, anti-repair, clean-off},
# argmax, ep10, seed 3 (the locally available matched pair).
# chi auditor selected via TRUST_GATE_SOURCE=resid (GPU probe finding);
# gate thresh 0.2, repair thresh class-default -0.5. Most informative cells first.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/auditor_local.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
declare -A CK EXP
CK[van]=logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
CK[eucrd]=logs/creg_eucrd_local_s3/multidc_gtrxl_training/PPO_multidc_env_014b9_00000_0_2026-07-15_14-42-21/checkpoint_000010
EXP[eucrd]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4

echo "===== AUDITOR LOCAL (seed-3 pair, ep10, argmax, resid chi) $(date '+%m-%d %H:%M') =====" >>"$OUT"
run_cell () { # arm perturb auditor_mode
  local A="$1" PM="$2" AM="$3"
  local tag="${A}_${PM}_${AM}"
  local lg=$REPO/local_eval_rt/aud_${tag}.log
  local envs=(FORECAST_PERTURB_MODE=$PM DECODE_TOPK=0)
  if [ "$AM" != "off" ]; then
    envs+=(TRUST_GATE_SOURCE=resid TRUST_GATE_MODE=$AM)
    [ "$AM" = "gate" ] && envs+=(TRUST_GATE_THRESH=0.2)
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
  env "${envs[@]}" nice -n 15 .venv/bin/python -m src.baselines.evaluate \
    --experiment "${EXP[$A]}" --global rllib --local rllib --checkpoint "${CK[$A]}" \
    --new-api --shared-local --global-defer --episodes 10 --seed 3 \
    --output $REPO/local_eval_rt/eval_csv/aud_${tag}.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local ng=$(grep -acE "gated|repair" "$lg" 2>/dev/null || echo 0)
  echo "[aud ${A} ${PM} ${AM}] carbon=${cc:-?} completion=${cf:-?} (auditor-lines=${ng})" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
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
