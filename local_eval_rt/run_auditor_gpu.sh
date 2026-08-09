#!/bin/bash
# Auditor minimal experiment -- GPU-BOX PORTABLE VERSION.
# 8 cells: {van, eucrd_v4} x {anti-off, anti-gate, anti-repair, clean-off},
# argmax, ep10, seed 3, chi auditor (TRUST_GATE_SOURCE=resid -- REQUIRED, the
# default qvar source needs a Q-ensemble the vanilla ckpt does not have).
# Ckpts ship via the ckpt-transfer branch tarball; this script self-extracts.
# All 8 cells run on THIS machine -> internal comparisons are same-machine and
# self-anchored (clean cells included); do NOT compare against local-box
# historicals (16% cross-machine noise floor).
set -uo pipefail
REPO=$(cd "$(dirname "$0")/.." && pwd)
OUT=$REPO/local_eval_rt/auditor_gpu.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
mkdir -p $REPO/local_eval_rt/eval_csv

# self-extract the ckpt pair if missing
if [ ! -d logs/creg_van_local_s3 ] || [ ! -d logs/creg_eucrd_local_s3 ]; then
  echo "[aud-gpu] extracting ckpt pair from ckpt-transfer branch" >>"$OUT"
  git -C $REPO fetch origin ckpt-transfer
  git -C $REPO show origin/ckpt-transfer:ckpt_cregime_s3_pair.tar.gz | tar xzf - -C logs/
fi

declare -A CK EXP
CK[van]=logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010
EXP[van]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
CK[eucrd]=logs/creg_eucrd_local_s3/multidc_gtrxl_training/PPO_multidc_env_014b9_00000_0_2026-07-15_14-42-21/checkpoint_000010
EXP[eucrd]=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4
for A in van eucrd; do
  [ -d "${CK[$A]}" ] || { echo "[aud-gpu] MISSING CKPT ${CK[$A]} -- abort" >>"$OUT"; exit 1; }
done

echo "===== AUDITOR GPU (seed-3 pair, ep10, argmax, resid chi) $(date '+%m-%d %H:%M') host=$(hostname) =====" >>"$OUT"
run_cell () { # arm perturb auditor_mode
  local A="$1" PM="$2" AM="$3"
  local tag="${A}_${PM}_${AM}"
  local lg=$REPO/local_eval_rt/audgpu_${tag}.log
  local envs=(FORECAST_PERTURB_MODE=$PM DECODE_TOPK=0)
  if [ "$AM" != "off" ]; then
    envs+=(TRUST_GATE_SOURCE=resid TRUST_GATE_MODE=$AM)
    [ "$AM" = "gate" ] && envs+=(TRUST_GATE_THRESH=0.2)
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  echo "[aud-gpu ${tag}] start $(date '+%m-%d %H:%M')" >>"$OUT"
  env "${envs[@]}" .venv/bin/python -m src.baselines.evaluate \
    --experiment "${EXP[$A]}" --global rllib --local rllib --checkpoint "${CK[$A]}" \
    --new-api --shared-local --global-defer --episodes 10 --seed 3 \
    --output $REPO/local_eval_rt/eval_csv/audgpu_${tag}.csv >"$lg" 2>&1
  local rc=$?
  local ne=$(grep -ac "^Episode .*/10:" "$lg" 2>/dev/null || true)
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local ng=$(grep -acE "gated|repair" "$lg" 2>/dev/null || true)
  echo "[aud-gpu ${A} ${PM} ${AM}] rc=${rc} eps=${ne} carbon=${cc:-?} completion=${cf:-?} (auditor-lines=${ng})" >>"$OUT"
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
echo "AUDITOR GPU DONE $(date '+%m-%d %H:%M')" >>"$OUT"
