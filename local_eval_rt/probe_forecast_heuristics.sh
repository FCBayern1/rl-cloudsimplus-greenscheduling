#!/bin/bash
# Forecast-discriminativeness probes: green_forecast (forecast-consuming
# heuristic) vs green_aware (reactive heuristic) on BOTH bands.
# Waits for the running green_aware/RR OOB probes to finish first (serialized
# against ourselves; the heldout chain runs concurrently on its own gateway).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/probe_forecast_heuristics.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager
while ! grep -qa "Avg Carbon/MI" $R/oob_rr.log 2>/dev/null; do sleep 120; done
echo "===== FORECAST-HEURISTIC PROBES $(date '+%m-%d %H:%M') =====" >>"$OUT"
run_p () { # tag experiment global
  local TAG="$1" EXPN="$2" G="$3"
  local lg=$R/probe_${TAG}.log
  echo "[probe ${TAG}] start $(date '+%m-%d %H:%M')" >>"$OUT"
  nice -n 12 .venv/bin/python -m src.baselines.evaluate --experiment "$EXPN" \
    --global "$G" --local drain --episodes 10 --seed 2 \
    --output $R/eval_csv/probe_${TAG}.csv >"$lg" 2>&1
  local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
  local cf=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
  local gr=$(grep -a "Avg Green" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
  echo "[probe ${TAG}] carbon=${cc:-?} completion=${cf:-?} green=${gr:-?}" >>"$OUT"
}
# 2021 TRAINING band: does a forecast help even a heuristic here?
run_p gf_2021 experiment_v2026_gamble_oracle     green_forecast
run_p ga_2021 experiment_v2026_gamble_noforecast green_aware
# 2020 OOB band: same question on the flat-verdict band
run_p gf_oob  experiment_v2026_gamble_oracle_oob2020 green_forecast
echo "FORECAST-HEURISTIC PROBES DONE $(date '+%m-%d %H:%M')" >>"$OUT"
