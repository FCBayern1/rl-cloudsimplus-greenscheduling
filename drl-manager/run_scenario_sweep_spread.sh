#!/bin/bash
# Round 2 of the scenario sweep: SAME five scenarios, but with the fixed
# global routing (--routing spread: batch split across green DCs
# proportionally to current green power) instead of argmax-pile-up.
# Purpose: separate "forecast value" from the routing flaw that drowned
# 84-89% of load on one DC in round 1. Full log per scenario in
# ~/sweep_logs/spread_<name>.log; verdict lines in ~/scenario_sweep_spread.summary.
set -uo pipefail
cd "$(dirname "$0")"
OUT=~/scenario_sweep_spread.summary
# sync freshly-generated traces into the JVM classpath (build/resources/main)
cp -u ../cloudsimplus-gateway/src/main/resources/traces/*.csv ../cloudsimplus-gateway/build/resources/main/traces/ 2>/dev/null || true
mkdir -p ~/sweep_logs
run () { # name experiment trace green horizon
  echo "=== $1 start $(date '+%H:%M') ===" | tee -a $OUT
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
  .venv/bin/python oracle_fdefer_gate.py --experiment "$2" \
    --trace "$3" --green-ts "$4" --horizon "$5" --routing spread \
    > ~/sweep_logs/spread_$1.log 2>&1
  grep -aE "drain |reactive |fcast |GATE|Δ|vs-" ~/sweep_logs/spread_$1.log | tail -8 | tee -a $OUT
}
run 拥挤轻1.25x experiment_sweep_rwv3l  traces/realwind_v3l.csv           data/green_stretch.npy        504
run 拥挤中1.5x  experiment_sweep_rwv3m  traces/realwind_v3m.csv           data/green_stretch.npy        504
run 缺电版      experiment_sweep_scarce traces/realwind_stretch_tight.csv data/green_stretch_half.npy   504
run 错峰版      experiment_sweep_offset traces/realwind_stretch_tight.csv data/green_stretch_offset.npy 504
run dc8         experiment_sweep_dc8    traces/realwind_v3.csv            data/green_stretch_8dc.npy    504
echo "SPREAD SWEEP DONE $(date '+%H:%M')" | tee -a $OUT
