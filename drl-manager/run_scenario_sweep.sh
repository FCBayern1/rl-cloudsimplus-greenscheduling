#!/bin/bash
# Scenario-portfolio rule sweep (NO training; scripted rules driving the real
# simulator). Run on any box with the jar built. Each scenario ~15-25 min.
# Verdict lines land in ~/scenario_sweep.summary; the PASS bar is the fcast
# rule beating the reactive rule by >=15% carbon at >= equal completion.
set -uo pipefail
cd "$(dirname "$0")"
OUT=~/scenario_sweep.summary
run () { # name experiment trace green horizon
  echo "=== $1 start $(date '+%H:%M') ===" | tee -a $OUT
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
  .venv/bin/python oracle_fdefer_gate.py --experiment "$2" \
    --trace "$3" --green-ts "$4" --horizon "$5" 2>&1 \
    | grep -avE "Calling Java|STEP" | tail -8 | tee -a $OUT
}
run 拥挤轻1.25x experiment_sweep_rwv3l  traces/realwind_v3l.csv          data/green_stretch.npy        504
run 拥挤中1.5x  experiment_sweep_rwv3m  traces/realwind_v3m.csv          data/green_stretch.npy        504
run 缺电版      experiment_sweep_scarce traces/realwind_stretch_tight.csv data/green_stretch_half.npy   504
run 错峰版      experiment_sweep_offset traces/realwind_stretch_tight.csv data/green_stretch_offset.npy 504
echo "SWEEP DONE $(date '+%H:%M')" | tee -a $OUT
# appended: 8-DC scale scenario
run dc8 experiment_sweep_dc8 traces/realwind_v3.csv data/green_stretch_8dc.npy 504
echo "SWEEP+8DC DONE $(date '+%H:%M')" | tee -a $OUT
