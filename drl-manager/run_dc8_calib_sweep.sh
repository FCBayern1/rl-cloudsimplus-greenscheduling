#!/bin/bash
# Load-calibration sweep for the 8DC champion. Round-2 showed 8DC has the
# cleanest carbon lever (fcast vs drain -26.5%) but is OVERLOADED (drain
# completion only 0.28), so the carbon number is contaminated by dropped work.
# These two candidates cut the load (anti-phase workload, brown-frac 0.9) to
# push drain toward healthy completion while keeping the temporal lever:
#   dc8_light : ~0.8 G-MI
#   dc8_med   : ~1.3 G-MI
# Goal: find the load where drain completes ~>=90% AND fcast still buys >=15%
# carbon at near-iso-completion. Same green (green_stretch_8dc), spread routing.
# Full log per scenario in ~/sweep_logs/dc8calib_<name>.log; verdicts in
# ~/scenario_sweep_dc8calib.summary.
set -uo pipefail
cd "$(dirname "$0")"
OUT=~/scenario_sweep_dc8calib.summary
mkdir -p ~/sweep_logs
run () { # name experiment trace green horizon
  echo "=== $1 start $(date '+%H:%M') ===" | tee -a $OUT
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 2
  .venv/bin/python oracle_fdefer_gate.py --experiment "$2" \
    --trace "$3" --green-ts "$4" --horizon "$5" --routing spread \
    > ~/sweep_logs/dc8calib_$1.log 2>&1
  grep -aE "drain |reactive |fcast |GATE|Δ|vs-" ~/sweep_logs/dc8calib_$1.log | tail -8 | tee -a $OUT
}
run dc8_light experiment_sweep_dc8_light traces/realwind_dc8_light.csv data/green_stretch_8dc.npy 504
run dc8_med   experiment_sweep_dc8_med   traces/realwind_dc8_med.csv   data/green_stretch_8dc.npy 504
echo "DC8 CALIB SWEEP DONE $(date '+%H:%M')" | tee -a $OUT
