#!/bin/bash
# v3 scenario: prepare -> EMPIRICALLY calibrate the power divisor -> train.
#
# Why the calibration step exists: the v3 trace carries 58% more PE-steps than
# the v2026 gamble trace, so the old divisor (1600) would leave rho far from the
# designed 1.29. The gamble campaign already burned two rounds guessing this
# ("assumed 3000 -> measured 0.69"), hence the rule: measure demand, compute the
# divisor from it, never extrapolate.
#
# Arms (serial): oracle s1 -> noforecast s1 -> oracle s2 -> noforecast s2.
# First pairwise verdict lands after the first two.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
OUT=$R/v3_chain.txt
GATE=$R/nofc_ref_seeds.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib

echo "[v3] start (rebuilt config, audit-gated) $(date '+%m-%d %H:%M')" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3

# ---- 1. rebuild the gateway so the v3 trace + 9xxx wind land in the jar ----
echo "[v3] rebuilding gateway $(date '+%m-%d %H:%M')" >>"$OUT"
(cd $REPO/cloudsimplus-gateway && ./gradlew installDist -q) >>"$R/v3_build.log" 2>&1
JAR=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar
n9=$(unzip -l "$JAR" | grep -c "Turbine_9[0-9]*_2021.csv")
ntr=$(unzip -l "$JAR" | grep -c "v3b_n1200.csv")
echo "[v3] jar contains ${n9} 9xxx wind files, ${ntr} v3 trace" >>"$OUT"
if [ "$n9" -lt 7 ] || [ "$ntr" -lt 1 ]; then
  echo "[v3] ABORT: resources missing from jar" >>"$OUT"; exit 1
fi

# ---- 2. measure demand with a drain smoke, then compute the divisor ----
cd $REPO/drl-manager
SM=$R/v3_smoke_demand.log
echo "[v3] demand smoke $(date '+%m-%d %H:%M')" >>"$OUT"
.venv/bin/python -m src.baselines.evaluate --experiment experiment_v3_noforecast \
  --global round_robin --local drain --episodes 1 --seed 1 \
  --output $R/eval_csv/v3_smoke_demand.csv >"$SM" 2>&1
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3

DIV=$(.venv/bin/python - "$SM" <<'PY'
import csv, re, sys
import numpy as np
log = open(sys.argv[1], errors="ignore").read()
m = re.findall(r"Avg Total Energy:\s*([\d.]+)\s*Wh", log)
if not m:
    print("FAIL"); sys.exit(0)
demand_W = float(m[-1]) / 2.0                     # 7200 sim-s = 2 h
GATE = "../cloudsimplus-gateway/src/main/resources/windProduction/simplified"
TIDS = [9012, 9036, 9095, 9091, 9096, 9101, 9103]
tot = None
for t in TIDS:
    v = np.array([float(r["power_kw"]) for r in csv.DictReader(open(f"{GATE}/Turbine_{t}_2021.csv"))])
    tot = v if tot is None else tot + v
offs = [(1009 * k) % 4800 for k in range(10)]
means = [tot[13 + o: 13 + o + 7200].mean() for o in offs if tot[13 + o: 13 + o + 7200].size == 7200]
green_kW = float(np.mean(means))                  # mean summed kW over deployment windows
div = green_kW * 1000.0 / (1.29 * demand_W)       # want mean green W = 1.29 x demand W
print(f"{div:.1f}" if 100 < div < 20000 else "FAIL", file=sys.stderr)
print(f"demand_W={demand_W:.0f} green_kW={green_kW:.1f} -> divisor={div:.1f}", file=sys.stdout)
PY
)
CALC=$(echo "$DIV" | grep -oE "divisor=[0-9.]+" | cut -d= -f2)
echo "[v3] calibration: $DIV" >>"$OUT"
if [ -z "$CALC" ]; then echo "[v3] ABORT: calibration failed" >>"$OUT"; exit 1; fi
sed -i "s/compressed_power_divisor: 1010.0/compressed_power_divisor: ${CALC}/g" $REPO/config_C.yml
echo "[v3] divisor set to ${CALC} in both v3 blocks" >>"$OUT"

# ---- 3. verification smoke: report the realised green ratio ----
VS=$R/v3_smoke_verify.log
.venv/bin/python -m src.baselines.evaluate --experiment experiment_v3_noforecast \
  --global green_aware --local drain --episodes 1 --seed 1 \
  --output $R/eval_csv/v3_smoke_verify.csv >"$VS" 2>&1
gr=$(grep -a "Avg Green Ratio" "$VS" | tail -1 | grep -oE "[0-9.]+%" | head -1)
cf=$(grep -a "Avg Finished" "$VS" | grep -av Calling | tail -1 | grep -oE "[0-9.]+%" | head -1)
echo "[v3] verify smoke (green_aware): green=${gr:-?} completion=${cf:-?}" >>"$OUT"
pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3

# ---- 3b. HARD GATE: the scenario audit must pass on the calibrated config ----
echo "[v3] preflight audit $(date '+%m-%d %H:%M')" >>"$OUT"
.venv/bin/python preflight_scenario.py experiment_v3_oracle experiment_v3_noforecast >>"$OUT" 2>&1
if [ $? -ne 0 ]; then
  echo "[v3] ABORT: preflight audit failed after calibration -- no GPU spent" >>"$OUT"; exit 1
fi

# ---- 4. train the four arms, evaluating each ----
declare -A EXP
EXP[oracle]=experiment_v3_oracle
EXP[nofc]=experiment_v3_noforecast
run_arm () {  # arm seed
  local A="$1" SEED="$2"
  local OD=$REPO/drl-manager/logs/v3_${A}_s${SEED}
  pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  echo "[v3 ${A}_s${SEED}] train start $(date '+%m-%d %H:%M')" >>"$OUT"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py --config "$REPO/config_C.yml" \
    --experiment "${EXP[$A]}" --total-timesteps 600000 --num-workers 6 --seed $SEED \
    --output-dir "$OD" > "$R/v3_${A}_s${SEED}_train.log" 2>&1
  echo "[v3 ${A}_s${SEED}] train exit rc=$? $(date '+%m-%d %H:%M')" >>"$OUT"
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 3
  local RUN=$(ls -d "$OD"/multidc_gtrxl_training/PPO_*/ 2>/dev/null | head -1)
  for CK in $(ls -d ${RUN}checkpoint_* 2>/dev/null | sort -V | tail -3); do
    local ckn=$(basename "$CK" | sed 's/checkpoint_0*/ck/')
    local lg=$R/v3_${A}_s${SEED}_${ckn}.log
    FORECAST_PERTURB_MODE=none DECODE_TOPK=0 \
    .venv/bin/python -m src.baselines.evaluate --experiment "${EXP[$A]}" --global rllib \
      --local rllib --checkpoint "$CK" --new-api --shared-local --global-defer \
      --episodes 10 --seed $SEED \
      --output "$R/eval_csv/v3_${A}_s${SEED}_${ckn}.csv" >"$lg" 2>&1
    local cc=$(grep -a "Avg Carbon/MI" "$lg"|tail -1|grep -oE "[0-9.]+"|head -1)
    local cf2=$(grep -a "Avg Finished" "$lg"|grep -av Calling|tail -1|grep -oE "[0-9.]+%"|head -1)
    local gr2=$(grep -a "Avg Green Ratio" "$lg"|tail -1|grep -oE "[0-9.]+%"|head -1)
    echo "[v3 ${A}_s${SEED} ${ckn}@ARGMAX clean] cc=${cc:-?} completion=${cf2:-?} green=${gr2:-?}" >>"$OUT"
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  done
}
run_arm oracle 1
run_arm nofc   1
echo "V3 PAIR-1 DONE $(date '+%m-%d %H:%M')" >>"$OUT"
run_arm oracle 2
run_arm nofc   2
echo "V3 ALL DONE $(date '+%m-%d %H:%M')" >>"$OUT"
