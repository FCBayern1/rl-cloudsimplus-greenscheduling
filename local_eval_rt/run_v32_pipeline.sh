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
  local rc=$?
  echo "[pipe] train $4 exit rc=$rc $(date '+%m-%d %H:%M')" >>"$OUT"
  if [ $rc -ne 0 ]; then
    echo "[pipe] ABORT: training $4 failed - stopping the whole chain" >>"$OUT"
    pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null
    exit 1
  fi
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; pkill -9 -f "ray::[A-Za-z]" 2>/dev/null; sleep 5
}
probe () {  # outdir jsonname
  local CK=$(ls -d logs/$1/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V | tail -1)
  [ -z "$CK" ] && { echo "[pipe] $1 no checkpoint" >>"$OUT"; return 1; }
  echo "----- probe $1 $(basename $CK) -----" >>"$OUT"
  .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$CK" --trials 40 \
    --raw-logits --json-out $R/probe/$2.json 2>>"$OUT" | grep -E "difference|fraction|raw defer|route logits|ratio" >>"$OUT"
}
probe_late () {  # outdir; probe every retained checkpoint in the final 40%
  local run=$1
  local cks=( $(ls -d logs/$run/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | sort -V) )
  local n=${#cks[@]}
  [ $n -eq 0 ] && { echo "[pipe] $run has no checkpoints for late scan" >>"$OUT"; return 1; }
  local start=$(( n * 6 / 10 ))
  for ((i=start; i<n; i++)); do
    local ck=${cks[$i]}
    local ckn=$(basename "$ck" | sed 's/checkpoint_0*/ck/')
    local name=${run}_${ckn}
    echo "----- lateprobe $run $ckn -----" >>"$OUT"
    .venv/bin/python probe_forecast_sensitivity.py --checkpoint "$PWD/$ck" --trials 40 \
      --json-out "$R/probe/$name.json" >>"$R/${name}_probe.txt" 2>>"$OUT" \
      || { echo "[pipe] lateprobe $run $ckn FAILED" >>"$OUT"; return 1; }
  done
}
summarize_run () {  # outdir jsonname [probe-glob]
  local run=$1 jsonname=$2 pattern=${3:-}
  local extra=()
  [ -n "$pattern" ] && extra=(--probe-glob "$pattern")
  .venv/bin/python summarize_v32_gate_run.py --run-dir "logs/$run" \
    "${extra[@]}" --json-out "$R/probe/$jsonname.json" \
    >"$R/${jsonname}.txt" \
    || { echo "[pipe] summarize $run FAILED" >>"$OUT"; return 1; }
}
# Gate-2 delta reads the JOB-ALIGNED channel (the factorized gate ignores
# dc_future_* BY DESIGN - seventh-review catch: the dc_* sweep would auto-fail
# a healthy V3.2 model). Falls back to the dc_* temporal delta for non-v32
# checkpoints (reference-wave probes).
delta () { .venv/bin/python -c "
import json
d = json.load(open('$R/probe/$1.json'))
jt = d.get('job_temporal')
print(jt['delta'] if jt else d['temporal']['delta'])" 2>/dev/null; }

# 3. V3.2 Gate 2 smoke (oracle s1 100k) -- prereg threshold +0.05, frozen
train experiment_v3_2_oracle 1 100000 v32_smoke_s1
probe v32_smoke_s1 v32_smoke_s1 || exit 1
summarize_run v32_smoke_s1 v32_smoke_s1_rollout || exit 1
# Gate 2 is MULTI-CONDITION (eighth review): one synthetic delta is not a
# verdict. All of: job-aligned delta >= 0.05, P(defer) monotone in
# forecast_gain and in slack, forecast channel >= 10x null (judgeability, A2).
# The real-rollout sign check is reported when Codex's rollout aggregator has
# produced a file; it is listed as NOT-AVAILABLE rather than silently skipped.
G2=$(PROBE_JSON=$R/probe/v32_smoke_s1.json ROLLOUT_JSON=$R/probe/v32_smoke_s1_rollout.json .venv/bin/python - <<'PYG' 2>/dev/null
import json, os
d = json.load(open(os.environ["PROBE_JSON"]))
r = json.load(open(os.environ["ROLLOUT_JSON"]))
jt = d.get("job_temporal") or {}
mo = d.get("monotone") or {}
s = d.get("summary") or {}
delta = jt.get("delta")
mg, ms = mo.get("monotone_frac_gain"), mo.get("monotone_frac_slack")
fc, nu = s.get("forecast", {}).get("tv"), s.get("null", {}).get("tv")
real = (r.get("rollout") or {}).get("rollout_temporal_delta")
conds = {
    "delta>=0.05": (delta is not None and delta >= 0.05, delta),
    "monotone_gain>=0.75": (mg is not None and mg >= 0.75, mg),
    "monotone_slack>=0.75": (ms is not None and ms >= 0.75, ms),
    "forecast>=10x_null": (fc is not None and nu is not None and fc >= 10 * nu,
                           None if fc is None else round(fc / max(nu, 1e-12), 1)),
    "real_rollout_same_sign": (real is not None and real > 0.0, real),
}
for k, (ok, v) in conds.items():
    print(f"  cond {k}: {'PASS' if ok else 'FAIL'} (value={v})")
print("PASS" if all(ok for ok, _ in conds.values()) else "FAIL")
PYG
)
echo "$G2" >>"$OUT"
G2=$(echo "$G2" | tail -1)
ROLL=$R/probe/v32_smoke_s1_rollout.json
if [ -f "$ROLL" ]; then echo "  rollout evidence: $ROLL" >>"$OUT";
else echo "  rollout evidence: NOT AVAILABLE" >>"$OUT"; fi
echo "V32 GATE2 $G2 $(date '+%H:%M')" >>"$OUT"

# 4. V3.2 Gate 3 FIRST (only if Gate 2 passed) - verdict ~21:00 tonight
#    so the human 600k decision can happen before midnight; the reference
#    s2 pair fills the rest of the night either way.
if [ "$G2" = "PASS" ]; then
  train experiment_v3_2_oracle 1 300000 v32_g3_s1
  probe v32_g3_s1 v32_g3_s1 || exit 1
  probe_late v32_g3_s1 || exit 1
  summarize_run v32_g3_s1 v32_g3_s1_summary "$R/probe/v32_g3_s1_ck*.json" || exit 1
  train experiment_v3_2_oracle 2 300000 v32_g3_s2
  probe v32_g3_s2 v32_g3_s2 || exit 1
  probe_late v32_g3_s2 || exit 1
  summarize_run v32_g3_s2 v32_g3_s2_summary "$R/probe/v32_g3_s2_ck*.json" || exit 1
  S1_SUM=$R/probe/v32_g3_s1_summary.json S2_SUM=$R/probe/v32_g3_s2_summary.json \
    EVIDENCE=$R/probe/v32_gate3_evidence.json .venv/bin/python - <<'PYG3'
import json, os
s1 = json.load(open(os.environ["S1_SUM"]))
s2 = json.load(open(os.environ["S2_SUM"]))
block = {
    "late_checkpoint_temporal_deltas_by_seed": {
        "1": s1.get("late_checkpoint_temporal_deltas", []),
        "2": s2.get("late_checkpoint_temporal_deltas", []),
    },
}
ratios = [s.get("learner", {}).get("defer_route_td_residual_ratio") for s in (s1, s2)]
ratios = [x for x in ratios if x is not None]
if len(ratios) == 2:
    block["defer_route_td_residual_ratio"] = max(ratios)
for key in ("all_defer", "backstop_dominant"):
    vals = [s.get("rollout", {}).get(key) for s in (s1, s2)]
    if all(v is not None for v in vals): block[key] = any(vals)
vals = [s.get("completion", {}).get("completion_collapse") for s in (s1, s2)]
if all(v is not None for v in vals): block["completion_collapse"] = any(vals)
json.dump({"gate3": block}, open(os.environ["EVIDENCE"], "w"), indent=2)
print("Gate3 diagnostic evidence:", json.dumps(block, sort_keys=True))
for seed, s in ((1, s1), (2, s2)):
    print(f"seed{seed} wait_realization={s['rollout'].get('wait_carbon_improvement_rate')} "
          f"adv_by_wait={s['learner'].get('advantage_by_wait_sec')}")
PYG3
  .venv/bin/python v32_gate_verdict.py --evidence "$R/probe/v32_gate3_evidence.json" \
    --json-out "$R/probe/v32_gate3_verdict.json" | grep '^GATE3:' >>"$OUT"
else
  echo "[pipe] Gate2 FAIL -> skipping Gate3 per prereg (no weight tuning, no extension)" >>"$OUT"
fi

# 5. reference wave s2 pair (600k each, new jars, pair-consistent)
train experiment_v3_1_oracle    2 600000 v31_oracle_s2
train experiment_v3_1_noforecast 2 600000 v31_nofc_s2
echo "[pipe] reference wave all trained $(date '+%m-%d %H:%M')" >>"$OUT"

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
