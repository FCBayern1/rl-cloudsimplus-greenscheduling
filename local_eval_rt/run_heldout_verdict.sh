#!/bin/bash
# HELD-OUT WINDOW VERDICT for the noforecast_s2 anomaly (nofc 0.0277 == oracle
# level; s1 gap was -46%). Suspect: memorization. Training (600k steps, 6
# workers) exposed only k=0..~14 windows, ~6 visits each; eval k=0..9 is
# IN-DISTRIBUTION. This runs 26 episodes per arm: episodes 1-10 replay the
# seen windows, episodes 17-26 (k=16..25) are windows training NEVER saw.
#   memorization  => nofc_s2 held-out carbon degrades toward s1's ~0.06+
#   real reactive => nofc_s2 held-out stays ~0.028 (scenario claim in danger)
# Waits for the gamble fv chain to fully finish, then owns the machine.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
OUT=$REPO/local_eval_rt/heldout_verdict.txt
SUMMARY=$REPO/local_eval_rt/v2026_gamble_summary.txt
export EVAL_CONFIG_PATH=$REPO/config_C.yml
export GATEWAY_LIBS=$REPO/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
cd $REPO/drl-manager

echo "[heldout] armed $(date '+%m-%d %H:%M'); waiting for gamble seed 2 DONE" >>"$OUT"
while ! grep -qa "V2026-GAMBLE-FV seed 2 DONE" "$SUMMARY" 2>/dev/null; do sleep 60; done
sleep 120

declare -A CK EXP NEP
CK[nofc_s2]=logs/v2026gb_noforecast_s2/multidc_gtrxl_training/PPO_multidc_env_10704_00000_0_2026-08-08_23-46-59/checkpoint_000008
EXP[nofc_s2]=experiment_v2026_gamble_noforecast
CK[oracle_s2]=logs/v2026gb_oracle_s2/multidc_gtrxl_training/PPO_multidc_env_e4c8d_00000_0_2026-08-08_14-13-06/checkpoint_000009
EXP[oracle_s2]=experiment_v2026_gamble_oracle
CK[nofc_s1]=logs/v2026gb_noforecast_s1/multidc_gtrxl_training/PPO_multidc_env_51dd6_00000_0_2026-08-08_05-47-54/checkpoint_000008
EXP[nofc_s1]=experiment_v2026_gamble_noforecast
# OOB arms: same checkpoints, 2020-wind band (turbine 8xxx), 10 eps only.
CK[nofc_s2_oob]=${CK[nofc_s2]}
EXP[nofc_s2_oob]=experiment_v2026_gamble_noforecast_oob2020
CK[oracle_s2_oob]=${CK[oracle_s2]}
EXP[oracle_s2_oob]=experiment_v2026_gamble_oracle_oob2020
CK[nofc_s1_oob]=${CK[nofc_s1]}
EXP[nofc_s1_oob]=experiment_v2026_gamble_noforecast_oob2020

echo "===== VERDICT (OOB=2020-wind 10eps; in-band 26eps: 1-10 seen, 17-26 unseen-phase) $(date '+%m-%d %H:%M') =====" >>"$OUT"
run_arm () { # tag episodes
  local T="$1" NE="$2"
  local lg=$REPO/local_eval_rt/heldout_${T}.log
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
  echo "[heldout ${T}] start $(date '+%m-%d %H:%M') eps=${NE}" >>"$OUT"
  nice -n 5 .venv/bin/python -m src.baselines.evaluate \
    --experiment "${EXP[$T]}" --global rllib --local rllib --checkpoint "${CK[$T]}" \
    --new-api --shared-local --global-defer --episodes $NE --seed 2 \
    --output $REPO/local_eval_rt/eval_csv/heldout_${T}.csv >"$lg" 2>&1
  .venv/bin/python - "$lg" "$T" "$NE" >>"$OUT" <<'PY'
import re, sys, statistics as st
lg, tag, ne = sys.argv[1], sys.argv[2], int(sys.argv[3])
rows = []
for line in open(lg, errors="ignore"):
    m = re.search(r"Episode (\d+)/\d+.*?Finished=([\d.]+)%.*GreenRatio=([\d.]+)%.*Carbon=([\d.]+)kg.*DeadlineForced=(\d+)", line)
    if m:
        rows.append(tuple(float(m.group(i)) for i in (1,2,3,4,5)))
def agg(sel, label):
    s = [r for r in rows if int(r[0]) in sel]
    if not s:
        print(f"[heldout {tag}] {label}: n=0 (EVAL FAILED?)"); return
    print(f"[heldout {tag}] {label}: n={len(s)} carbon_kg med={st.median(x[3] for x in s):.4f} "
          f"mean={sum(x[3] for x in s)/len(s):.4f} compl={sum(x[1] for x in s)/len(s):.2f}% "
          f"green={sum(x[2] for x in s)/len(s):.2f}% dlforced={sum(x[4] for x in s)/len(s):.0f}")
if ne >= 26:
    agg(range(1,11),  "SEEN(k0-9)   ")
    agg(range(17,27), "UNSEEN(k16-25)")
else:
    agg(range(1,ne+1), "OOB(all)     ")
PY
  pkill -9 -f "exe[.]edu[.]cspg[.]MainMultiDC" 2>/dev/null; sleep 3
}
# OOB first: the decisive cells (new wind, statistics matched to training band)
run_arm nofc_s2_oob   10
run_arm oracle_s2_oob 10
run_arm nofc_s1_oob   10
# in-band phase-shift (weaker test; windows overlap the training band)
run_arm nofc_s2   26
run_arm oracle_s2 26
run_arm nofc_s1   26
echo "HELDOUT VERDICT DONE $(date '+%m-%d %H:%M')" >>"$OUT"
