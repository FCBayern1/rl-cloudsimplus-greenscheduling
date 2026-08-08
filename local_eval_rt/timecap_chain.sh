#!/bin/bash
# Chain: after gamble seed 2 finishes, run TimeCAP arms for seeds 1 and 2.
# HARD GATE: the TimeCAP offset-sync probe must have passed (closed-book
# desync would make the arms silently measure a different day's forecast).
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
SUMMARY=$REPO/local_eval_rt/v2026_gamble_summary.txt
LOG=$REPO/local_eval_rt/timecap_chain.log
echo "[tc-chain] armed $(date '+%m-%d %H:%M'); waiting for gamble seed 2 DONE" >>"$LOG"
while ! grep -qa "V2026-GAMBLE-FV seed 2 DONE" "$SUMMARY" 2>/dev/null; do sleep 60; done
if ! grep -qa "TIMECAP OFFSET SYNC OK" /tmp/tcprobe_out.log 2>/dev/null; then
  echo "[tc-chain] PROBE NOT PASSED -- refusing to launch timecap arms $(date '+%m-%d %H:%M')" >>"$LOG"
  exit 1
fi
echo "[tc-chain] seed2 done + probe OK -> timecap s1 then s2 $(date '+%m-%d %H:%M')" >>"$LOG"
pkill -9 -f "ray::[A-Za-z]" 2>/dev/null || true
sleep 5
bash $REPO/local_eval_rt/run_v2026_gamble_timecap.sh 1 >>"$LOG" 2>&1
bash $REPO/local_eval_rt/run_v2026_gamble_timecap.sh 2 >>"$LOG" 2>&1
echo "[tc-chain] both timecap seeds done $(date '+%m-%d %H:%M')" >>"$LOG"
