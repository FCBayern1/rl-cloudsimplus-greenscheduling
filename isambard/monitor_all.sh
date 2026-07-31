#!/bin/bash
# Watch every C-regime experiment (Isambard main + ablation, plus the two local
# arms) until all finish, then pull the eval summaries and build the main +
# ablation tables. Snapshot every 10 min to logs/monitor_all.log.
set -uo pipefail
REPO="/home/joshua/rl-cloudsimplus-greenscheduling"
PY="$REPO/drl-manager/.venv/bin/python"
HOST="u6kd.aip2.isambard"
RUNS="/scratch/u6kd/joshualmw.u6kd/rl-runs"
LOG="$REPO/drl-manager/logs/monitor_all.log"
SUMS="$REPO/paper_materials/isambard_summaries"
OUT="$REPO/drl-manager/logs/FINAL_TABLES.txt"
PREFIXES="creg_van creg_mv creg_eucrd cca_ eh_ca cnc_ nq_ sc_ nf_"
mkdir -p "$SUMS" "$REPO/drl-manager/logs"

stamp () { date +'%Y-%m-%d %H:%M:%S'; }

snapshot () {  # -> writes one status block, echoes remaining-count
  local sq rem=0 line
  sq=$(timeout 60 ssh "$HOST" 'squeue --me -h -o "%j %t"' 2>/dev/null)
  echo "[$(stamp)] ---- Isambard ----" >> "$LOG"
  for pre in $PREFIXES; do
    local n r p
    n=$(echo "$sq" | grep -c "^$pre" || true)
    r=$(echo "$sq" | grep "^$pre" | grep -cw R || true)
    p=$(echo "$sq" | grep "^$pre" | grep -cw PD || true)
    [ "$n" -gt 0 ] && printf '  %-12s total=%s R=%s PD=%s\n' "$pre" "$n" "$r" "$p" >> "$LOG"
    rem=$((rem + n))
  done
  # local arms
  local lloc=0
  if kill -0 374621 2>/dev/null; then echo "  local: EU-CRD s3 ALIVE" >> "$LOG"; lloc=$((lloc+1)); fi
  if pgrep -f "creg_van_local_s3" >/dev/null 2>&1; then echo "  local: vanilla s3 ALIVE" >> "$LOG"; lloc=$((lloc+1)); fi
  if kill -0 403390 2>/dev/null; then echo "  local: vanilla-s3 monitor waiting" >> "$LOG"; lloc=$((lloc+1)); fi
  echo "  REMAINING: isambard=$rem local=$lloc" >> "$LOG"
  echo $((rem + lloc))
}

echo "[$(stamp)] monitor_all started" >> "$LOG"
while :; do
  n=$(snapshot)
  [ "$n" -eq 0 ] && break
  sleep 600
done

echo "[$(stamp)] ALL EXPERIMENTS DONE — pulling summaries + building tables" >> "$LOG"
rsync -q "$HOST":"$RUNS"/'*summary*.txt' "$SUMS"/ 2>>"$LOG"
# also fold in the local summary
cp "$REPO/drl-manager/logs/cregime_local_summary.txt" "$SUMS"/ 2>/dev/null || true
{
  echo "===== built $(stamp) ====="
  "$PY" "$REPO/paper_materials/build_table.py" "$SUMS"/*.txt 2>&1
} > "$OUT"
echo "[$(stamp)] tables written to $OUT" >> "$LOG"
