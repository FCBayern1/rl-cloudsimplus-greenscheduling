#!/bin/bash
# Master watcher for the C-regime campaign. Every 15 min it:
#  1. snapshots the Isambard queue (creg_/evs_/ehc_ jobs) — SSH failures (e.g. the
#     clifton cert expiring ~12h) are logged and RETRIED, never treated as "queue empty";
#  2. when a training arm reaches checkpoint_000010, submits its stochastic re-eval
#     (evs_) exactly once (marker files), then cancels the parent training job whose
#     remaining work is only the useless argmax eval — cancel happens ONLY after both
#     the ckpt_10 check (numeric sort) and the evs submission succeeded;
#  3. tracks the local ablation chain (v2) via its ALL-COMPLETE line;
#  4. when the queue is empty (verified on a SUCCESSFUL ssh), all evals are submitted
#     and the local chain is done -> pulls summaries and builds the three paper tables.
set -uo pipefail
REPO="/home/joshua/rl-cloudsimplus-greenscheduling"
PY="$REPO/drl-manager/.venv/bin/python"
HOST="u6kd.aip2.isambard"
RUNS="/scratch/u6kd/joshualmw.u6kd/rl-runs"
ISB="/projects/u6kd/rl-cloudsimplus-greenscheduling/isambard"
LOG="$REPO/drl-manager/logs/master_watcher.log"
STATE="$REPO/drl-manager/logs/watcher_state"; mkdir -p "$STATE"
SUMS="$REPO/paper_materials/isambard_summaries"; mkdir -p "$SUMS"
OUT="$REPO/drl-manager/logs/FINAL_TABLES.txt"
LOCAL_SUM="$REPO/drl-manager/logs/cregime_local_ablation_summary.txt"

# arm:seed pairs whose stochastic re-eval this watcher owns (others already submitted)
PENDING_EVALS="cvar:1 cvar:2 cvar:3 dcvar:1 dcvar:2 dcvar:3 dr:1 dr:2 dr:3 mv:3 eucrd:3 eucrdv5:1 eucrdv5:2 eucrdv5:3"
exp_of () { case "$1" in
  cvar)  echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_risk_cvar;;
  dcvar) echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_risk_dcvar;;
  mv)    echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_risk_mv;;
  eucrd) echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v4;;
  eucrdv5) echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_v5;;
  dr)    echo experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap;;  # DR evals on base env
esac; }

stamp () { date +'%m-%d %H:%M'; }
log () { echo "[$(stamp)] $*" >> "$LOG"; }

log "master watcher started (pid $$)"
while :; do
  # ---- one ssh round-trip: queue + ckpt status of pending arms ----
  probe=$(timeout 50 ssh "$HOST" '
    echo "QOK"
    squeue --me -h -o "%j %t" 2>/dev/null | grep -E "^(creg_|evs_|ehc_)" | sort
    echo "CKPTS"
    for d in cregime_cvar_s1 cregime_cvar_s2 cregime_cvar_s3 cregime_dcvar_s1 cregime_dcvar_s2 cregime_dcvar_s3 cregime_dr_s1 cregime_dr_s2 cregime_dr_s3 cregime_mv_s3 cregime_eucrd_s3 cregime_eucrdv5_s1 cregime_eucrdv5_s2 cregime_eucrdv5_s3; do
      hi=$(ls -d '"$RUNS"'/$d/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | grep -oE "checkpoint_[0-9]+" | sort | tail -1)
      echo "$d ${hi:-none}"
    done' 2>/dev/null)
  if ! echo "$probe" | grep -q "^QOK"; then
    log "ssh FAILED (cert expired or network) — retrying next round, nothing assumed"
    sleep 900; continue
  fi
  queue=$(echo "$probe" | sed -n '/^QOK$/,/^CKPTS$/p' | grep -vE "^(QOK|CKPTS)$" || true)
  nq=$(echo "$queue" | grep -c . || true)

  # ---- submit stochastic evals for freshly finished arms ----
  for pair in $PENDING_EVALS; do
    arm="${pair%%:*}"; seed="${pair##*:}"
    marker="$STATE/evs_${arm}_s${seed}.submitted"
    [ -f "$marker" ] && continue
    dir="cregime_${arm}_s${seed}"
    ck=$(echo "$probe" | grep "^$dir " | awk '{print $2}')
    if [ "$ck" = "checkpoint_000010" ]; then
      exp=$(exp_of "$arm")
      sub=$(timeout 50 ssh "$HOST" "cd $ISB && sbatch --job-name=evs_${arm}_s${seed} --export=ALL,EXP=$exp,OUT=$arm,SEED=$seed,RUNDIR=$RUNS/$dir eval_stoch.sbatch" 2>/dev/null)
      if echo "$sub" | grep -q "Submitted"; then
        touch "$marker"
        log "SUBMITTED stoch eval $arm s$seed ($sub); cancelling parent trainer (argmax eval is useless)"
        timeout 40 ssh "$HOST" "scancel --me -n creg_${arm}_s${seed}" 2>/dev/null
      else
        log "submit FAILED for $arm s$seed — will retry"
      fi
    fi
  done

  # ---- local chain status ----
  local_done=0
  grep -q "ALL ABLATION ARMS COMPLETE" "$LOCAL_SUM" 2>/dev/null && local_done=1

  missing=$(for p in $PENDING_EVALS; do [ -f "$STATE/evs_${p%%:*}_s${p##*:}.submitted" ] || echo -n "$p "; done)
  log "queue=$nq jobs | evals pending submit: ${missing:-none} | local_chain_done=$local_done"

  # ---- completion gate (only on a successful ssh round) ----
  if [ "$nq" -eq 0 ] && [ -z "$missing" ] && [ "$local_done" -eq 1 ]; then
    log "ALL DONE — pulling summaries and building tables"
    timeout 120 rsync -q "$HOST":"$RUNS"/'*summary*.txt' "$SUMS"/ 2>>"$LOG"
    cp "$LOCAL_SUM" "$SUMS"/ 2>/dev/null
    { echo "===== built $(date +'%F %T') ====="
      "$PY" "$REPO/paper_materials/aggregate_cregime.py" "$SUMS"/*.txt 2>&1
    } > "$OUT"
    log "tables written to $OUT"
    break
  fi
  sleep 900
done
