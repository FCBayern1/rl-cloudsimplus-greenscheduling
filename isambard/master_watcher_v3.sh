#!/bin/bash
# Master watcher v3 — wave of 2026-07-18 (rs2 retrain, van s3, v5b full-ckpt
# retrain, v5nl local-CF ablation). Differences from v2:
#   1. Trigger is no longer "checkpoint_000010 EXACTLY": fire on ckpt_10 (then
#      cancel the parent trainer whose remaining work is the useless argmax
#      eval), OR when the trainer has LEFT the queue and any checkpoint exists
#      (catches runs that top out at ckpt 7-9 — v2 silently ignored those and
#      eucrdv5 s1-3 never got evals).
#   2. Arm table carries rundir/exp/out explicitly (creg_van lives under a
#      creg_ prefix, not cregime_; v5b/v5nl are new OUT dirs).
#   3. eval_stoch.sbatch already has the newest-PPO-dir-by-mtime CK fix.
# SSH failures are logged and retried, never treated as "queue empty".
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

B=experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap
# arm|seed|rundir|exp|out|trainjob  — evals this watcher owns
ARMS="
rs2|1|cregime_rs2_s1|${B}_risk_rs|rs2|creg_rs2_s1
rs2|2|cregime_rs2_s2|${B}_risk_rs|rs2|creg_rs2_s2
rs2|3|cregime_rs2_s3|${B}_risk_rs|rs2|creg_rs2_s3
van|3|creg_van_s3|${B}|van|creg_van_s3
v5b|1|cregime_eucrdv5b_s1|${B}_eucrd_v5|eucrdv5b|creg_v5b_s1
v5b|2|cregime_eucrdv5b_s2|${B}_eucrd_v5|eucrdv5b|creg_v5b_s2
v5b|3|cregime_eucrdv5b_s3|${B}_eucrd_v5|eucrdv5b|creg_v5b_s3
v5nl|1|cregime_eucrdv5nl_s1|${B}_eucrd_v5_nolocalcf|eucrdv5nl|creg_v5nl_s1
v5nl|2|cregime_eucrdv5nl_s2|${B}_eucrd_v5_nolocalcf|eucrdv5nl|creg_v5nl_s2
v5nl|3|cregime_eucrdv5nl_s3|${B}_eucrd_v5_nolocalcf|eucrdv5nl|creg_v5nl_s3
v5cp|1|cregime_eucrdv5cp_s1|${B}_eucrd_v5_cap|eucrdv5cp|creg_v5cp_s1
v5cp|2|cregime_eucrdv5cp_s2|${B}_eucrd_v5_cap|eucrdv5cp|creg_v5cp_s2
v5cp|3|cregime_eucrdv5cp_s3|${B}_eucrd_v5_cap|eucrdv5cp|creg_v5cp_s3
v5nc|1|cregime_eucrdv5nc_s1|${B}_eucrd_v5_nolocalcf_cap|eucrdv5nc|creg_v5nc_s1
v5nc|2|cregime_eucrdv5nc_s2|${B}_eucrd_v5_nolocalcf_cap|eucrdv5nc|creg_v5nc_s2
v5nc|3|cregime_eucrdv5nc_s3|${B}_eucrd_v5_nolocalcf_cap|eucrdv5nc|creg_v5nc_s3
rws|1|rws_eucrd_s1|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws|rwsec|creg_rws_s1
rws|2|rws_eucrd_s2|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws|rwsec|creg_rws_s2
rws|3|rws_eucrd_s3|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws|rwsec|creg_rws_s3
rws|4|rws_eucrd_s4|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws|rwsec|creg_rws_s4
rws|5|rws_eucrd_s5|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws|rwsec|creg_rws_s5
rwkn|1|rws_kn_s1|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_kn|rwkn|creg_rwkn_s1
rwkn|2|rws_kn_s2|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_kn|rwkn|creg_rwkn_s2
rwkn|3|rws_kn_s3|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_kn|rwkn|creg_rwkn_s3
rwknv|1|rws_knv_s1|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_knv|rwknv|creg_rwknv_s1
rwknv|2|rws_knv_s2|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_knv|rwknv|creg_rwknv_s2
rwknv|3|rws_knv_s3|experiment_multi_5dc_carbon_v2_gdpd_rwtight_timecap_eucrd_rws_knv|rwknv|creg_rwknv_s3
"

stamp () { date +'%m-%d %H:%M'; }
log () { echo "[$(stamp)] [v3] $*" >> "$LOG"; }

log "master watcher v3 started (pid $$)"
while :; do
  rundirs=$(echo "$ARMS" | awk -F'|' 'NF{print $3}' | tr '\n' ' ')
  probe=$(timeout 50 ssh "$HOST" "
    echo QOK
    squeue --me -h -o '%j %t' 2>/dev/null | grep -E '^(creg_|evs_|ehc_)' | sort
    echo CKPTS
    for d in $rundirs; do
      hi=\$(ls -d $RUNS/\$d/multidc_gtrxl_training/PPO_*/checkpoint_* 2>/dev/null | grep -oE 'checkpoint_[0-9]+' | sort | tail -1)
      echo \"\$d \${hi:-none}\"
    done" 2>/dev/null)
  if ! echo "$probe" | grep -q "^QOK"; then
    log "ssh FAILED (cert expired or network) — retrying next round, nothing assumed"
    sleep 900; continue
  fi
  queue=$(echo "$probe" | sed -n '/^QOK$/,/^CKPTS$/p' | grep -vE "^(QOK|CKPTS)$" || true)
  nq=$(echo "$queue" | grep -c . || true)

  while IFS='|' read -r arm seed rundir exp out trainjob; do
    [ -z "$arm" ] && continue
    marker="$STATE/evs_${out}_s${seed}.submitted"
    [ -f "$marker" ] && continue
    ck=$(echo "$probe" | grep "^$rundir " | awk '{print $2}')
    in_queue=$(echo "$queue" | grep -c "^$trainjob " || true)
    fire=""
    if [ "$ck" = "checkpoint_000010" ]; then
      fire="ckpt10"
    elif [ "$in_queue" -eq 0 ] && [ -n "$ck" ] && [ "$ck" != "none" ]; then
      fire="trainer-gone ($ck)"
    fi
    if [ -n "$fire" ]; then
      sub=$(timeout 50 ssh "$HOST" "cd $ISB && sbatch --job-name=evs_${out}_s${seed} --export=ALL,EXP=$exp,OUT=$out,SEED=$seed,RUNDIR=$RUNS/$rundir eval_stoch.sbatch" 2>/dev/null)
      if echo "$sub" | grep -q "Submitted"; then
        touch "$marker"
        log "SUBMITTED stoch eval $out s$seed [$fire] ($sub)"
        if [ "$fire" = "ckpt10" ] && [ "$in_queue" -gt 0 ]; then
          log "cancelling parent trainer $trainjob (argmax tail is useless)"
          timeout 40 ssh "$HOST" "scancel --me -n $trainjob" 2>/dev/null
        fi
      else
        log "submit FAILED for $out s$seed — will retry"
      fi
    fi
  done <<< "$ARMS"

  missing=$(echo "$ARMS" | awk -F'|' 'NF{print $5"_s"$2}' | while read -r t; do [ -f "$STATE/evs_${t}.submitted" ] || echo -n "$t "; done)
  log "queue=$nq jobs | v3 evals pending submit: ${missing:-none}"

  if [ "$nq" -eq 0 ] && [ -z "$missing" ]; then
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
