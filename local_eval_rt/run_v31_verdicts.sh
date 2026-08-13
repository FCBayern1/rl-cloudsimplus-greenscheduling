#!/bin/bash
# Verdict aggregator: fires after drainfix (the last queued job) exits, parses
# all four verdict sources into ONE file, so the morning starts from a single
# document instead of four logs. Pure log/CSV parsing - no gateway, no sim.
set -uo pipefail
REPO=/home/joshua/rl-cloudsimplus-greenscheduling
R=$REPO/local_eval_rt
while pgrep -f "run_v3_sp_eval[.]sh|run_v3_track0[.]sh|run_v3_drainfix[.]sh|baselines[.]evaluate|oracle_hold_until_green|entrypoint_rlmodule" >/dev/null 2>&1; do
  sleep 300
done
sleep 60
cd $REPO/drl-manager
.venv/bin/python - <<'PY'
import csv, glob, re, statistics as st, pathlib
R = pathlib.Path("../local_eval_rt")
out = ["# V3.1 判决聚合(自动生成,勿手改;生成于 drainfix 结束后)", ""]

def section(title, path, pattern):
    out.append(f"## {title}"); out.append("```")
    try:
        for l in open(R/path):
            if re.search(pattern, l): out.append(l.rstrip())
    except FileNotFoundError:
        out.append(f"({path} 不存在)")
    out.append("```"); out.append("")

section("P3:oraclesp 评测(对照盲臂看 §2 表)", "v3_sp.txt", r"final oraclesp|EVAL DONE")
section("track0:heuristic 物理上限(≥15% 进赛道1;<10% 改场景)", "v3_track0.txt",
        r"RESULT|no-hold|hold-until|TOTAL|Δ|TIMEOUT|rc=|green is")
section("drainfix:旧判决去混杂(@DRAIN 对照 §2 的 rllib-local 数)", "v3_drainfix.txt",
        r"drainfix .*@DRAIN|SMOKE|TIMEOUT|DONE")

# sp_s2 vs nofc_s2 paired (same offset schedule -> pair by worker+episode)
def load(arm):
    D = {}
    for f in sorted(glob.glob(f"logs/{arm}/monitor_worker*.csv")):
        w = int(re.search(r"worker(\d+)", f)[1])
        for r in csv.DictReader(open(f)):
            try:
                D[(w, int(float(r["episode"])))] = (float(r["carbon_per_mi"]), float(r["green_ratio"]))
            except Exception:
                pass
    return D
A, B = load("v3_oraclesp_s2"), load("v3_nofc_s2")
keys = sorted(set(A) & set(B))
if keys:
    emax = max(k[1] for k in keys)
    tail = [k for k in keys if k[1] >= emax * 0.6]
    dc = [A[k][0] - B[k][0] for k in tail]
    worse = sum(1 for x in dc if x > 0)
    out += ["## sp_s2 逐局配对(判\"干扰项 vs 种子噪声\";s1 是 30/30 更差)", "```",
            f"配对 n={len(tail)}(末40%局)  Δ碳中位={st.median(dc):+.3e}  sp更差局数={worse}/{len(tail)}",
            "```", ""]

out += ["## 明早动作(照 V31_PREREG.md §6)",
        "1. 据上填开关决定表(drain 已无条件定死;真值表已用真实 μ/σ 复验通过);",
        "2. 翻模板开关 + 抄入 μ=3.524/σ=2.512(calib/experiment_v3_1_oracle_carbon_norm.json);",
        "3. preflight --v31-cert 过门;",
        "4. 队列确认空 → gradlew installDist(新奖励代码进 jar,只能在队列空时做)→ 武装 run_v31_smoke.sh。", ""]
pathlib.Path("../docs/V31_VERDICTS.md").write_text("\n".join(out))
print("verdicts -> docs/V31_VERDICTS.md")
PY
