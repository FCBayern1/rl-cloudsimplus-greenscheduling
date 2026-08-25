#!/bin/bash
# TB12 50k smoke 启动器(Codex 硬阻塞#6:启动前强制核验冻结修复 jar)。
# 仅在 Codex 签发 50k 后运行。顺序:核验 -> fc smoke -> nofc smoke(同机配对,
# 一次一件事)。ck0 由 save_initial_checkpoint 开关在训练内固化。
set -euo pipefail
R=/home/joshua/rl-cloudsimplus-greenscheduling
cd "$R/drl-manager"
export EVAL_CONFIG_PATH=$R/config_C.yml
export GATEWAY_LIBS=$R/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1

# 硬阻塞#6:jar SHA + config hash 核验,不过立即退出
.venv/bin/python -c "from tb12_smoke_gate import verify_repair_jar; verify_repair_jar()"

for ARM in fc nofc; do
  echo "[SMOKE] tb12_rl_${ARM}_v2s50k train start $(date '+%m-%d %H:%M')"
  .venv/bin/python entrypoint_rlmodule_gtrxl.py \
    --config "$R/config_C.yml" \
    --experiment "experiment_tb12_rl_${ARM}_v2s50k"
  echo "[SMOKE] ${ARM} train exit rc=$? $(date '+%m-%d %H:%M')"
done
echo "SMOKE TRAIN DONE — 用 tb12_smoke_gate.py 对 ck0/ck50 跑四门"
