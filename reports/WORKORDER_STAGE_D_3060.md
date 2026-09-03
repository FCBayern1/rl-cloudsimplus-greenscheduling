# 3060 工单:Stage D 四线 1-seed / 56k 健康烟测(2026-09-03)

> 2026-09-03 更新:健康烟测改在本机 RTX 5080 上执行(预注册附录 E),3060 继续 PARKED;本工单保留给长训阶段,命令不变。

身份:HEALTH_SMOKE。只回答"接线是否通、策略是否活",不产生任何效果宣称、不进主表。开跑前提:Codex 对 R-q(物理奖励变体)、R-s(开 50k)放行。GPU 到此才解封。

## 0. 环境

```
git pull                                   # 需要 >= 本工单所在 commit
cd cloudsimplus-gateway && ./gradlew -q test installDist && cd ..     # jar 必须重建:窗口白名单在 Java 侧
sha256sum cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar
cd drl-manager && .venv/bin/python -m pytest -q tests/test_episode_offset_allowlist.py tests/test_forecast_hollow.py tests/test_planner_static_env.py && cd ..
cd g1/compressed_timecap_s2 && ../../drl-manager/.venv/bin/python -m pytest -q test_gen_stage_d.py test_p0_verdict.py test_window_preflight.py && cd ../..
```

配置(已提交,不要重新生成):`g1/compressed_timecap_s2/config_stage_d_physical.yml`(四条训练线,56k = 7 次 PPO 更新 × 8000,每次更新存一个 checkpoint),`config_stage_d_eval.yml`(30 个部署评测块)。记录 jar SHA、两份配置 SHA、`stage_d_manifest_physical.json` 里的 CRD 子树 SHA(700f3da6…)到你的报告里。

## 1. 训练(四线,同一 seed,两两并行)

```
cd drl-manager
S=20260903
for L in NV V; do
  nohup .venv/bin/python entrypoint_rlmodule_gtrxl.py --config ../g1/compressed_timecap_s2/config_stage_d_physical.yml \
    --experiment sd_${L}_s2_r48_w72_c3_n35 --total-timesteps 56000 --num-workers 0 --seed $S \
    --output-dir logs/stage_d/${L}_s$S > logs/stage_d/${L}_s$S.log 2>&1 &
done; wait
for L in NE E; do (同上) ; done; wait
```

四线只差:experiment 名、`forecast_mode`(NV/NE = none)、`crd.enabled`(NE/E = true)。不要改任何超参。每线预期 7 个 checkpoint;把第 1 个(8k)和最后一个(56k)的路径写进报告。若某线崩溃,原样附日志,不要重试改参。

## 2. 部署评测(冻结末检查点,随机解码,认证窗 k=26/34/42)

对每条线的 56k 检查点,在六个评测格 × 3 个窗口上跑;V/E 跑四个 tier,NV/NE 只跑 hollow 块:

```
cd drl-manager
export EVAL_CONFIG_PATH=$PWD/../g1/compressed_timecap_s2/config_stage_d_eval.yml
export GATEWAY_LIBS=$PWD/../cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib
CELLS="s2_r48_w72_c1_n20 s2_r48_w72_c1_n50 s2_r48_w72_c3_n20 s2_r48_w72_c3_n50 s2_r48_w72_c5_n20 s2_r48_w72_c5_n50"
# L in {V,E}: TIERS="godeye calibrated_shrink_v1 shuffle anti";  L in {NV,NE}: TIERS="hollow"
for C in $CELLS; do for T in $TIERS; do for K in 26 34 42; do
  .venv/bin/python -m src.baselines.evaluate --experiment sde_${C}_${T} --global rllib --new-api --stochastic \
    --checkpoint <ck56k of line L> --local drain --episodes 1 --seed 42 --reset-skip $K \
    --output results/stage_d/${L}/${C}_${T}_k${K}.csv > results/stage_d/${L}/${C}_${T}_k${K}.log 2>&1
done; done; done
```

共 (6×4×3)×2 + (6×1×3)×2 = 180 次评测。每行结果必须含 `completion_rate_mi`、`ontime_mi_share`、`deadline_forced_count`、`total_carbon_kg`、`global_reward_sum`、`ep_carbon_norm_clip_count`。同样在 8k 检查点上只跑 godeye/hollow(六格 × 3 窗)供"奖励与碳同向"检查。

## 3. 探针

对 V 与 E 的 56k 检查点,用 `g1/compressed_timecap_s2/rl_step2_probe.py`(F2 pilot 的同一探针)在 c3_n50 / k=26 上输出:defer 率、clean 与 calibrated_shrink_v1 下动作边缘分布的 KL、控制通道敏感度;对 NE/E 另附训练日志里的 crd delta-r 均值/方差、critic disagreement、rho 分布(训练侧 result.json 里的 crd_* 字段)。

## 4. 交付

- `results/stage_d/`(全部 CSV/log)、四线 `logs/stage_d/*/result.json`、探针 JSON、检查点路径与 SHA。
- 一页报告:接线项(检查点存在可加载、tier 生效、hash 一致、字段齐全)逐项打勾;实质项(策略非全 route/全 defer、clean vs shrink 改变动作分布、V 有预报敏感度、E 的 delta-r 有方差、奖励与碳同向、合同全绿)只报数字,不下结论。判读由仓库里的 `stage_d_health_verdict.py` 机械执行。
- 不得根据本烟测调任何超参、换 trace、换窗口。

预算参考:F2 pilot 两臂并行 104k 步用了 8.25 小时;本工单四线 56k 约 8–9 小时训练 + 评测约 2 小时。
