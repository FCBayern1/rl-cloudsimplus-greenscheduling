# V3.2 任务书 — Codex(工单 B → E' → A-seed,按此顺序)

**规格权威 = `docs/V32_FORECAST_REVIVAL_PLAN.md`(§4.1/§6/§7/§9)+ 其 §11 Fable5 复核修订
(有冲突以 §11 为准)。**本文件只定归属/顺序/交付。禁令与 V3.1 任务书相同
(机器队列在跑参照波:禁评测/训练/网关/pkill;门控默认关;逐位回归锁;测试必配;
worktree 作业,Claude 先合你 rebase)。

## 工单 B:作业对齐预报特征 + 盲臂填充语义(最高优先)

1. 新观测键(`obs_v32_job_forecast: false` 门控,两臂对称):
   `batch_cloudlet_forecast_gain` / `batch_cloudlet_time_to_best_green` /
   `batch_cloudlet_best_now_carbon` / `batch_cloudlet_best_future_carbon`。
   计算:从预报 provider 取 12–20 bin 原始轨迹 → 按该作业 slack 截断 →
   `gain_i = bestNow_i − min_{τ≤slack_i} bestFuture_i(τ)`,
   `relative_time_i = timeToBest_i / max(slack_i, ε)`。
2. **信息集纪律**:各臂用**自己的**预报源(godeye=真值、blind=persistence 语义);
   未来 greenRatio 的 demand 假设 = persistence demand,写进 artifact(§11 Q3)。
3. **盲臂填充改 persistence 语义**(替换现行零填充,`hierarchical_multidc_env.py:1473`):
   `short/long_mean=当前绿电归一值、trend=0、peak_timing=0.5;gain=0、
   best_future=best_now、relative_time=1`。中性值即此,预注册,训练后不许调。
4. 边界:全部归一化+显式截断+声明 bound ≥ 实际 p99(obs 压扁旧坑);
   零填充→persistence 的回归测试:`obs_v32_job_forecast:false` 且旧 forecast_mode
   行为逐位不变。

## 工单 E':探针协议修订 + rollout 仪表

1. **探针阴性对照修订(§11 已预警)**:盲臂改 persistence 填充后会合法使用预报通道,
   "扰动后非零响应"不再等于泄漏。探针的扰动基线改为**各臂自己的填充语义**
   (blind 的"扰动"=偏离其 persistence 填充值),输出里标明基线类型。
2. rollout 仪表:按 forecast-gain/slack 分桶的真实 defer 率、raw gate logit、
   forced-route 计数、逐局落盘(不能只靠合成观测探针,§6.4)。
3. Gate 0–5 的自动判决行(照 §7 阈值,写死在脚本输出里)。

## 工单 A-seed(你名下的文件)

`train_rlmodule_gtrxl.py`:把 CLI seed 真正传进 `PPOConfig.debugging(seed=...)`;
补 result-config 回归测试(训练后 `result.json.config.seed == CLI 值`,用 mock/dry-run 验)。

## 交付口径

同 V3.1:分支名、测试清单+结果、每个新键的默认值证明、无法测项说明。
工单 B 完成先报告再进 E'。
