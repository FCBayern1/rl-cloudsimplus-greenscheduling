# V3.2 任务书 — Claude(工单 C → D → A-probe → preflight/prereg)

规格权威 = `docs/V32_FORECAST_REVIVAL_PLAN.md` + §11 复核修订。归属按既有文件所有权。

## 工单 C:factorized temporal gate(`rlmodule_gtrxl_models.py`,我的地盘)

1. `factorized_temporal_gate.enabled: false` 门控;默认关时前向与旧 checkpoint 逐位一致
   (回归测试)。
2. 开启时:`gate_input_i = concat(per_cloudlet_i 全量(含 v31 slack/age/deferred +
   B 单的 gain/relative_time), ctx_features)`;`p_hold_i = sigmoid(gate MLP)`,
   clamp [1e-6, 1−1e-6];输出 9 项归一化 log-prob:
   `logp_defer = log(p_hold)`,`logp_route_j = log1p(−p_hold) + log_softmax(q·k)_j`。
3. 测试:①梯度连通 `∂temporal_logit/∂forecast_gain ≠ 0`、`∂/∂slack ≠ 0`(Gate 1 右半);
   ②9 项和为 1;③defer 被 mask 时 8 路重归一;④p_hold→0/1 无 NaN;
   ⑤save/restore 一致;⑥默认关逐位回归。

## 工单 D:双尺度空间奖励(Java 我的区,按 §11 Q4 修订版,不是原 §4.3)

1. `per_action_spatial_center: none(默认)|candidate_mean`:
   `r_spatial = −w_s·(C_j − mean_{k∈feasible}C_k)/σ_spatial`(复用
   `pickGreenestAvailableDc` 已算的全候选);**level 项保留 V3.1 centered_zscore
   (σ_level=标定件)**——空间管选哪、level 管现在值不值得跑,两个 σ 两个权重。
2. `calibrate_reward_norm.py` 加 σ_spatial 标定(候选间差值分布);artifact 记账同规格。
3. 真值表 JUnit 四行在新组合下重跑 + 新增一行:同时刻两个 DC 候选、碳差一档,
   centered spatial 必须给出正确排序且幅度 ≥ level 噪声。
4. raw/centered/clip 仪表照 V3.1 模式。

## 工单 A-probe(探针是我建的)

`probe_forecast_sensitivity.py` 增加 `--raw-logits` 输出(Δ raw defer / Δ route 分开报),
把 08-14 的临时脚本固化;对参照波 600k checkpoint 出正式 direct-edge 证据 JSON。

## preflight / prereg(我的文件)

- preflight 加 V3.2 门:`factorized_temporal_gate` / `obs_v32_job_forecast` 两臂对称;
  中性填充值符合预注册清单;
- `docs/V32_PREREG.md`:Gate 0–5 判据固化(Δ≥0.05 或按 planner 行为训前冻结等价阈值、
  单调性判据、TD-residual 条款),V3.1 的 P4(anti-forecast ≥10%)原样继承。

## 时序与合并

参照波占机器到明早——所有工单纯 CPU 可测,不冲突。合并顺序:我先合(C/D/A-probe),
Codex rebase(B/E'/A-seed)。Gate 1(代码级)双边合入后即可判;Gate 2 冒烟等机器空。
