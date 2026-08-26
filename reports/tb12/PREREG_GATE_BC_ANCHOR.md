# 预注册草案:gate-only BC 探针 + 常驻锚 PPO(待 Codex 签发)

日期:2026-08-26。上游裁定:v3 50k STOP 维持,断点 = actor credit(非场景、
非奖励);不延长 vanilla PPO,不仓促实现完整逐作业 GAE;优先 gate-only BC +
常驻锚。v3 作为"奖励正确但共享信用未传导"的干净负结果永久保留。
**全程禁止读取 T116+117。**

---

## Phase A:gate-only BC 探针(本文重点,请求签发)

回答:**架构在 TB12 上能不能表达并学会逐作业的等/不等选择。**
不回答"PPO 能否自行发现"——那是 Phase B 之后的事。

### A.1 为什么 TB12 的 gate-only 是结构性保证

tb12 块已开 `factorized_temporal_gate: true`
(`rlmodule_gtrxl_models.py`,V3.2 分支),前向为

```
gate_logit = temporal_gate(q)                        # 只依赖 cloudlet query q
P(defer)   = sigmoid(gate_logit)                     # 精确二元
P(dc_j)    = (1 − sigmoid(gate_logit)) · softmax(scores)_j
```

推论(两条,均为结构性而非约定):
1. **route 冻结**:route 的条件分布 `softmax(scores)` 完全不含 `temporal_gate`
   的参数 ⇒ 只训 `temporal_gate` 时空间 route 无法被改变。
   (Codex 原文允许一并训"必要的预测特征投影";**我们不采纳**——
   `cloudlet_encoder`/`ctx_to_cloudlet` 一旦可训,q 变化会经 `scores` 泄漏到
   route,route 冻结就退回成约定。改为严格 gate-only,把"表示层是否已携带
   预报"变成探针的**被测量**而不是被绕过的前提。)
2. **q 是常量**:q 不依赖 `temporal_gate` ⇒ 整个 BC 期 q 恒定,可一次性缓存;
   BC 退化为在冻结表示上训练一个 6,273 参数的 MLP(Linear 96→64,Tanh,
   Linear 64→1),秒级完成。

### A.2 对称性(唯一差异 = forecast_mode)

- **共用同一个 ck0**:v3 两臂 ck0 实测是**不同随机初始化**
  (global_policy module_state sha 8be9a01a… vs 03877bcf…),各用各的会引入
  第二个变量。两臂统一从 **fc 的 v3 ck0** 起。
- 共用同一套 **clair 标签**、同样的 steps / lr / batch / loss 权重 / seed。
- 驱动轨迹同为**冻结 greenfollow**(离线释放计划,不依赖 obs)⇒ 两臂动力学
  逐位相同,由 `assert_corpora_aligned`(obs 步数 / first_seen / 逐步 slotmap /
  分歧集四项逐位比对)在训练前哨兵核验。
- 唯一差异 = 各自 forecast_mode 下的 obs 内容 ⇒ q 不同。

### A.3 数据与标签

- 分布:**T100+101/2021**(训练/校准),6 个校准 offset。held-out 封存。
- 标签:**clair**(冻结,读当集完整未来风)在每个决策点的 gate 决定 ——
  `hold ⟺ clair_release[rank] ≥ t·600 + 600`(与 runner 量化语义同源,
  即 Codex ④ 修正后的语义)。
- 样本 = 每步每个有效槽一个决策点(实测每 offset 22–247 个)。

### A.4 判据(Codex 冻结,三条全过才准 Phase B)

| # | 判据 | 机械定义 |
|---|---|---|
| C1 | fc 方向 gap ≥ +0.05 且 ≥4/5 有效 offset 正向 | 复用 v3 已测过的 `movement_gate_verdict`:分歧作业首次 eligible 决策点,signed p_hold 移动(ck0→ck_bc,target=clair) |
| C2 | fc 明显优于 nofc | 池化移动差 ≥ **+0.02** **且** fc 拟合准确率 > nofc |
| C3 | 非全等、非全不等 | fc 的 gate 在全部决策点上 argmax 两类各占 ≥5% |

任一不过 ⇒ **断点在特征/gate 架构,停止 PPO 线,先修表示层**(Codex 原文)。

### A.5 一个必须报告的方法学风险(诚实标注)

样本量约 600–900 个决策点,gate 有 6,273 参数 ⇒ **可记忆**。C1/C2 是
in-sample 度量,"fc 拟合更好"可能只反映 fc 的 q 更可分,而非映射可泛化。
因此**额外报告留一偏移交叉验证**(每折在其余 5 个 offset 上训练,在留出
offset 的分歧作业上量方向移动):

- **非判据、不改变放行条件**(判据以 Codex 冻结版为准);
- 但若 in-sample 过而留一不过,应视为"可记忆、未必可泛化",
  建议 Codex 在签 Phase B 前把留一升为判据。裁定权在 Codex。

### A.6 冻结项

- jar / config:沿用 v3 冻结基线(jar 940078777d788d68…,source 13348ae5673d,
  config 2f31042f5e38adba…)。BC 探针不触碰 Java、不训练环境。
- corpus:`calib/tb12_bc_corpus_{fc,nofc}_v4.npz`(哈希见执行回报)。
  v3 方向门 corpus(4c9577eb…)保持封存不动。
- 超参:steps=2000, lr=1e-3, batch=256, seed=20260826(两臂逐位相同)。

---

## Phase B:PPO + 常驻锚(草案,BC 过后另行签发)

- 从 BC checkpoint **热启动**;
- 训练全程加 BCE/KL 模仿锚,权重**可退火但下限永不归零**
  (历史数据已证明锚归零后 PPO 会侵蚀预报映射);
- fc/nofc 两臂共用冻结 teacher 标签、优化预算、锚强度;唯一差异
  forecast_mode;**route 冻结**;
- 阶梯(Codex 冻结):
  - **50k**:方向映射仍在、无饱和坍缩、奖励与物理同向;
  - **300k**:fc 在物理碳上开始优于 nofc,SLA/cap 全过
    (**行为硬门在此恢复**,即 v3 降级的 argmax 坍缩门);
  - **600k**:固定末 checkpoint,不挑点;
  - 前述全过后,**首次**读取 **T116+117** 做正式判决。
- 若路线成功,论文口径为"**蒸馏增强的 PPO 使用预报**";若须证明纯 PPO
  自行学会,则第二顺位才实现真正的逐槽 advantage(需作业身份追踪、
  逐作业回报、逐槽 critic),另立预注册。

---

## 执行状态

Phase A 机具已实现并单测(`tb12_gate_bc.py` + 8 例纯函数测试:量化标签语义、
退化检测边界、四判据合成、含 v3 事故指纹回归与"nofc 同样拟合则不过")。
两臂 corpus 构建中。**待 Codex 签发后执行并回报。**
