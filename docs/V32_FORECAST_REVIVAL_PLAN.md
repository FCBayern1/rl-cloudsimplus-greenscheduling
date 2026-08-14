# V3.2 预报复活方案（交给 Fable5 复核）

状态：设计提案，尚未实现、尚未验证。日期：2026-08-14。

背景证据总账：`docs/V3_FORECAST_DIAGNOSIS.md`。V3.1 的修复状态与判据：
`docs/V31_WORK_ORDERS.md`、`docs/V31_PREREG.md`。

> **先划清证据边界**：本文新发现的 actor 数据流是代码级事实；“它是低绝对预报响应的
> 主因”是与历次干预结果高度吻合的机制解释，仍需 raw-logit 探针和 V3.2 消融确认。
> V3.2 的 factorized gate、候选中心化和 BC warm-start 都是待验证修复，不是既成结论。

---

## 1. 结论先行

当前不应继续优先调 PPO 变体、GAE λ 或碳权重。应保留 V3.1 已完成的环境/奖励公平性
修复，另开 **V3.2**，把策略拆成：

```text
作业 slack/age/backlog + 作业对齐的预测收益
                    │
                    ▼
          Temporal Gate: hold / route
                    │ route
                    ▼
      当前碳、容量、队列 ──> 8-DC Spatial Router
```

Local 继续 deterministic drain，不加入碳奖励或预测。系统只能有一个时间控制器。

V3.2 的存在性目标不是“探针刚好正号”，而是同时满足：

1. 预测对 raw temporal logit 有直接且可测的影响；
2. `P(defer)` 对预测收益和 slack 呈正确单调关系；
3. iso-completion 下 oracle 相对 blind 的 carbon/MI 改善超过 13%；
4. anti-forecast 会让 oracle 碳排恶化至少 10%。

---

## 2. 新发现：当前 defer head 没有直接读取预报

### 2.1 代码级事实

全局观测中的预测键均以 `dc_` 开头：

```text
dc_future_short_mean
dc_future_short_trend
dc_future_long_mean
dc_future_long_peak_timing
```

定义位置：
`drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py:616-635`。

`ScoreBasedGlobalRLModule` 按前缀分桶：

```text
dc_*              -> per_dc
batch_cloudlet_*  -> per_cloudlet
其他               -> context
```

实现：`drl-manager/src/models/rlmodule_gtrxl_models.py:1137-1179`。

actor 前向计算为：

```python
q = cloudlet_emb + ctx_to_cloudlet(ctx_features)
k = dc_emb       + ctx_to_dc(ctx_features)

route_logits = q @ k
defer_logit  = defer_head(q)
```

实现：`drl-manager/src/models/rlmodule_gtrxl_models.py:1323-1336`。

因此 raw `defer_logit` 只直接读取作业特征和非 `dc_*` context；四个预测键只进入
`dc_emb/k`，直接作用于 8 个 route logits。当前结构中，预测改变 defer 概率的唯一途径是
改变 softmax 分母：

\[
P(defer_i)=\frac{e^{d_i}}
{e^{d_i}+\sum_j e^{z_{ij}(forecast)}}.
\]

也就是说：

- 只改预测时，raw `defer_logit` 理论上应逐位不变；
- 预测只能通过 route logits 间接挤压 `P(defer)`；
- 某个“未来更绿”的 route logit 上升时，`P(defer)` 反而下降，天然容易出现时间反号；
- route 分支一旦不再读取预测，时间响应也会随 softmax 间接通路一起消失。

### 2.2 为什么它解释了完整实验时间线

| 阶段 | 观测结果 | 与当前数据流的对应解释 |
|---|---|---|
| 旧 oracle (`actual`) | forecast TV 0.38–0.49，但时间符号稳定为负 | 泄漏窗口价把未来绿电直接写进当步 route 奖励，预测强烈推动“现在 route”；route logit 上升从 softmax 中挤掉 defer |
| oracle_sp (`persistence`) | forecast/control 0.074–0.078，时间差约 +0.0024，接近 blind | 一步的 route 奖励捷径被拆掉；真正路径变成 defer→等待→未来 route→节碳，而 temporal head 没有直接预测输入 |
| V3.1 300k | 两种子、7 个保留 ck 的时间符号均正，但绝对差仅 +0.0018…+0.0088 | slack/age/backlog 进入 `q` 后，head 知道“能否等”；它仍不知道“未来是否值得等”，只能走 route-softmax 间接通路 |
| V3.1 control/forecast TV | control 0.003–0.009；forecast 0.008–0.013 | centered z-score 又把 route 空间信号压小约 70 倍，唯一的间接 temporal 通路也随之被压小 |

这比“PPO 不喜欢预测”更精确：V3.1 修了训练条件，但没有给 temporal actuator 建立预测
直连。

### 2.3 per-slot credit 仍只修了 padding

`PerSlotCreditPPOTorchLearner` 排除了约 124 个 padding 槽，解决了 joint PPO ratio 被
padding 推入 clipping 的问题；但约 4 个真实槽仍共享同一个 timestep advantage：
`drl-manager/src/learners/per_slot_credit_loss.py:82-130`。

它尚未消费 Java 的 per-slot reward，也没有把“某个作业延迟后在未来省下的碳”专门归还
给该作业的 temporal decision。因此 actor 直连是第一优先，作业级信用仍是后续第二优先。

---

## 3. 为什么不能给 local 同时加碳奖励和预测

当前 V3.1 是 Architecture B：

| 控制器 | 唯一职责 |
|---|---|
| global | 每个作业 `defer/route`，route 时选择 DC |
| local | 作业到 DC 后立即尽可能执行；不拥有碳时间权 |

训练时 `fixed_local_scheduler=drain` 只训练 `global_policy`：
`drl-manager/src/training/train_rlmodule_gtrxl.py:94-99,707-714`。执行时 wrapper 把所有 local
动作覆盖为 live mask 的最大合法值：
`drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_pettingzoo.py:582-613`。

若解除 drain 并开启 local 碳控制，同一个任务会有两层独立等待：global 可留在全局队列，
route 后 local 又可留在 DC 队列。这会重新引入已由 drainfix 证实足以改变判决的
co-learning 混杂、双重延迟信用和非平稳性。

代码中的 `reward_local_carbon_enabled` 是另一套 Architecture A，不是给 Architecture B
追加 20% 奖励的普通开关。它给 local 注入 `green_now/short/long forecast`，并把奖励改为：

```text
local reward = green_coef * green_fraction
             + completion_coef * log1p(completed_now)
```

位置：
`hierarchical_multidc_env.py:779-789,1572-1588`、
`MultiDatacenterSimulationCore.java:2121-2151`。

这套现成奖励也不能直接复活预测：hold 得 0；棕电执行虽无 green reward，却仍有正的
completion reward，仍可能严格优于等待。若将来试 Architecture A，必须关闭 global defer，
让 local 成为唯一 temporal controller，并重做 deadline/slack 观测与真实增量碳奖励。
它只能作为独立消融，不能混入 V3.2 主线。

---

## 4. V3.2A：干净的 factorized PPO（主线）

### 4.1 作业对齐的预测特征

四个 DC 级摘要保留给 spatial router；另外基于**预测值而非真实未来**，为每个 batch
槽位生成：

```text
batch_cloudlet_forecast_gain
batch_cloudlet_time_to_best_green
batch_cloudlet_best_now_carbon
batch_cloudlet_best_future_carbon
```

核心定义：

\[
forecastGain_i = C_{best-now,i}
- \min_{\tau \le slack_i}\hat C_{best,i}(\tau)
\]

\[
relativeTime_i = timeToBestGreen_i / \max(slack_i, \epsilon).
\]

这些量只用当前状态、作业 slack 和预测轨迹，合法且不偷看未来。因为键名为
`batch_cloudlet_*`，它们直接进入每个作业的 `q_i`。full/none 两臂维度必须相同；blind
填预注册的中性值（例如 gain=0，relativeTime=1），不能删键。

建议优先直接从现有 raw forecast provider 取 12–20 bin，再按每个作业 slack 截断。
只用固定 240/1000 行摘要会继续让网络自己反推“该作业的 slack 窗内最低点”。

### 4.2 Factorized action distribution

对每个真实作业槽位：

```python
gate_input_i = concat(
    cloudlet/slack/age/defer features,
    global backlog features,
    job-aligned forecast features,
)
p_hold_i = sigmoid(temporal_gate(gate_input_i))

route_prob_i = softmax(spatial_score_i)
P(defer_i)   = p_hold_i
P(dc_j | i)  = (1 - p_hold_i) * route_prob_i[j]
```

对 RLlib 仍可返回 9 项 categorical logits：

```python
logp_defer   = log(p_hold)
logp_route_j = log1p(-p_hold) + log_softmax(route_logits)[j]
```

这 9 项已经归一化；Java、action mask 和 evaluator 不需要改变。实现必须配置门控，例如：

```yaml
factorized_temporal_gate:
  enabled: false       # 默认关，旧 checkpoint/旧实验字节不变
  job_aligned_forecast: false
```

V3.2 模板才显式打开。不要直接替换旧 `defer_head` 路径。

### 4.3 空间奖励改为候选中心化

V3.1 的单一全局 σ 同时承担 route-vs-defer 与 DC-vs-DC，已经实测把空间控制通道压至
0.003–0.009。V3.2 应把 temporal 和 spatial 尺度拆开。空间项使用同一作业所有可行 DC
的当前 persistence marginal carbon 作为控制变量：

\[
r_{spatial}(dc_j) =
-w_c\frac{C_j-mean_{k\in feasible}(C_k)}{\sigma_{spatial}}.
\]

该式不看未来、两臂完全同账、减掉公共现货波动，并保留候选之间真正有用的差异。
raw/current candidate costs、中心化后数值、clip rate 必须记录。

第一版不通过盲调 `w_carbon` 补偿尺度；先标定候选差值本身的分布，再冻结一个共同 artifact。

### 4.4 暂不先动的项目

- local 继续 fixed drain，不加碳奖励/预测；
- PPO、GAE λ=0.98 先不换；
- persistence、no_offset、incremental urgency、deadline backstop 保持；
- critic 是否试 λ=0.99，仍只由 defer 条件 TD residual 决定；
- 不把未来预测直接写进环境 reward，避免 oracle/blind 再次成为两本账。

---

## 5. V3.2B：教师蒸馏兜底（仅当干净 PPO 失败）

仿真内 slack-aware godeye planner 已在 100% 完成率下降碳 21.1%，θ=0.5 时下降 29.1%。
它证明场景有足够物理杠杆，也提供现成教师。

若 V3.2A 在 direct-edge 已验证后仍学不出明显 temporal response：

1. 用 planner 生成 `(observation, hold/route, dc)`；
2. 先监督训练 temporal gate；
3. PPO 用真实 persistence 物理奖励微调；
4. imitation coefficient 逐步退火到 0；
5. 最终仍用 iso-completion 碳和 anti-forecast 判决，不以 imitation accuracy 代替结果。

为保持比较诚实，应把 V3.2B 声明为一个完整的 forecast-enabled method，而不是继续声称
“两臂只差裸 observation”。若需要 blind 对照，应给 blind 使用只基于当前 CI/slack 的
reactive teacher，明确比较的是两个信息集下的完整方法。V3.2A 的 clean feature ablation
与 V3.2B 的方法级对比必须分表。

---

## 6. 先于任何训练的测试与探针

### 6.1 当前模型 raw-logit 定案（零仿真）

扩展 `probe_forecast_sensitivity.py`，同时输出：

```text
Δ raw defer logit
Δ route logits
Δ P(defer)
```

只改预测、其他输入逐字节相同时，旧 score-based 模型预期：

```text
Δ raw defer logit == 0
Δ route logits     != 0（训练后视 checkpoint 而定）
Δ P(defer)          仅来自 softmax 分母
```

这一步验证“没有 direct edge”，不需要 Java 或 GPU。

### 6.2 V3.2 梯度连通测试

必须验证：

```text
∂ temporal_logit / ∂ forecast_gain != 0
∂ temporal_logit / ∂ slack         != 0
```

同时验证旧 gate 关闭时 forward 与已有 checkpoint 行为不变。

### 6.3 分布数学与 mask 测试

- 9 项概率逐槽和为 1；
- padding 槽仍由现有 mask 排除；
- defer 被 mask 时 8 个 route 概率重新归一；
- 全部 route 被 mask 的非法状态 fail-fast；
- 极端 `p_hold≈0/1` 不产生 NaN/Inf；
- checkpoint save/restore 后输出一致。

### 6.4 行为真值表

| 场景 | 期望 temporal 行为 |
|---|---|
| 当前棕、绿电将至、slack 足 | hold 上升 |
| 当前绿、未来变差 | route 上升 |
| deadline 紧 | route 上升 |
| backlog 超阈值 | route 上升 |
| forecast gain≤0 | 不因预测增加 hold |

训练 rollout 还必须记录 forecast-gain/slack 分桶的真实 defer 率、raw gate logit、defer 次数、
forced-route 和 resolution carbon；不能只依赖合成观测探针。

---

## 7. 预注册式实验阶梯

### Gate 0：实验完整性

1. 修复 CLI seed 只写 `seed.txt`、未进入 `PPOConfig` 的接线；
2. smoke 的 `result.json.config.seed` 必须明确为 1/2，而不是 `null`；
3. oracle/blind 同 seed、同机器、同 offset 日程；
4. preflight 确认 V3.2A 两臂只差 `forecast_mode`/由此生成的中性预测值。

### Gate 1：代码级 direct edge

- 旧 checkpoint 的 raw defer logit 对预测不变（诊断成立）；
- V3.2 gate 的 raw temporal logit 对预测有非零梯度；
- full/none 维度、参数量和 mask 语义一致。

### Gate 2：100k oracle seed1 行为冒烟

建议把“明显使用”预注册为比 V3.1 更强的门：

- temporal 差 `P(defer|将至)-P(defer|将去) > 0`；
- synthetic 与真实 rollout 同号；
- 绝对差建议至少 0.05，或在训练前根据 planner 行为冻结等价阈值；
- forecast perturbation 显著高于 null；
- `P(defer)` 对 forecast gain 单调增、对 slack 紧迫度单调减。

不过门不延长到 600k、不调权重，直接检查实现或进入 V3.2B。

### Gate 3：300k 双真实 seed

- 两 seed、保留的所有后段 checkpoint 同号；
- critic 的 defer 条件 TD residual 不得长期大于 route 的 3 倍；
- 无 all-defer、backstop 主导或完成率塌缩。

### Gate 4：配对 oracle/blind 物理评测

- argmax、local drain、同 seed/offset；
- 完成率均 ≥99.5%；
- oracle carbon/MI 低于 blind 最好合格 checkpoint 超过 13%；
- defer 差来自 forecast-gain/slack 相关行为，而非平均无差别多 defer。

### Gate 5：机制

仅 P3 通过后跑 anti-forecast：oracle 碳恶化 ≥10% 且仍满足完成率合同，才叫“预测载重”。

只有 Gate 1–3 通过才批准 600k 全量。

---

## 8. 当前 V3.1 fullwave 的处置建议

当前 V3.1 600k 链条不能作为正式认证，至少有两项独立原因：

1. `entrypoint_rlmodule_gtrxl.py` 在 driver 调 `set_seed(seed)` 并写 `seed.txt`，但没有把
   seed 传进 `PPOConfig.debugging(seed=...)`；已完成 300k 和当前 600k 的
   `result.json.config.seed` 均为 `null`，不满足预注册的同 seed 配对；
2. 当前 temporal head 没有预测 direct edge，正在测试的是一个结构受限的策略。

可保留当前 oracle checkpoint 作为 raw-logit/诊断样本，但不建议继续消耗完整四臂
fullwave 并把它称为认证。是否停止现有进程属于运行队列决策，本文不擅自操作。

---

## 9. 建议拆给实现 agent 的工单

### 工单 A：seed 与 raw-logit 诊断

- seed 传入 RLlib config，补 result-config 回归测试；
- 探针输出 raw defer/route logits；
- 用旧/V3.1 checkpoint 产出 direct-edge 证据。

### 工单 B：作业对齐预测特征

- 从预测轨迹按每槽 slack 生成 gain/time-to-best/current/future cost；
- full/none 中性填充与观测边界测试；
- 不改奖励、不读真实未来。

### 工单 C：factorized temporal gate

- 配置门控、默认关；
- 9 项联合分布、mask、数值稳定、checkpoint 测试；
- raw-logit 梯度测试。

### 工单 D：候选中心化空间奖励

- 当前 persistence 候选均值控制变量；
- 两臂共同标定 artifact；
- raw/centered/clip 仪表与 Java truth-table/compile。

### 工单 E：V3.2 preflight、probe 与预注册

- 两臂 diff 白名单；
- rollout 单调性仪表；
- Gate 0–5 自动判决；
- 保留全部 checkpoint 或每 checkpoint 立即产出 probe JSON。

### 工单 F（条件工单）：BC warm-start

- 仅 V3.2A Gate 2/3 失败时启动；
- forecast planner 与 reactive blind teacher 分开；
- imitation 退火、PPO 微调、方法级对照单独报告。

---

## 10. 请 Fable5 重点复核的问题

1. `dc_* -> k`、`defer_head(q)` 是否确实意味着 raw defer logit 对预测严格不变；是否存在
   本文漏掉的 hidden/context 数据通路？
2. 用 `log(p_hold)` 与 `log(1-p_hold)+log_softmax(route)` 拼成 9-way categorical，
   在 RLlib MultiDiscrete/per-slot learner/action mask 下是否有遗漏？
3. job-aligned forecast gain 是否严格只用预测和当前状态，是否存在尺度或时间单位错配？
4. 候选中心化是否应只作用于 spatial route reward；temporal advantage 应如何保持物理账本
   一致且避免再次把预测写进奖励？
5. V3.2A 的 clean feature ablation 与 V3.2B 的方法级蒸馏对比，论文声明边界是否足够诚实？
6. 当前 V3.1 fullwave 是否还有完成一对 s1 的诊断价值，还是应在 seed/direct-edge 修复前止损？

最终原则：**让预测直接控制唯一的时间执行器，用物理碳收益和反预测扰动证明它载重；
不要再用“观测里有预测”“探针略微正号”或“训练回报提高”替代这个证明。**

---

## 11. Fable5 复核结果(08-14 下午,逐条)

**核心发现:✅ 代码级+实证双重确认。**分桶(L1137-1179:dc_*→per_dc)与前向
(L1323-1336:`q=cloudlet+ctx`,`defer=defer_head(q)`,forecast 只进 k)逐行核过;
raw-logit 实证:只改预报时 |Δ defer logit| 比 |Δ route logits| 低约三个数量级
(v31 ck5: 2.4e-5 vs 0.081;旧 oracle ck10: 3.3e-3 vs 7.56)。Gate 1 的"诊断成立"半边
已闭合。残差非严格零,量级在 float 噪声水平,不影响判决,工单 A 顺手查一下来源。

**Q1(隐藏通路)**:无。GTrXL 只吃 context 流,context 按构造排除 dc_*;q 的成分 =
batch_cloudlet + ctx;唯一交叉是常量 `_dc_scale/_cloudlet_scale` 与标量温度。确认。

**Q2(9 项拼接)**:数学成立(softmax 对归一化 log-prob 不变,熵/KL 不受影响)。
三个实现注意:p_hold 需 clamp [ε,1−ε] 防 log(0);defer 被 mask 时 8 路重归一
(§6.3 已列);per-slot learner 的 ratio 计算不依赖 logits 是否归一,无遗漏。

**Q3(forecast-gain 合法性)**:合法(只用预报轨迹+当前状态+slack)。一个必须记档的
实现假设:未来 greenRatio 需要 demand 估计(与标定件同一循环),用 persistence demand
并写进 artifact;单位一致的充分条件是所有 C 都走同一 effFactor 公式。盲臂中性值
(gain=0, relativeTime=1)预注册。

**Q4(⚠️ 本文档的一个真空洞,已有修法)**:§4.3 纯候选中心化会**杀死时间层的奖励信号**
——中心化后"现在 route"与"等到全局更绿再 route"的空间项都是零均值,gate 没有等待的
奖励理由。修法 = 双尺度拆分:`r_route = w_s·centered_spatial(σ_spatial)
+ w_t·zscore_level(σ_level=标定件) + completion(no_offset)`——空间项管"选哪个 DC",
level 项保留 V3.1 的碳阈值语义管"现在值不值得跑"。工单 D 按此实现,真值表四行照跑。

**Q5(声明边界)**:如写。V3.2A 消融与 V3.2B 方法级对比分表,蒸馏臂如实声明信息集。

**Q6(fullwave 处置)**:**跑完,降级命名**。裁定理由:①seed 布线缺陷(已核:config.seed
=None)是全项目历史等同的——所有旧臂同一入口,配对有效性靠 offset 日程(逐局确定、
种子无关)+同机,协议未变,修复归工单 A、写进局限,不构成中止理由;②盲臂 600k 是
V3.2 Gate 4 直接复用的对照(盲臂无预报,结构限制不影响它);oracle 600k = 间接通路
天花板参照 + raw-logit 成熟样本;③机器夜间无他用,V3.2 实现不占机器。
正名:**"V3.1 间接通路参照波"**,非认证。若 P3 意外 ≥13% → 间接通路已够,故事更简单;
不过 → 它是 V3.2 的文档化阴性对照。
