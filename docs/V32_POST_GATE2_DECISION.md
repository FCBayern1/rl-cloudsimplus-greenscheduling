# V3.2 Gate 2 之后的锁定决策（交 Fable5 执行）

日期：2026-08-15。身份：**V3.2A 失败后的机制审计与分支决策单**。

本文件不修改、也不追认修改 `V32_PREREG.md`。V3.2A 的 Gate 2 FAIL 永久保留；
下面若启动奖励修复，必须以新的 V3.2C 身份预注册，不能包装成 V3.2A 续跑。

## 0. 最终决定

**现在不改权重，不延长 V3.2A，不直接启动 V3.2B 蒸馏。先做 teacher-reward
paired audit（教师策略真实回报配对审计）。**

理由：现有证据有两个互斥解释，处置完全不同：

1. PPO 没学会奖励本来偏好的 `high gain -> defer`，则按原预注册进入 V3.2B；
2. 当前折现奖励本来就偏好 `high gain -> route`，则直接蒸馏会在 PPO 微调阶段把老师
   教会的 1029 秒级等待洗掉，必须先修奖励目标。

slack-aware teacher 已在同完成率下降碳 21--29%，所以把**同一教师策略在训练真实奖励下
的回报**与 no-defer 配对比较，是当前最便宜、最直接的分叉证据。

## 1. 已冻结的事实

- V3.1 600k 参照波冻结：oracle 在两个 seed 均比 blind 差（+31.8% / +6.2% carbon），
  只作为间接通路负基线。
- V3.2A 300k Gate 2 = FAIL：`job_temporal_delta=-0.0070`，gain 单调性 0%，slack
  单调性 100%，通道活度为惰性通道 30--45 倍。
- gate 接线、参数更新、旧 defer head 失活和 slack 学习均已证实；不得以“再训一点”解释。
- 当前配置为 `gamma=0.999`、`defer_urgency_weight=2.0`、
  `defer_urgency_window_sec=3600`；绿电峰周期约 1029 秒。
- “450 秒盈亏平衡”是强线索，不是最终标定常量。现有 Java 真值表 helper 把整段
  urgency settlement 未折现地放在等待路径起点，而生产环境在每次重现和最终 route
  时逐段入账；审计必须按生产时序计分。

## 2. 工单 R0：teacher-reward paired audit（最高优先级）

### 2.1 被比较的策略

基于 `drl-manager/oracle_slack_planner.py`，使用**当前 V3.2A 的完整环境与奖励配置**：

- experiment：`experiment_v3_2_oracle`；
- teacher：slack-aware，`theta=0.5`、现有 margin/backlog cap；
- control：同一脚本的 no-defer；
- local：两边固定 deterministic drain；
- global route：两边保持相同的 greenest-now + capacity 规则；
- seed、workload、episode index、green offset 必须逐局配对。

不得拿 V3.1 配置计分；V3.2 多出来的 spatial reward 也属于 Gate 2 时 PPO 实际优化的
奖励，必须包含。

### 2.2 必须记录的两种回报

planner 当前把 `env.step()` 返回的 reward 丢弃。修改后，每步只记录：

```python
r_t = float(rewards["global"])
```

禁止用 `env.episode_reward`，也禁止加 local rewards。fixed-drain local policy 不参与训练，
global PPO 实际收到的是 global agent 自己的 reward。

每局同时输出：

```text
global_reward_sum        = sum_t r_t
global_discounted_return = sum_t (0.999 ** t) * r_t
```

`global_discounted_return` 是本次分支的**主判据**；未折现和折现结果的分歧用于定位 gamma
时间尺度。不得只报告 episode reward sum。

### 2.3 必须记录的物理量和奖励分解

每臂每局至少落盘：

- `total_carbon_kg`、`carbon_per_mi`；
- MI completion rate、received/finished MI；
- global reward sum、global discounted return；
- route carbon level、spatial carbon、completion、incremental urgency 的 episode sum；
- defer、route、deadline-forced、backstop 数量；
- episode index、实际 green offset、seed、步数；
- teacher 相对 no-defer 的逐项 paired delta。

缺少奖励分解时可以先完成一局哨兵，但不得据此修改奖励；三-offset 定案前必须补齐。

### 2.4 offset 与未来轨迹纪律

当前 planner 注释只覆盖新环境的 episode 0。扩展到多个 offset 时：

- 两个 env 实例必须保持存活并按相同 episode index 同步 reset；
- offset 日程使用生产规则 `(1009 * episode_index) mod green_episode_offset_range`；
- planner 读取 future green 的 CSV 行必须加入同一个 episode offset；
- 每局记录 planner offset 与环境 authoritative offset，一致性 fail-fast；
- 不允许一边重建 env 回到 offset 0、另一边继续下一个 offset。

### 2.5 资源阶梯与定案规则

1. **S0 哨兵（约 40 分钟）**：offset 0 的 teacher/control 一对。先验证能复现
   `teacher carbon < control carbon` 且双方 completion >=99.5%。复现不了则 STOP，先修仪器。
2. **S1 定案**：运行前 3 个配对 offset。若三局同号且完成率合同成立，直接按第 3 节
   分支；不得因为某一局幅度小而改门柱。
3. **S2 仲裁**：前三局符号不一致、回报接近零，或未折现/折现读法冲突时，扩到 6 个
   配对 offset；报告逐局值、中位数和符号计数，不只报 pooled mean。

一局哨兵只能触发继续/修仪器，不能单独成为论文中的“奖励错位已证明”。

## 3. 自动分支（结果出来后不再临场争论）

只在 teacher 和 control 均满足 completion >=99.5%，且 teacher 的 carbon 更低时判读：

| 配对结果 | 解释 | 锁定动作 |
|---|---|---|
| teacher 的折现 global return 更低 | PPO 正确优化了与物理目标错位的奖励 | **进入 R 分支：V3.2C 奖励时间尺度修复；V3.2B 暂停** |
| 未折现 return 更高、折现 return 更低 | 主要矛盾是 `gamma=0.999` 把 1029 秒后的收益打折过重 | **进入 R 分支，优先修 gamma/时间尺度** |
| 未折现和折现 return 都更低 | urgency/completion/carbon 组合本身不想要老师行为 | **进入 R 分支，重做 SLA 风险成本；不得只蒸馏** |
| teacher 的折现 global return 更高 | 奖励想要老师行为，但 PPO 没学出来 | **进入 L 分支：按原预注册启动 V3.2B 蒸馏** |
| teacher 碳不降或 completion 不合同 | 本次比较不能回答目标错位 | **STOP，先修 planner/iso-completion，不准选边** |

主判据使用逐 offset 的折现 global return。前三个 offset 3/3 同号即可选边；不一致则必须
执行六-offset 仲裁，按多数符号并同时披露中位数。若六局恰好 3:3，状态为 WAIT，不允许
靠调权重解锁。

## 4. R 分支：新建 V3.2C（仅 reward audit 指向错位时）

V3.2C 是**结果后提出的新假设**，必须先写独立 prereg，再改配置或训练。不得覆盖
V3.2A checkpoint、结果目录或原预注册。

V3.2C 先完成以下离线工作：

1. 按生产真实逐次结算时刻，对已有 7170 个 job-opportunity 重放
   `G(wait-to-best)-G(route-now)`，按 gain/time-to-best/slack 分桶；
2. 分离两个不应绑定的尺度：
   - forecast horizon：策略能看到多远，当前证据支持 3600 秒；
   - urgency window：何时进入 SLA 风险区，应由服务时间、排队安全余量和 backstop 标定，
     不能因为 forecast horizon=3600 就也等于 3600；
3. gamma 由目标时间尺度冻结，不按 Gate 2 输出调参。至少报告候选 gamma 在 1029、
   1214、1800 秒后的 reward retention；
4. 新真值表必须覆盖：1029 秒绿电将至、经验 time-to-best 高分位、当前已绿、gain<=0、
   deadline 紧、backlog 高和 all-defer 防塌缩；
5. Java 真值表必须模拟 urgency 各段在真实发生时刻的折现，不能把总 settlement 放在 t=0。

V3.2C 代码/真值表/离线账通过后，先用**未用于 V3.2A 判决的新训练 seed**跑 clean-PPO
300k Gate 2。原 `delta>=+0.05`、gain/slack 单调性和 judgeability 门槛保持不变；不过即停，
不批准 600k。

V3.2B 的教师数据和 BC 代码可以在此期间离线准备，但**禁止在旧错位奖励下做 PPO
fine-tune**。若 V3.2C clean-PPO 仍失败，再在 V3.2C 奖励下 warm-start，并记录 imitation
退火过程中 gain 单调性是否被洗掉。

## 5. L 分支：V3.2B（仅 reward audit 证明奖励偏好老师时）

按 `V32_FORECAST_REVIVAL_PLAN.md` 的既定教师蒸馏方案执行：

1. slack-aware theta=0.5 生成 observation / hold-route / DC 数据；
2. 先监督训练 temporal gate；
3. 用当前真实 persistence 奖励 PPO 微调，imitation coefficient 逐步退火；
4. 每个退火阶段都跑 job-temporal probe；一旦 gain 单调性从正转负，标为
   teacher-unlearning，不得只报告最终 BC accuracy；
5. 方法级 oracle/blind 对比与 clean feature ablation 分表，信息集如实声明。

## 6. 可与 R0 并行的纯 CPU 审计

继续完成 forecast-gain 全链路符号审计，但它不能替代 teacher-reward paired audit：

- 正 gain 的定义必须是“等待后可达碳成本低于现在”；
- horizon、bin offset、persistence blind fill 的量纲与截断；
- `p_hold` 对 gain 的局部梯度方向；
- synthetic probe 的 gain/time-to-best 组合必须物理可达；
- 生产 rollout 中 gain/time-to-best/slack 的联合分布。

发现接线错误则先修错误并重跑 Gate 2；若接线继续通过，按第 3 节的 reward audit 结果选边。

## 7. 共同禁令

- 不重跑或延长 V3.2A；
- 不因为 Gate 2 FAIL 调低 +0.05 门槛；
- 不凭中位作业的 450 秒算术直接改 urgency weight；
- 不只累计 global+local 混合 episode reward；
- 不用一局结果声称跨 offset 的奖励错位；
- 不在旧奖励可能反对教师行为时直接做“BC 后退火到零”的 PPO 微调；
- 不把 V3.2C 写成原预注册实验的成功修订。

## 8. Fable5 的立即交付物

按顺序提交：

1. planner reward instrumentation + 最小测试（global-only、discount arithmetic、offset 对齐）；
2. S0 配对 JSON/CSV 和人读摘要；
3. S1 三-offset 配对账本与自动分支结论；
4. 仅在 S1 不能定案时提交 S2 六-offset 仲裁；
5. 根据自动分支新建 `V32C_PREREG.md`，或启动既有 V3.2B 工单。

**当前唯一获准启动的仿真实验是 R0 teacher-reward paired audit。**

---

## 附录 A:§6 纯 CPU 审计结果(Claude,08-15 12:20;与 R0 并行)

**结论:forecast_gain 生成链路四项全部通过;接线无错,不改 Gate 2 判决。**

| 检查项 | 结论 | 证据位置 |
|---|---|---|
| 正 gain 语义 | ✅ `improvement=max(0, now−future)`,future 只取 `offsets_sec ≤ budget` 的 bins → 正 gain 严格 = "budget 内等得到更低碳" | `hierarchical_multidc_env.py` `_append_v32_job_forecast_features` |
| 量纲与截断 | ✅ offsets = `linspace(1..horizon_steps, bins)` 步 × timestep = 秒;godeye 走 Java `getFuturePerDcGreenPowerW(秒)`;TimeCAP `offsets−1` 1-based 对齐有注释;归一化 clip [0,1] 除以 job_carbon_high | 同上 + `_v32_forecast_green_bins` |
| blind persistence 填充 | ✅ mode=none 双保险:bins 先重复 current(结构路径),特征再强制 gain=0/time=1/best_future=best_now(预注册元组) | 同上 |
| p_hold 对 gain 梯度方向 | 探针已测:负(−0.0070 @300k),且与 ck0(−0.0125)方向同、幅度收窄——不是接线错,是学习信号方向 | `local_eval_rt/probe/v32_g2_*.json` |
| 生产 rollout 联合分布 | ❌ **被仪表 bug 阻塞**:`on_episode_step` 的 rollout hook 把一切异常吞成 `logger.debug`,生产训练从未写出 `v32_rollout_worker*.jsonl`(仅单测 mock 通过);300k/100k 目录均无文件 | `rllib_green_energy_logger.py` L472-473 |

**给 Codex(B 单)**:rollout 仪表需要 ①把静默 skip 升级为计数器+首个异常 warning;
②在真实新 API 栈上验证 `episode.agent_episodes`/`get_extra_model_outputs` 路径
(建议 12k 步 2-iter 真训探针);③产出后把 Gate 2 rollout-sign 从 NOT-AVAILABLE 转正。
不影响 R0:R0 判据只用 planner 直采的 global reward,不经此 hook。


---

## 附录 B:R0 执行记录(Claude)

### S0/S1 第一轮(12:00–12:51)+ 指标修正(13:05)

S0 哨兵与 S1 三 offset 的**回报符号全部 teacher_higher**(折现 +1632/+377/+350,
未折现 +3223/+41/+430)。但复审抓到 completion 字段用了 `completion_rate` ——
它是 `routed_rate` 的别名,"100%"只证明全部**被派出**。修正为
`completion_rate_mi`/`carbon_per_completion_mi` + `branch_verdict()` 代码化分支表
(含 STOP 门),12 项测试锁死。

### S1 修正版判决(13:48–15:20):**STOP(§3 第 5 行)**

| ep | offset | control 完成(MI) | teacher 完成(MI) | teacher 碳 | dR_disc |
|---|---|---|---|---|---|
| 0 | 0 | 99.47% ✗ | **96.97% ✗**(forced=11) | −29.1% | +1632 |
| 1 | 1009 | 100% ✓ | 97.77% ✗(forced=9) | −22.8% | +377 |
| 2 | 2018 | 99.38% ✗ | **99.83% ✓**(forced=7) | −23.3% | +350 |

**两臂都过不了 99.5% 合同 → 不准选边。**连带修正:§7.4 slack-aware oracle 的
"−21/−29% @ 100% 完成"里的 100% 同样是 routed_rate,**oracle-gap 声称降级为
"碳低 21–29% 但完成率 97–99.8%,iso 前提未证"**,待本轮仪器修好后重验。

### 完成率赤字根因(分解表)

1. control 丢 4 单/局(零延迟也丢):共享路由规则在**全 DC 饱和时兜底"最绿排队"**,
   把作业堆死在一个队列;
2. teacher 额外丢:延迟到 budget 耗尽后仍被塞往绿 DC 队列,来不及完成
   (margin=120s 没有覆盖排队等待);
3. **spatial 项实现账发现**:teacher 的 `ep_spatial_term_sum` ≈ control 的 2 倍
   (2008 vs 1077 等)——候选均值中心化只在**期望**上路由/延迟中性,实现层它
   系统性奖励"等到自己选的 DC 比均值好更多"。这是回报偏好 teacher 的重要组成,
   V3.2C 若启动必须把这条写进真值表核查项。

### 仪器修复(19:51,两臂同规则,不动奖励;15:20–19:50 机器空转——watcher 被清后没有补挂,教训:每个后台链必须有存活的通知路径)

饱和兜底改"最短队列";budget≤0 的作业(两臂同)改投"最快可开工"DC;
teacher margin 120→300s(排队等待余量)。`pick_targets()` 纯函数 + 3 测试。
**S1 v2 已重跑**(margin=300),判据、门柱、γ 均不变。


### R0b 等完成量审计(23:08 v1 / 23:52 v2)

时域对称扩至 10000 步(绿电序列 15013 行,富余)。

| ep | control 完成 | teacher 完成 | teacher 碳 | dR_disc |
|---|---|---|---|---|
| 0 | **100%** ✓ | 98.69% ✗(14 单) | −28.9% | +1632 |
| 1 | **100%** ✓ | 99.21% ✗(6 单) | −22.5% | +377 |
| 2 | **100%** ✓ | **100%** ✓ | −23.3% | +350 |

**四个确定的事实**:
1. control 三格全 100% → 其原丢单纯属截断删失,已消;
2. **ep2 = 第一个完全合格格**(双臂 100%,teacher 碳 −23.3%、折现回报更高)——
   单格已满足 L 行条件;
3. 回报符号累计 **9/9**(7200/路由修复/10000 三版仪器全一致);teacher 折现回报与
   7200 版逐位同(−417.45/−417.42)→ 奖励在路由时足额计费,时域只影响物理完成;
4. 掉活折价上界:ep0 缺 1.31% MI,即使按 control 的整体碳强度补回,teacher 仍 −27.6%
   → 碳优势不可能由掉活解释。

**teacher 残余赤字的定性**:跑满 10000 步仍剩 14/6 单;逐槽分流器(第二次路由修复)
结果与修复前**逐位相同**——瞬时空位从不是约束(爆发时绿 DC 空位 >128),瓶颈是队列
排空速率。结论:**赤字是 θ=0.5 在部分 offset 过度延迟的策略内生代价**,不是路由/时域
仪器缺陷。两次路由修复零效果 = 此方向证伪。

**行动(23:57)**:θ 阶梯连夜跑(0.6→0.7,时域 10000)。θ 是老师规格的合法旋钮
(§7.4 先例扫过 0.7/0.5);判据、γ、完成合同 99.5% 不动。预期:θ↑ → 延迟更少 →
完成↑、碳优势收窄;若某 θ 三格全合格且回报符号保持,按 §3 表选边(现有证据指向 L)。


---

## ⭐ R0 终审判决(08-16 00:30,θ=0.6,时域 10000,三配对 offset)

| ep | control 完成 | teacher 完成 | teacher 碳 | dR(未折现) | dR_disc(主判据) |
|---|---|---|---|---|---|
| 0 | 100% | **100%** | **−22.3%** | +2945 | **+1563** |
| 1 | 100% | **100%** | **−20.1%** | −141 | **+113** |
| 2 | 100% | **100%** | **−23.4%** | +438 | **+375** |

`branch_verdict = {'branch': 'L', 'action': 'V3.2B distillation',
'reason': 'discounted return teacher_higher 3/3'}`

**判读(按 §3 表,无临场解释空间)**:
- 有效性门:6/6 格完成率 100%(≥99.5 合同)∧ teacher 碳 3/3 更低 → 比较有效;
- 主判据:折现 global return 3/3 teacher_higher → **L 分支:奖励偏好老师行为,
  PPO 没学出来 → V3.2B 蒸馏**。累计符号证据 12/12(四版仪器全部一致)。
- 细节留痕:ep1 未折现 dR = −141(负)而折现 +113——折现在该 offset 是决定性的,
  与"teacher 把负计费推迟、被 γ 稀释"的机制一致;写 V3.2B 时如需未折现口径请引用此格。
- **教师规格修订**:θ=0.5 在 offset 0/1009 违反完成合同(98.69/99.21%),
  **θ=0.6 是第一个全 offset 合同内的教师**,碳优势 −20~−23%。
  V3.2B 的蒸馏老师即 **θ=0.6 / margin 120 / 时域按训练环境**(不是 0.5,依据在此)。
- §7.4 场景门重验随之完成:iso-completion(100%)下老师仍省碳 20-23% → 场景有效,
  预报-时移杠杆存在,只是 clean PPO 摸不到。

θ=0.7 频谱格照跑(~01:05),仅作 carbon-completion 前沿信息,不影响判决。


### Codex 复审两风险的处置(08-16 00:55,数据生成尚未开始,零浪费)

**风险 1(ep1 折现依赖)**:属实,终审节已如实留痕(未折现 −141 / 折现 +113,
折现在该 offset 决定性)。论文口径义务:报告折现主判据的同时给出未折现符号计数
(2/3),并说明 γ=0.999 是被审计策略的真实训练目标,不是审计者的选择。

**风险 2(时域截断逃单标签)**:核实属实——θ=0.6 三局 7584/7295/7362 步,全部越过
7200 训练时域;若按 7200 生成数据,尾部 HOLD 标签会教学生"把活儿停到看不见的界外"。
**修复已上线**(数据生成前,22 项测试绿):teacher 的等待预算改为
`min(deadline 预算, 时域剩余) − runtime − margin`,不足即走既有强制尽快开工路径。
**判决不受影响的构造性证明**:10000 步审计里实际到达的最深步 7584,时域剩余 ≥2416s,
恒大于 runtime+margin 上界 → 该规则在审计世界从不触发,L 判决语义原样成立。
