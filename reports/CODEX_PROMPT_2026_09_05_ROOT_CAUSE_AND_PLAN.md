# 致 Codex:Stage D 根因分析与下一轮修改方案(2026-09-05,PI 指示独立复查后提交裁定)

Stage D 已按你的裁定以 STOP_STAGE_D_CONTRACT 终止并归档(附录 G)。本文件是 PI 指示"不依赖裁定、自行重查"的结果:第 1 节是证据,第 2 节是根因,第 3 节是修改方案,第 4 节是执行顺序与门,第 5 节是请你裁定的问题。所有断言都标了来源;假设与事实分开写。

## 1. 证据(全部为本次会话核实的代码或数据)

**E1 判定性分解(零训练,确认集 18 格,定义先冻结于 `HZ_DECOMPOSITION_DIAGNOSTIC.md`)。**

| 臂 | 信息 | 延迟 | 池化碳强度 |
|---|---|---|---|
| B 冻结盲臂 reactive_wait | 只看当前 | 反应式 | 4.110e-11 |
| S 真值规划器 + `PLANNER_ALLOW_DEFER=0` | 真实未来 | 禁止 | 4.226e-11 |
| ST 真值规划器 | 真实未来 | 允许 | 2.494e-11 |

空间可捕获量 −1.16e-12(S 比 B 差 2.8%),时间增量 1.73e-11,空间份额 **−0.072**。规划器计数:ST 每格 1226 次延迟决策,S 0 次,B 241 次。三臂合同全绿。逐窗口份额 −0.50/+0.60/−0.53,逐格 −0.45…+0.23,池化结论稳定。**解析杠杆全部来自开工时刻。**

**E2 观测(`spaces_only` 直接读出冻结 V 块的全局观测空间)。** 18 个键。每作业:`batch_cloudlet_mi`、`batch_cloudlet_pes`,**没有**截止期、剩余时间、等待时长、是否已延迟、每作业时机提示。预报:每机房 4 个标量——`dc_future_short_mean/trend`(3 行 = 30 分钟)、`dc_future_long_mean`(144 行)、`dc_future_long_peak_timing`(最早峰值时刻)。作业 48 行、截止期余量数百行。`obs_v31_features`、`obs_v32_job_forecast` 在 HZ 与 Stage D 配置中均缺省 False。溯源:`gen_s2.py` 从 `config_C.yml` 的 `experiment_g1eval_matchedvan`(评测块,无观测开关)派生;V3.1 战役 26 个块全部 `obs_v31_features: true`,并在该特征下证得 oracle gap −21% @ 100% 完成(见 memory 记录)。

**E3 奖励(Java `MultiDatacenterSimulationCore.java`)。** 646–650 行:DEFER 期间无每动作奖励;692 行注释:延迟成本存在的理由是"so the argmax policy stops drifting to always defer";ledger-aligned 变体将 `defer_base_cost`、`defer_urgency_weight` 置 0;完成项(`per_action_completion_weight` 1.0)只在 ROUTE 支付;`global_reward_gamma` 0,准时率不入奖励;碳项为全机群逐步量。P0 真值表比较四个固定行为,检测不到延迟轴无梯度。

**E4 兜底(`PerActionRewardMath.deadlineForceLatestStart`)。** 强制条件 now + length/(mips·u) + slack ≥ deadline,`defer_deadline_slack_sec` = 0.0;强制作业派往边际碳最低且有空核的机房(`pickGreenestAvailableDc`)。E 线 36 格中 23 格迟到,**23 格全部 forced > 0**;另 10 格 forced > 0 但准时;N_E forced = 0。

**E5 EU-CRD(`crd_q_loss.py`)。** 2235 行注释:未训好集成的 ρ 放大 defer 信用→"不可恢复的 always-defer 盆地"(vanilla 1/10,v2 3/10);2270 行:不加帽 normalize_rho→defer 饱和 0.99。冻结责任块:`normalize_rho: true`、`reweight_warmup_calls: 450`、**无 `normalize_rho_cap`**。

**E6 种子 1 行为。** 延迟率 init→final:N_V 5.7→72.2%,V 7.8→3.8%,N_E 6.4→9.9%,E 6.8→95.6%;碳 0.0061–0.0067。V 相对 clean:shrink +1.1%,shuffle +3.4%,anti +2.5%。

## 2. 根因(按层级,上层不修则下层无意义)

**R1 时机变量不可观测(E2)。** 杠杆 100% 在开工时刻(E1),而策略看不见截止期与任何每作业时机量,预报只有 30 分钟/24 小时标量。策略被要求控制一个它观测不到的变量。这是配置派生遗漏,不是设计决定。

**R2 时机变量不计酬(E3)。** 等待零成本、准时不计分、等待期间零反馈,唯一信号是几十步后的全机群碳。延迟轴平坦,策略随机漂移(E6 的 25 倍延迟率差、5% 碳差)。

**R3 兜底把"等到底"变成迟到(E4)。** 零余量 + 争抢排队。任何落入高延迟角落的策略都会违约,而它在训练中从未见过迟到的代价。

**R4 EU-CRD 缺护栏,把策略推入 R2 造成的平坦盆地(E5)。** 机制假设:ρ≈1 的转移放大 1.16×;策略集中于 defer 后 `policy_self` 基线 ΔQ→0、ρ→0.05、w→0.06,defer 转移的优势(含兜底带来的负优势)被抹掉——进去被放大、出来被抹掉。**这是假设**,需按动作类型审计证实;N_E 免疫(无预报时 ΔQ(defer) 无系统性正号)与之一致。

**对门的解释。** 门 2 结构性不可通过:shrink 是容量幅度误差,不看也不用时机的策略天然免疫,E1 是零训练 18 格证据。门 1 因错误理由通过:S≈B 说明真值预报做站点选择价值≈0,V 对 shuffle/anti 敏感度仅 3.4%/2.5%,V 赢的是 N_V 的 72% 盲目延迟。E 违约 = R4 × R2 × R3。

**与上一轮裁定的差异。** 上轮框架是"修责任分配 vs 改动作参数化"。R1/R2 在两者之上:宏动作策略同样看不见截止期、同样等待不计酬。E1 回答了"门 2 非结构性"的保留。

## 3. 修改方案(每项:改什么 / 依据 / 风险 / 如何验证)

**M1 重开每作业时机观测。** `obs_v31_features: true`(截止期、剩余时间、等待时长、已延迟、全局已延迟量)。依据 E2 与 V3.1 战役。风险:观测维度变化,与旧检查点不兼容(本轮全部重训,无影响)。验证:`spaces_only` 读出新键;单步探针证实 `time_to_deadline` 逐作业不同且随时间递减。**是否同时开 `obs_v32_job_forecast`(每作业最佳开工时刻提示,需 `obs_v32_demand_model: job_counterfactual_v1`,否则在零地板下退化为常数)请裁定**:开则策略更容易学到时机,但那是把规划器的答案喂给它,论文主张会弱化;我倾向第一轮不开,只给原始信息。

**M2 让时机进奖励,二选一请裁定。** (a) 准时项:`global_reward_gamma > 0` 的 SLA 项或按迟到量的惩罚,与合同门一致;(b) 恢复延迟成本 `defer_base_cost`/`defer_urgency_weight`(Java 注释所述原设计)。依据 E3。风险:(a) 改变 P0 通过的奖励,需重做 P0;(b) 是启发式成本,方向易但幅度需标定。我倾向 (a):它奖励的是合同本身,不是行为。验证:**P0 扩展**——增加只在时机上不同的重放行为(立即开工 / 延迟到最佳绿窗 / 延迟到截止),要求奖励排序 = 物理碳排序 **且** 迟到行为的奖励严格更低。

**M3 兜底余量。** `defer_deadline_slack_sec` 从 0 改为覆盖一次排队等待的量(按机房排队时间分布的 p95 标定,或改为感知空核)。依据 E4。验证:always_defer 探针在新余量下准时率 ≥ 0.995。

**M4 EU-CRD 护栏。** `crd.responsibility.normalize_rho_cap` 设为有限值(代码注释建议区间 2–3 倍以下;具体值请裁定,建议 1.5)。依据 E5。验证:按动作类型审计中 DEFER/ROUTE 的 w 分布上界受限。

**M5 按动作类型信用审计(诊断,不改主实验)。** 用归档的 E/N_E 检查点离线重放,拆 DEFER/ROUTE 统计 ρ、w、优势前后、ΔQ、Δr、c_t、τ。证实或否定 R4 的棘轮假设。**在 M4 定值之前完成。**

**M6 动作参数化(暂缓)。** (DC, 开工偏移) 宏动作只在 M1–M3 之后仍学不到时机时才启动;否则改了也看不见截止期。

**M7 不改的:** 场景(HZ ×2 零地板)、作业、窗口、五个配对种子、四线结构、污染集与负控、判读器与合同规则、CCA-PG 与风险基线的定义。

## 4. 执行顺序与门(建议,待裁定后冻结为 Stage D′ 预注册)

1. M5 审计(1 天,零训练)→ 决定 M4 的 cap 值。
2. M1 + M2 + M3 配置改动 → **P0′**(扩展的奖励真值表,含时机行为)必须通过,否则不训。
3. 健康烟测(同 Stage D 标准)→ 四线 400k × 5 种子,两平台安排不变(工作站主判,Isambard 复现 + 基线表)。
4. 门 1–5 不变;新增 **门 0′**:V 线的延迟率与解析 ST 的延迟决策比例同量级(否则时机通道仍未打开,判 STOP 而非继续调)。
5. 全部先冻结、后运行;任何中途查看按附录 G 方式披露。

## 5. 请裁定

- Q1 M1 是否同时开 `obs_v32_job_forecast`(倾向否)。
- Q2 M2 选 (a) 准时进奖励 还是 (b) 恢复延迟成本(倾向 a)。
- Q3 M4 的 `normalize_rho_cap` 取值(倾向 1.5,待 M5 数据)。
- Q4 门 0′ 的量级阈值如何定(建议:V 延迟率 ∈ [0.3×, 3×] ST 的延迟比例)。
- Q5 上一轮"门 2 非结构性"的保留是否由 E1 撤销。

## 6. 文件指针

`reports/HZ_DECOMPOSITION_DIAGNOSTIC.md`、`g1/compressed_timecap_s2/stage_a_out/hz_decomp_m2.json`、`reports/CODEX_PROMPT_2026_09_05_SEED1_DIAGNOSIS.md` §7–8、`reports/STAGE_D_LONGRUN_PREREG.md` 附录 A–G、归档 `drl-manager/{logs,results}/stage_d_longrun*_STOPPED_CONTRACT`。当前 commit a3703679(修正:此前误写为 df2ce234)。两台机器空闲,无实验在跑,主实验代码未改。
