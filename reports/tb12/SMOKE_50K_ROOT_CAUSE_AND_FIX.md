# TB12 50k G3 失败：根因与最小修复

日期：2026-08-25。前置判决：`SMOKE_50K_VERDICT.md`。原 50k 结果维持
G3 FAIL，不追认、不续跑；本文定义一个新的修复候选。

## 根因

### A. 已确认的实现/规格脱节（本次修复）

`defer_cost_mode=incremental_urgency` 的生产分支此前在第一次 DEFER 时只
建立 urgency ledger，返回 0；距 latest-start 超过一小时的后续 DEFER 也因
`U=0` 返回 0。因而配置中的 `defer_base_cost=0.5` 在该模式下完全失效：

- route 立即承担 per-action 碳项；
- early defer 的即时回报为 0；
- 较差的最终物理碳只能经后续共享 episode return 回传。

这与生产代码旁“等待有一个 always-on opportunity cost”的注释和配置含义
相冲突，也直接解释了为什么修复后的总回报最终厌恶 always-defer，50k 的
策略梯度却仍可能先把 gate 推向 defer。

修复语义：第一次显式 DEFER 收一次 `-defer_base_cost`；后续 DEFER 与最终
route 只结算 `-w[U(now)-U(last)]`。因此总等待成本为
`-base-w[U(final)-U(first)]`，与作业被重新呈现多少次无关。立即 route 不收
base。

### B. 已确认的结构风险（本次不改）

Java 已输出 `per_slot_reward_csv`，但 Python learner 没有消费它。
`PerSlotCreditPPOTorchLearner` 当前只做两件事：屏蔽 padding slot，并把所有
有效 slot 的 log-prob 相加；所得联合 ratio 仍乘同一个标量 GAE advantage。
它不是逐作业 advantage。

TB12 每集只有五个真实作业，且 temporal gate 参数共享。一次联合回报会把
多个作业的 gate 同向推动，能够产生本轮看到的“六个窗口整体跨过 0.5”现象。
这是下一候选修复，但不能与 A 同时落地，否则无法判断哪一项恢复了学习。

## 已落地的最小修复

- `PerActionRewardMath.firstDeferBaseCharge(...)`：生产公式；
- `MultiDatacenterSimulationCore`：用 GlobalBroker 的显式-defer 生命周期判定
  首次等待；route 侧不调用 base 公式；
- 两个回归测试：base 只收一次且 urgency 仍望远镜化；后续不重复收 base；
- `SimulationSettings` 文档口径同步。

urgency ledger 仍只接收“有 deadline 且 urgency 权重为正”的作业；无 deadline
作业不会因为一次 defer 留下额外 urgency 条目。base 的一次性由已有的
GlobalBroker defer ledger 保证。

目标测试与 `./gradlew -q compileJava` 均通过。

## 新实验的固定顺序

1. 旧 v2 checkpoint、判决和 jar 全部保留，不覆盖。
2. 用新 jar 重跑零训练四轨真值表；必须满足 reward/kg 同序、cap=0、
   always-defer 不夺冠、正确选择性等待仍优于 nowait，并新增哨兵：每个作业
   首次 DEFER 的 per-slot reward 精确包含 `-0.5`，重复 DEFER 不重复收 base。
3. 全过后才生成 append-only v3 block 和新 manifest；不得复用 v2 名称。
4. 新 50k 仍使用 argmax G3，同时前置报告 ck0/ck50 的 `p_hold` 分位数、
   logit margin、训练采样 defer 率和 defer/route advantage。概率在同一份
   冻结 observation corpus 上读取，只取每作业首次 eligible decision，避免
   不同 rollout 状态与重复 defer 造成伪样本。跨槽方差仅是必要条件；还要按
   冻结 teacher 标签报告：两类样本均非空、
   `mean p_hold(worth)-mean p_hold(not-worth) >= 0.05`（沿用 V3.2 Gate-2
   既有强响应线）、六个 offset 至少 4 个方向为正，以及 pooled AUC。AUC
   只作连续诊断，不另设事后阈值。这样不会把 slack/作业大小造成的无关方差
   当成预测学习。50k 只判健康；argmax G3 与上述方向门均过才准进入 300k。
5. 若 A 修复后仍全 defer，不调 base、不移动 G3、不延长到 300k；转入 B：
   让每个有效 slot 的 PPO ratio 使用对应的 per-slot return/advantage，或采用
   已认证老师的 gate-only BC + 常驻锚。二者需另立预注册。

T116+117 在整个修复开发阶段继续封存。

## 共享代码影响矩阵

- TB12 `defer_cost_mode=incremental_urgency`：行为改变，必须新 jar、新配置名、
  新预注册并重跑真值表。
- C-regime matched Vanilla / knSV3b：配置缺省为 `flat`，不进入修改分支；
  正在运行的 G1 另由冻结 jar 隔离。
- legacy `flat` 实验：行为不变。

以后共享 Java/Python 语义变更在冻结前必须列出“配置组合 × 是否进入修改
分支 × runner jar/hash”，不能只凭实验名称判断隔离。
