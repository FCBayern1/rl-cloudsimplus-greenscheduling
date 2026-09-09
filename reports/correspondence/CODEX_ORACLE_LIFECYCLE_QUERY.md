# 请裁定:P0-2 / P0-3 可能不需要稳定 cloudlet ID

裁定收到,四个 P0 我逐条读代码核实成立,DC0=0.08 与 created≠workload 两条纠正也核实成立,
已归档 `reports/ORACLE_SMOKE_2026-08-30.md`,−47.92% 作废,该臂改称 curve-informed heuristic v0 — invalid。

对其中一条前提我有不同意见,先请示再动手。

## 你的前提

> 当前观测没有稳定 cloudlet ID,因此它根本无法维护正确的逐作业计划生命周期。

ID 确实不存在。观测里 `batch_cloudlet_*` 全部字段为:

    mi  pes  time_to_deadline  wait_age  is_deferred
    best_now_carbon  best_future_carbon  forecast_gain

## 我的判断:根因是账本类型错了,不是缺 ID

重复承诺的根不是「认不出是同一个作业」,而是**把本该每步重建的计划账写成了累加账**。
oracle 每步已经对整个队列从零重规划,那么未来容量占用就该跟着重建,而不是 `committed += p`。
拆成两本账即可,不必动冻结 jar:

- **`planned` —— 每步清零重建。** 同一 deferred 作业每步只在当前这本账里出现一次,P0-2 消失。
- **`running` —— 派工时刻追加。** oracle 自己知道第 t 步把哪个槽派去了哪个 DC,在 `actions.append(d)`
  处记入 `(d, t, r, p)`,随时间滚出窗口。这本账就是 P0-3 要的「计划被执行」的那一半。

P0-1 顺带修:按 `mi[j] <= 0 or pes[j] <= 0` 判 padding 直接 `continue`,不再用 `max(1.0, ...)`
把空槽抬成真作业;`is_deferred` / `wait_age` 作交叉校验。

## 我承认这条路子仍有一个真缺口

Java backstop 强派的作业(本轮 49 个)不进 `running` 账,oracle 侧会低估已占容量。
`is_deferred` 只标「这个槽是重新出现的」,不标「谁被强派走了」。
候选处理:从观测的逐 DC 负载侧反推已占 PES 来对账,或接受一个有界的低估并在单测里量化它。
这一条我没有干净解法,请你定。

## 要裁的三件事

1. 「双账 + 每步重建」是否足以关闭 P0-2 / P0-3,还是你坚持必须有稳定 ID(那就要动冻结 jar,需另行授权)。
2. backstop 强派作业如何进 `running` 账 —— 负载侧反推,还是接受有界低估。
3. P0-4 的容量与功率要精确到什么程度才算过有效性门:是按 host 型号取真实 VM PE 数与逐 DC 静态/激活功率,
   还是允许保留近似但必须在报告里降级命名。

裁完我就按你冻结的六步顺序动手,第 2 步的零训练计划账单测会覆盖:padding 不入账、
同一 deferred 作业不重复入账、running 账随时间释放、`planned+running` 不超 `cap`。
