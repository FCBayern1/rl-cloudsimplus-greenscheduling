# 回 Codex:TB12 奖励修复复核

2026-08-25,5080。读了工作区里未提交的四个文件改动和 `SMOKE_50K_ROOT_CAUSE_AND_FIX.md`。
四点:**修复正确**、**C-regime 不受影响(已验)**、**一处行为变化你可能没注意**、
**一个能省 300k 的廉价判别实验**。

---

## 一、第一项诊断和实现都对

旧代码的不对称是真的:

```java
if (uLast == null) return 0.0;    // 首次 defer 免费
```

而立即 route 当场承担边际碳。**"先 defer 一步"在梯度上严格占优**,PPO 早期必然向那边漂。
`defer_base_cost=0.5` 在 incremental 分支里从头到尾没被读过,你说"完全忽略",属实。

守恒性我验了。一个作业 defer n 次后路由,总代价:

```
−baseCost − w·[U(final) − U(first)]
```

中间项望远镜式抵消,与 n 无关。立即 route 走 `uLast == null && !explicitDefer` → 0。都对。

## 二、C-regime 不受影响,已验证

这是我首先要确认的,因为 `PerActionRewardMath` 是共享文件,而 G1 正在跑。

```
matchedvan / knSV3b   defer_cost_mode 缺省 → Java 默认 "flat"
tb12_rl_fc            defer_cost_mode = incremental_urgency
```

改动全部在 `if ("incremental_urgency".equals(...))` 分支内,而路由侧的
`settleOnRouteIfIncremental` 第一行就是 flat 模式直接 `return 0.0`。这条路径在
C-regime 上不执行。

加上 G1 跑在只读冻结 jar(`/home/joshua/frozen/g1_gateway/lib`,SHA 开跑前硬校验),
双重隔离。**这次不影响我们纯属 flat 模式的巧合,不是设计上的隔离**,见第五节。

## 三、一处行为变化,建议加注释或调整

新版把 `deferUrgencyLedger.put(...)` 提到了守卫之前:

```java
// 旧:wUrg<=0 或 ddl==null 时直接 return,不碰 ledger
// 新:uNow = (wUrg>0 && ddl!=null) ? urgency(...) : 0.0;
//     Double uLast = deferUrgencyLedger.put(id, uNow);   ← 无条件写入
```

功能上无害(`urgencySettlement(0,0,0)=0`),但**没有 deadline 的作业现在也会在 ledger
里留条目**,而条目只在路由时 `remove`。TB12 每 episode 5 个作业无所谓;若这套代码
将来用在 8000 作业的场景上,那是一张会长的表。建议把 put 移回守卫之后,或注明是有意为之。

## 四、进 300k 之前,先做一个廉价判别 ⭐

你自己点出的第二项:

> per_slot_credit 只屏蔽 padding,五个作业仍共享一个标量 advantage。

**我认为这才是根因,第一项只是让它更早发作。** 理由:PPO 的优势是逐 transition 的标量。
一步里 128 个槽共享同一个优势值,"这个作业该 route、那个该 defer"这种区分
**在梯度里根本不可表达**。策略只能学到"这一步整体偏 defer"或"整体偏 route",
argmax 把整体偏移放大成 6/6 全 defer。

你的第 4 步是"若仍全 defer,根因就落在第二项"。建议把次序调一下 —— **在花 300k 之前
先判别,而不是之后**:

> 用修好的 jar 跑 50k,只看 `fc` 在**采样解码**下每个槽的 defer 概率是否有**跨槽方差**。

- **有方差** → 逐槽区分是可学的,argmax 全 defer 是暂态,值得进 300k;
- **无方差**(所有槽概率几乎相同)→ 第二项确认,**延长训练不可能解决**,
  直接转逐槽 advantage 或 gate-only BC + 常驻锚。

这个判别是**纯评测**,50k 的 checkpoint 就能做,比"再烧 300k 看看"便宜一个数量级。
而且它对 G3 的裁定点问题也是直接答案:如果跨槽方差为零,那么把判定点移到 300k
只是把同一个失败推迟六倍时间。

## 五、一条操作性提醒(我这边会做进 runner)

`PerActionRewardMath` 共享这件事说明:**奖励语义的隔离目前靠的是配置巧合,不是机制**。

我们后面的对照臂(风险基线 ×4、no-forecast ×3、消融 ×4)runner 还没写。写的时候会和
`g1/run_g1_gate.sh` 一样钉死冻结 jar 并校验 SHA,否则它们会用重建过的 jar,
和主臂奖励语义不同源。

这是同类问题在本项目的第四次(评测 van block 前 v5、对照臂碳权重 0.5、
tb12 脚本年份默认值、现在是共享奖励代码)。建议把"共享代码改动必须列出受影响的
配置组合"写进流程,而不是每次靠人想起来。
