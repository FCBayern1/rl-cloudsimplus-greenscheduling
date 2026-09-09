# 三格解析微考场 —— 手算预期(写于任何微考场结果之前,2026-09-01)

目的:把两个问题分开。

    A  REAL_TIME 仿真器是否正确结算
    B  curve planner 是否正确利用 REAL_TIME 曲线

## 构造

    单 DC,单作业,单涡轮 900,确定性方波风电,10 分钟行
    brown_carbon_factor 1.0,green_carbon_factor 0.0     (碳 = 棕电能量,便于手算)
    host_count 1,VM 4 小(2 PE)                          容量 8 PE
    作业 length 6,000,000,pes 2                          runtime = 6e6/(40000*0.5) = 300 s
    deadline 3000 s,arrival 0
    latest_start = 3000 − max(300+2, 602) = 2398 s
    max_episode_length 3600 s(6 个风电行)
    green_power_scale 1/1500,故 DC 见到的绿电瓦数 = kW * 1000 / 1500 = kW / 1.5

    风电(kW → 见到的 W)
      flat  全程 300 kW → 200.0 W
      step  行 0-1 为 30 kW → 20.0 W;行 2 起 900 kW → 600.0 W(t ≥ 1200 s)
      late  行 0-7 为 30 kW → 20.0 W;行 8 起 900 kW → 600.0 W(t ≥ 4800 s,超出 episode)

## 功率口径(规划器模型)

    static      332 W × (本 DC 主机数 / 全部主机数) = 332 W(单 DC)
    dynamic     p × dyn_per_pe × u = 2 × 2.5406 × 0.5 = 2.5406 W

**静态是单作业动态抽电的约 130 倍。**

## 三格手算预期

**格 1 flat(平坦风电)**

未来无更优窗口,任何起点的绿/棕划分相同。curve 必须立即执行,并与 nowait **逐位相同**
(carbon、green_used、brown_used、episode_length、start/finish 全部相等)。
若不相同,则规划器在无信息可用时仍产生了差异,属接线或决策规则缺陷。

**格 2 step(截止前明确升绿)**

t < 1200 时绿电 20 W < static 332 W,故 `gres = max(0, 20 − 332 − …) = 0`,作业全部走棕电。
t ≥ 1200 时绿电 600 W > static 332 W,`gres = 600 − 332 = 268 W > draw 2.54 W`,作业动态全绿。

规划器目标 J 只计作业自身动态功率,故它**应当**选择 s = 1200,预测碳从
`1.0 × 2.5406 W × 300 s` 降到 `0.0 × 2.5406 W × 300 s`,即预测改善 100%。

**真实碳账是否也改善,取决于仿真器如何结算静态功率**:

    若空闲主机断电(idle_host_power_down)且等待期间不计静态 → curve 应真实更低
    若等待期间仍计静态,则等待 1200 s 的额外静态棕电为 332 × 1200/3600 = 110.7 Wh,
    而节省的动态棕电仅 2.5406 × 300/3600 = 0.212 Wh,**净亏约 522 倍**

**这一格是判决关键:若规划器预测改善而真实恶化,即定位为规划器目标模型不完整,不是仿真器错误。**

**格 3 late(绿窗在 latest_start 之后)**

绿电在 t ≥ 4800 s 才出现,远超 latest_start 2398 s 与 episode 3600 s。
curve **不得**违规等待,应与 nowait 相同或在合法边界前释放,
`deadline_forced_count = 0`、`stale = 0`、`unplanned start = 0`。

## 恒等审计

把 curve 实际产生的执行日程离线重放,用与仿真器同口径的逐段功率/绿电积分计算碳,
必须与终端碳账闭合。不闭合则 A 侧存在问题。

## 判决规则(Codex 2026-09-01)

    微考场不过                          REAL_TIME/planner 接线存在 bug,+54.49% 判决降级为无效
    微考场通过但真实风上仍 +54.49%      REAL_TIME 有效,STOP 保持
    物理积分闭合但预测改善、真实恶化    规划器目标模型不完整,非仿真器错误

## 代码层已确立的先验(读源码所得,非推断)

`_costs_all` 的目标为

    draw = p × dyn_per_pe × u
    gres = max(0, green_view − static − occ × dyn_per_pe × u)
    J    = Σ_{[s, s+r)} [ cg·min(draw, gres) + cb·(draw − gres)⁺ ]

**静态功率从不计入 J,仅作为可用绿电的扣减项;积分区间恒为 r 步,与起点 s 无关。**
因此延迟在规划器自己的目标里代价为零,而延长 episode 的整机静态能耗对它完全不可见。
5-DC 配置下 static 110.7 W 对动态 2.54 W 为 44 倍,单 DC 微考场下为 130 倍。

格 2 用来端到端确认这一先验。
