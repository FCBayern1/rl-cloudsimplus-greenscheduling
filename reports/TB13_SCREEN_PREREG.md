# TB13 场景筛选 —— 协议冻结(2026-09-01,写于任何正式扫描之前)

依 Codex 2026-09-01 裁定。本文件冻结搜索域、判据、数据分割与停止规则。冻结后不得因结果修改。

## 0. 证据层级(三层隔离)

**DESIGN_PILOT** —— 已发生,永久标记,**不进入任何正式 EVPI、稳定区或论文数字**。

    g1/tb13/pilot/design_pilot_axis.py      fe1c61b73abcfd93
    g1/tb13/pilot/design_pilot_axis.out     2cde39c28ff05994
    g1/tb13/pilot/design_pilot_timing.py    1fddf0b24d0b3d6a
    g1/tb13/pilot/design_pilot_timing.out   6fde4c68408767f1
    涡轮 T12/T36/T96,2021,offset 20000

用途仅限:确认搜索轴有辨识力、估算运行时间、划定可证明最优的规模上界。
**pilot 中出现的 EVPI 23.52%(demand/green 1.111)不进入任何正式判据。**

**DISCOVERY** —— 冻结后用于搜索、预算标定与候选排序:

    涡轮 2, 5, 7, 8, 13, 17, 25, 32, 55, 62, 68, 83, 86, 88, 99, 101, 104, 110,
         119, 122, 125, 128, 129, 132        年份 2021

**CONFIRMATION** —— 冻结后从未进入 TB13 求解器:

    涡轮 4, 20, 22, 23, 30, 31, 33, 41, 44, 48, 59, 66, 69, 70, 76, 87, 97, 115,
         120, 121, 124, 130, 133, 134        年份 2021 与 2020

分割由 `seed 20260901` 确定性洗牌产生,排除 C-regime 与 pilot 用过的 12/36/91/95/96,
仅取 2021 与 2020 均存在的涡轮。记录于 `g1/tb13/data_split.txt`。

**表述限制**:2020/2021 的原始风况此前被人工浏览过,故只能称
"prospectively held out from TB13 scheduling analysis",**不得声称全球从未查看**。

## 1. load/green 的物理映射(不得用任意缩放系数)

每格必须能由下列物理量机械还原:

    demand_W   = concurrency × pes_per_job × dyn_w_per_pe
    green_W    = n_turbines_per_site × mean_turbine_kW × 1000 / installed_divisor
    ratio      = demand_W / green_W

    dyn_w_per_pe        2.5406      实测,= (214 − 51.4)/64 × cpu_util(0.5),不可调
    pes_per_job         {2, 4, 8}   受 max_cloudlet_pes = 8 约束
    concurrency         {1, 2, 3, 5}
    n_turbines_per_site {1, 2}
    installed_divisor   {1500, 3000, 6000, 12000, 24000}

`installed_divisor` 的物理解释**先于搜索固定**:它是"一个风场供养多少个同规模数据中心"的
倒数,即本站在风场装机中的份额。**不允许为命中某个 ratio 而临时归一化。**

由上述离散取值机械生成 ratio,覆盖两侧退化区与内部区。pilot 显示 ratio ≈ 0.01–0.1 为退化区
(碳目标平坦、动作无差异),**该区必须保留在域内**,不得因 pilot 结果剔除。

## 2. 两级规模与视界

**第一级 机制精确筛**

    作业       8–12
    视界 T     36–48 行(6–8 小时)
    DC         3
    要求       CP-SAT 必须 OPTIMAL,FEASIBLE 记 UNRESOLVED,不进 PASS/FAIL

**第二级 扩展确认**

    作业       20–50
    视界 T     ≥ 注册 runtime + slack(runtime 1–4 h,slack 4–24 h ⟹ T ≥ 168 行)
    要求       允许预注册的 optimality gap,**不得冒充精确解**,报告 gap 与 bound

第一级不代表完整 TB13 规格(6–8 小时 < 24 小时 slack),仅用于机制判别与候选排序。

## 3. 求解器与 UNRESOLVED 的固定处置

    第一遍          time_limit 30 s,num_workers 4
    UNRESOLVED 后   仅对预注册候选,固定第二档 120 s;或减少作业数至第一级下界
    仍非 OPTIMAL    记 unresolved,不晋级、不判失败,**不得按结果临时加时**

高负载 UNRESOLVED **不算失败**,也**不得据此删掉该物理区域**。

## 4. 延迟预算标定

    per-job wait   ≤ 24 h(= 144 行),完整落在 TimeCAP 视界内
    B_mean, B_p95  在 DISCOVERY 上一次标定,使 wait 与 route 动作均非退化,随后冻结
    completion     ≥ 0.995
    ontime         ≥ 0.995

**不得在 CONFIRMATION 上重标。** 不用作业过期或人为 static 代理抬高决策边界。

## 5. 分布门(替代单一 p*)

    至少 20% 的暴露决策,其阈值落在 [0.3, 0.7]
    盲策略中 wait 与 route 各至少占 20%
    forecast 与 blind 至少 10% 的作业动作不同

**术语更正**:小负载下"最优解从不等待"是**碳目标平坦、动作无差异**,不是过期损失公式里的
`p* → 1`。本文件不使用后一表述。

## 6. EU-CRD 非退化门(四条同时满足)

    至少 10% 的路由决策中,最佳与次佳 DC 的成本差 ≥ 该作业全棕碳的 5%
    forecast 与 blind 的 DC 选择分歧 ≥ 10%
    归一化 Δr 的中位绝对值 ≥ 典型全局奖励量级的 5%
    三个 DC 均获得非零路由质量,不得退化为永远同一站

`Var(Δr) > 0` 单独不作数(浮点噪声即可通过)。

RL 样本量由**增加独立 episode** 获得,不得靠提高单 episode 并发获得。

## 7. 站点选择(用真实 SDWPF 组合,不人工混合共同分量)

    pairwise correlation   0.7–0.95
    同时贫风占比           非退化(不得为 0,亦不得为 1)
    最优 DC 变化时刻占比   ≥ 10%

## 8. 稳定区域判据(不追单点最大值)

    至少 3 个相邻负载/延迟预算格通过
    至少 3 组真实涡轮三元组方向一致
    效应跨季节存在,不依赖单一窗口

## 9. 通过门(第一级)

    exact EVPI = (C_strongest_blind − C_optimum) / C_strongest_blind  ≥ 15%
    状态 OPTIMAL
    §5 分布门全部满足
    §6 EU-CRD 四条全部满足
    §7 站点条件满足

后续阶梯:真机 dominance-safe ≥10% → TimeCAP clean 相对同构盲 ≥5% 且负控被破坏 → BC → RL。

## 10. 空手停止条件

    搜索组合数上限          见 §11
    连续 200 格无通过即停
    结论表述仅限:
      「在预注册的真实涡轮、作业尺度与延迟合同范围内,未发现同时满足预测价值
        与信用非退化门的稳定区域」
    **不得写「多 DC 预测价值不存在」**

筛选器已在已知正例上被证明能找到解(`g1/tb13/test_known_positive.py`,16 项),
故空手结论有意义。

## 11. 搜索组合数与机时

    pes_per_job          3
    concurrency          4
    n_turbines_per_site  2
    installed_divisor    5
    n_jobs               {8, 10, 12}                    3
    T                    {36, 48}                       2
    涡轮三元组           6(从 DISCOVERY 的 24 个中预注册抽取)
    每格随机到达/时长种子 3

    组合数 = 3 × 4 × 2 × 5 × 3 × 2 × 6 × 3 = 12,960

实测第一级单实例 0.06–0.14 s(OPTIMAL 区),UNRESOLVED 区触发 30 s 上限。
按 20% 的格落入 30 s 上限估算:

    12,960 × (0.8 × 0.15 s + 0.2 × 30 s) ≈ 12,960 × 6.12 s ≈ 22 CPU-hours
    4 路并行 ≈ 5.5 小时;successive-halving 预计砍掉 60–80%,实际约 1.5–2.5 小时

## 12. successive halving

    轮 1   全部 12,960 格,仅算 §7 站点条件与 ratio,**不求解**,淘汰不满足者
    轮 2   存活格求解第一级(30 s),按 exact EVPI 排序,保留前 50%
    轮 3   保留格补齐 §5 分布门与 §6 EU-CRD 门,不满足即淘汰
    轮 4   对通过格检查 §8 稳定区域(邻域 + 涡轮三元组 + 季节)

每轮判据在本文件中已冻结,不得按中间结果调整。
