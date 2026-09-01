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

---

# Addendum A —— 2026-09-01,冻结前的七项修正与正式 Round 0

依 Codex 2026-09-01 两轮复核。本 addendum 与主文冲突处**以 addendum 为准**。

## A.0 已隔离的 PREFREEZE_DIAGNOSTIC

冻结前曾用 DISCOVERY 涡轮跑过一次存活率预筛,带有事后选定的 ρ 截断 `[0.2, 3.0]`。
该进程已终止,**输出未读取**,全部归档于 `g1/tb13/prefreeze_diagnostic/`,
**不进入正式筛选**。其中发现并已修复的三处逻辑错误:取绝对值使强负相关通过、
"同时贫风"实测的是"至少一站无绿"、最优 DC 不变时写 `pass` 而未淘汰。

**正式 Round 0 不设任何 ρ 截断。**

## A.1 容量必须绑定

    每站可调度容量   16 PE = 2 台 8-PE VM(仿真器侧须机械复现这两台 VM)
    作业             4 PE → 占站 25%    8 PE → 占站 50%
    2 PE             占站 12.5%,保留为【流体负控】,pes_share < 0.25 故不得成为正式通过格

不得把 host 的 64 PE 当作调度容量。

## A.2 runtime 对齐注册规格

    一级精确筛   {6, 12} 行 = 1–2 小时
    二级扩展     {6, 12, 24} 行 = 1–4 小时

runtime 自冻结集合抽取,**不得为塞进视界而缩短**;arrival 与 deadline 按可达性生成。

## A.3 延迟预算按运行总量在线结算

    floor_if_wait = Σ_已派出 (s_i − a_i) + Σ_已到达仍 pending ((t+1) − a_j)
    floor_if_wait > B  ⟹  该作业必须立即派出

修复了一处 off-by-one(原实现允许总等待到达 `B+1`)。回归测试覆盖:两作业同时等待不超预算、
容量在线遵守、deadline 遵守、**未来突变不变性**(只改决策触及不到的尾部,决策必须逐位不变)。

`persistence` 与 `immediate_current_only` 行为等价(平坦未来下每个起点同价),已由测试断言。

`strongest_blind` 更名 `blind_class_diagnostic`,**不是可部署策略**(逐实例事后择臂),
仅作盲臂类的保守下包络诊断。正式最强盲由 `pooled_strongest()` 在 **DISCOVERY 上按池化碳冻结单一臂**,
任一实例履约失败即取消资格,**CONFIRMATION 不重选**。

## A.4 climatology 口径统一

`instance_gen._climatology()` 聚合站内**全部**涡轮(原实现只读 `ts[:1]`,双涡轮站低估约一半),
用同一 divisor、同一历史窗口,并**在此处且仅在此处**扣除静态,返回 **residual green**。
`causal_blinds.climatology()` 直接用该值,成本为

    cb · max(draw − clim_residual, 0) + cg · min(draw, clim_residual)

不再二次扣除静态。数值测试:双涡轮站等于两台之和;静态只被扣一次。

## A.5 主轴与动态功率更正

    dyn_w_per_pe = (214 − 51.4)/64 × 0.5 = 1.2703 W/PE
    ρ = concurrency × pes_per_job × dyn_w_per_pe / mean(max(G − P_static, 0))

主文中的 2.5406 是满载值,算术矛盾已消除。

## A.6 triplet × season 改为笛卡尔

6 个涡轮三元组 × 6 个季节窗口 = **36 层**,不再一对一配对,以区分涡轮效应与季节效应。

## A.7 正式 Round 0:物理单元去重全扫

预筛只依赖物理键,与 workload/budget 无关。唯一物理单元数:

    3 pes × 4 concurrency × 2 turbines/site × 5 divisor × 2 T × 6 triples × 6 seasons
    = 8,640 个物理单元

纯 NumPy,**不求解**,全部扫完并记录:

    rho_residual
    pes_share
    pairwise correlation      要求【正相关】0.70 ≤ r ≤ 0.95,不取绝对值
    simultaneous-poor fraction  三站【同时】低于冻结阈值的时刻占比,非退化(不得为 0 或 1)
    best-DC change fraction   ≥ 10%,否则淘汰(用 continue,不是 pass)

结果映射回 36 个 workload/budget 组合。

## A.8 固定求解预算:36 层 × 4 anchor

不写"存活多少跑多少",也不按文件顺序截断。

    每个 triplet × season 为一层,共 36 层
    每层内按【冻结 SHA】排序,选 4 个 anchor,最多 144 个 anchor
    每个 anchor 携带完整邻域:3 个相邻 installed_divisor × 全部 4 个 budget fraction

    seed 0 求解实例上限 = 144 × 3 × 4 = 1,728

如此每个候选天然带有 3 load × 4 budget 的完整邻域,不会因随机抽点而无法判断稳定区域。

**无 anchor 的层如实记录,不跨层找替补。**

## A.9 固定晋级顺序

    Round 0   8,640 个物理单元全扫,不求解
    Round 1   通过物理门的单元,层内 SHA 选 anchor,至多 1,728 个 seed 0 求解
    Round 2   OPTIMAL + EVPI ≥ 15% + 基础分布门 的【整个邻域】才补 seed 1/2
    Round 3   三 seed 后检查完整 EU-CRD 门与 §8 稳定区域

**不按 EVPI 排名截半。** 主文 §12 的"按 EVPI 排名前 50%"作废。
主文 §10 的"连续 200 格无通过即停"**作废** —— 搜索顺序未冻结,退化区排前会错误早停;
空手结论只能在跑完 Round 0 全部单元与全部 anchor 的 seed 0 之后宣布。

## A.10 机时(承诺预算,非存活率估算)

    最坏      1,728 × 30 s ≈ 14.4 CPU-hours 串行
    本机      每个 CP-SAT 用 4 线程,8 核最多并行 2 个 ⟹ 最坏约 7.2 小时墙钟
    实际      多数实例秒级完成,预计明显更短

平均机时的估算只能引用 DESIGN_PILOT 数据,且**不得改变 144 anchor 上限**。

## A.11 冻结哈希

    grid_hash                4a24e7f3e6d8ffdd
    axes combinations        8,640
    physical units (Round 0) 8,640
    workload/budget per unit 36
    seed 0 solve cap         1,728
