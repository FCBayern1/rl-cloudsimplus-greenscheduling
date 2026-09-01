# C-regime 物理时间基门 —— 终判(2026-09-01)

## 1. 一句话判决

在把风电时间基修正为实测的 10 分钟节拍后,注册的 **perfect-curve-informed planner** 未能击败冻结的
最强有效因果盲,其池化碳反而**高 54.49%**,三个窗口**均无优势**。按预注册分支,
**PHYSICAL-TIMEBASE CURVE-PLANNER GATE: STOP**。

## 2. 时间基缺陷与修复

旧配置 time_scaling_mode 为 COMPRESSED,使 wind_row = sim_step + offset,即 1 仿真秒 = 1 个 SDWPF 行
= 10 真实分钟。**只有风电时钟被加速 600 倍**,而 CloudSim 的 runtime、deadline、backstop 与能量积分
仍按秒运行。该映射与论文正文三处声称(SDWPF 为 10 分钟 SCADA、workload 运行于 7200 仿真秒、
控制间隔 1 仿真秒)不自洽。

旧考场按风电时钟读出的语义:episode 约 50 天、可等待窗口约 41 天、TimeCAP 视界约 24 小时。

修复使用已有 Java 机制,未改 Java 代码:

    time_scaling_mode        REAL_TIME     (rowSeconds = 600)
    green_interpolation_mode STEP          (行 i 在其 600 秒单元内恒定)
    green_power_scale        1/1500        (模式无关乘子;divisor 仅在 COMPRESSED 分支生效)

    wind_row(t) = offset + tz + floor((absolute_sim_clock - origin) / 600)

规划器不再持有硬编码 warmup = 13,改为从观测读取首次决策的真实时钟 clock0,与 Java 共享同一绝对
时钟映射。修复后:episode 为真实 2 小时,backstop 前置 600 秒为真实 10 分钟,TimeCAP 96 点历史为
16 小时、144 点预测约 24 小时。

## 3. 冻结窗口与 artifact

选择规则(先于结果冻结):候选量为三站实际分段时长加权总绿能;经验分位 p10/p50/p90;取距目标分位
最近的实际候选;tie-break 最小 offset;三窗逐涡轮读取区间互不重叠。首选即合法,未触发替换。

    name     k   offset   green Wh   rank pct   read rows
    low   2528    33552     332.60       10.0   [33552, 33626]
    mid   3279    27161    3033.92       50.0   [27161, 27235]
    high   869    22771    9832.38       90.0   [22771, 22845]

    selection hash  fa8a53115b82cfbb
    jar             6d23d8790d3a4d997eb5867c180c0030c5ced264b794d18e32b39ed10de261b5
    phys config     79f7e6fd9e36c3e9e447643cdcce24e9b9d25ad8dff5422c8edfaea88ddeb492
    一格读 21 行(12000 步),窗口占 129 行(21 + 最大时区偏移 108)

负控(不参与判决):最小绿能窗 k=43 offset=43387 green=0.00 Wh,机制中性,未运行。

## 4. 15 格完整性与合同

**14/15 格合同有效。** 不是「全部全绿」。

    cell                            carbon/MI    comp   ontime  clock0 shift rowsec  verdict
    climatology_high                 0.187331  1.0000   1.0000     1.0     0    600  PASS
    climatology_low                  0.327088  1.0000   1.0000     1.0     0    600  PASS
    climatology_mid                  0.013704  1.0000   1.0000     1.0     0    600  PASS
    curve_high                       0.073726  1.0000   1.0000     1.0     0    600  PASS
    curve_low                        0.053444  1.0000   1.0000     1.0     0    600  PASS
    curve_mid                        0.011814  1.0000   1.0000     1.0     0    600  PASS
    nowait_high                      0.031105  1.0000   1.0000     1.0     0    600  PASS
    nowait_low                       0.047080  1.0000   1.0000     1.0     0    600  PASS
    nowait_mid                       0.011776  1.0000   1.0000     1.0     0    600  PASS
    persistence_high                 0.074650  1.0000   1.0000     1.0     0    600  PASS
    persistence_low                  0.058255  1.0000   1.0000     1.0     0    600  PASS
    persistence_mid                  0.011477  1.0000   1.0000     1.0     0    600  PASS
    reactive_wait_high               0.031105  1.0000   1.0000     1.0     0    600  PASS
    reactive_wait_low                0.072875  1.0000   0.9558     1.0     0    600  FAIL  ontime_mi_share
    reactive_wait_mid                0.011776  1.0000   1.0000     1.0     0    600  PASS

逐格核对:forced、stale、unplanned start、wrong DC、dispatched-never-started、
running PE over cap、occ over cap 在 **15 格全部为零**;workload 均为 8000;无截断异常。
planner_startup_row_shift = 0(物理基要求),planner_row_seconds = 600。

**分段签名与 clock0 逐窗一致**:同一窗口下五臂的 planner_rows_signature 与 planner_clock0 完全相同
(low 2c528b6ad9764ea3 / mid 360e77142cba6072 / high a5534b399005e95f),
planner_rows_12000 显示每 DC 读 21 行,首行与冻结 artifact 的 offset 相符
(low 33552 / mid 27161 / high 22771,DC1 +18、DC2 +54 为时区偏移)。

主比较格 curve × {low, mid, high} 与 nowait × {low, mid, high} **六格全部合同有效**。

## 5. 最强有效盲的选择

reactive_wait 池化 0.038585 低于 persistence 与 climatology,但其 low 窗
ontime_mi_share = 0.9558 < 0.995,合同不过,**不得作为最强有效盲**。

四个盲臂中三窗全部合同有效者的等权三制度池化:

    nowait_planner        0.029987      <- 冻结最强有效因果盲
    persistence_planner   0.048127
    climatology_planner   0.176041

## 6. 三窗与池化结果

    window      curve      nowait      delta
    low      0.053444    0.047080    +13.52%
    mid      0.011814    0.011776     +0.32%
    high     0.073726    0.031105   +137.02%
    pooled   0.046328    0.029987    +54.49%    方向有利 0/3

单位为仿真器真实 terminal carbon / completed MI,非规划器内部成本。
主结果为**等权三制度池化**,不是 2021 年度平均 —— p10/p50/p90 是分层抽样,不按全年频率加权。

## 7. 机械门判定

    1  有效比较臂满足全部合同        MET(curve 与 nowait 六格全绿)
    2  池化降碳 >= 5%                NOT MET   实际 +54.49%
    3  >= 2/3 窗口方向有利           NOT MET   实际 0/3

**PHYSICAL-TIMEBASE CURVE-PLANNER GATE: STOP。**

## 8. legacy 与 physical 的边界

    legacy(加速天气,600x 风电)    2021 -55.85% 3/3    2020 -68.36% 3/3
    physical(10 分钟真实节拍)      2021 +54.49% 0/3

旧的两位数正收益**只存在于 600 倍加速天气的 legacy 考场中**;该正结果**没有在修复后的物理时间基上存活**。

未做同窗、同年、仅切时间基的严格 A/B,故**不声称**时间基错误单独因果制造了全部 -68%;
只记录方向与量级的消失**与时间基加速一致**。

legacy 报告全部保留并已加横幅:G1_PLANNER_CALIBRATION_2021.md、G1_PLANNER_GATE_VERDICT_2020.md、
HORIZON_GATE_VERDICT.md、HORIZON_GATE_PREREG.md、PLANNER_GATE_PREREG.md。

## 9. 解释性分解(仅用现有列,未启动新实验)

    window   arm      total_Wh   green_Wh   brown_Wh   waste_Wh  green_ratio  ep_len  mean_comp
    low      curve      887.74     251.07     636.67      83.34       0.2828   10024     3767.0
    low      nowait     818.56     262.92     555.64      39.62       0.3212    6106     3081.5
    mid      curve      965.29     950.39      14.90    1596.94       0.9846   10099     4666.2
    mid      nowait     891.53     856.21      35.32     689.90       0.9604    6160     3073.9
    high     curve     1086.84     970.43     116.40    7503.56       0.8929   10198     6020.4
    high     nowait     900.63     859.71      40.92    4180.22       0.9546    6106     3080.3

curve 在三窗均延长 episode(6106->10024、6160->10099、6106->10198,约 +64...67%),平均完成时间上升
(最高 3080->6020),总能耗上升 8.5%/8.3%/20.7%。**与静态/持有能耗解释一致。**

绿电浪费在三窗均上升(39.6->83.3、689.9->1596.9、4180.2->7503.6),绿电占比在 low 与 high 下降
(0.321->0.283、0.955->0.893)。**追逐未来曲线未转化为更多绿能利用。**

high 窗路由分布(curve 相对 nowait 劣化最大的一格):

    arm         dc0    dc1    dc2    dc3   dc4
    curve      5102    215   2683      0     0
    nowait     7317      0    683      0     0
    brown      0.08   0.35   0.55   0.75  0.92

curve 把约 2200 个作业移离最干净的 DC0(棕 0.08)转往 DC2(棕 0.55),per-DC 绿电占比 DC2 仅 0.65。
**空间放置是主要伴随项。**

窗口内风电特征(一格实际读取的 21 行,三站合计 kW):

    window      min      max     mean      CV   单调段数
    high     3564.5   5240.7   4394.9   0.095          8
    low         0.0    797.8    143.4   1.736          9
    mid      1345.5   1387.0   1366.3   0.009          1

**机制未判定。** 上述为伴随观测,未做受控分解实验,不构成因果结论。
climatology 表现最差(0.176041)不能单独证明「依赖未来必然有害」。

## 10. 已证实 / 未证实

**已证实**

    物理时间基下,注册的 perfect-curve-informed planner 未击败冻结最强有效因果盲,池化 +54.49%,0/3 同向
    14/15 格合同有效,主比较六格全绿
    同窗五臂共享 clock0 与完整分段签名,唯一差异为信息源与因果规则
    legacy 正收益未在物理时间基上存活

**未证实**

    完美未来信息在 C-regime 上没有价值
    C-regime 的真实 EVPI 为零
    最优 clairvoyant 策略劣于 nowait
    +54.49% 的因果机制
    时间基错误单独制造了全部 legacy 正收益

**理由**:curve 臂读取完美未来曲线,但它只是一个**启发式规划器**,不是全局最优 clairvoyant。
真正的最优信息策略至少可以模仿 nowait,故理论上不应劣于 nowait。curve 明显更差说明
**当前规划器未能安全地利用信息**,不能据此证明信息论意义上的 EVPI 为零。

命名统一为 **perfect-curve-informed planner** 或 **registered curve-informed heuristic**;
不再使用 optimal oracle / clairvoyant optimum / perfect-future optimum。

## 11. STOP 后的处置

    不接 TimeCAP clean/shuffle/anti
    不烧 RL 训练
    不做 horizon 扫描
    不改窗口、不改盲臂
    不为求正结果重跑
    现有负结果永久保留

## 12. 可直接进入论文的保守措辞

> An apparent large forecast benefit observed under an accelerated renewable trace did not
> survive restoration of the measured 10-minute wind cadence. Under the corrected time base,
> our registered perfect-curve-informed planner increased carbon by 54.5% relative to the
> strongest valid causal blind policy and was unfavourable in all three wind strata. This
> result rejects the planner/testbed pair as evidence that the deployed forecast is
> load-bearing; it does not establish that the optimal value of perfect information is zero.
