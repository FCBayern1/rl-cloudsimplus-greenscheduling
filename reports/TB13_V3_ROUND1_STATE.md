# TB13-v3 执行记录:轴/窗口门到 Phase A 冻结

本文件记录 v3 冻结阶梯的机械执行结果。判据全部取自 `reports/TB13_V3_PREREG.md`
(含 Addendum A/B),本轮**未改动任何判据、轴、窗口、block 选择规则或 1,728 上限**。

## 1. 交付与提交

    bb6e119   workload_v3 / round0_v3 / preflight_v3 / test_v3
    46d2725   Round 0-v3 产物与冻结 cohort
    e7e1628   零碳 preflight(独立文件 zero_emission_v3.py)
    9c282c4   零碳 preflight 产物
    0d05214   reservation-EDF 臂注册 + round1_v3

`zero_emission_v3.py` 与 `preflight_v3.py` 分开,是因为后者的摘要已写进 Round 0-v3
的 manifest,已完成运行的 provenance 不应因为后续阶段追加代码而改变。

## 2. 轴 / 窗口门(preflight_v3)

    相容组合          89          与预注册一致
    workload key      267 = 3 pes x 89       上限即实际
    三条逐格断言      arrival_span > 1 / deadline <= horizon / S == ceil(sum r / c)
                      267 个 key 全过,零失败
    DISCOVERY 涡轮    24 个,逐文件计数
    行数唯一值        {52559}
    六窗              互不重叠、全在界内,三个 horizon 逐涡轮切片长度精确
    零时区            shift_map 全零,站点数与 N_DC 一致
    窗口 artifact     payload SHA e1574c954c85dd0f(与预注册一致)
                      文件字节 SHA 5020453eddb89213(两者不是同一个量,均记录)

判决 PASS。

## 3. Round 0-v3(物理门)

物理门的五个量与阈值原样复用 v1 的实现,只更换定义域。

    物理单元          3 pes x 4 concurrency x 2 tps x 5 divisor
                      x 3 horizon x 6 triplet x 6 season = 12,960
    通过              528
    落选              12,432
    落选原因          相关性出带 8,364 / 最优站点几乎不变 1,888
                      某站残余绿电无变化 1,776 / 同时贫风退化 404
    层                72 层,25 层有幸存者,47 层为空(按协议记录而非跳过)
    anchor            50 = 25 层 x 2
    commit            bb6e119(clean tree,记录源码 / 预注册 / 窗口 / data split 的 SHA)

## 4. 冻结 cohort

    候选 block        394
    冻结 block        144(达到上限)
    每 block cells    12(3 divisor 邻域 x 4 budget),无一例外
    cells             1,728,去重后仍 1,728
    因撞格跳过        0
    层覆盖            25 层,19 层 6 个 block、6 层 5 个 block
    cohort SHA        241764512eb5c658591f3a46
    CONFIRMATION      零触碰(逐 key 断言)

## 5. 零碳 preflight

11 门全过,`verdict = PASS`。

    cohort 摘要与 manifest 双向一致
    144 block / 1,728 cell / 每 block 12 cell / 无重复 cell
    114 个唯一 workload,全部在 retry 0 通过最严 budget 的 CP-SAT 可行性
    reservation-EDF 逐格履约:1,728/1,728
    同一 key 的内容哈希跨 budget 与绿电配置完全一致
    源码级禁读绿电与碳字段(自守文件排除声明行本身)

一个应当记录的数字:reservation-EDF 的总等待在 1,704 个 cell 上是 0,
在 24 个 cell 上是 2。容量在本 cohort 上几乎不构成压力。

## 6. Phase A(冻结单一盲臂)

按 v2 §5 与 A.4 的注册语义补上第五臂 `reservation_edf_blind`,它委托给
`schedule_feasibility.reservation_edf`,该文件源码级不得读天气或碳。

    臂                    全格有效    池化碳
    immediate_current_only   是       558.2241563257201   ← 冻结
    persistence              是       558.2241563257201
    reservation_edf_blind    是       558.3074834829989
    climatology              否(102 格失败)
    reactive_wait            否(132 格失败)

`persistence` 与 `immediate_current_only` 的池化碳**逐格完全相同**,
在 1,728 个 cell 上无一例外。在十分钟物理行距下,前一行的绿电值没有改变过任何一次决策。

## 7. 冻结后诊断:碳账本被一个调度动不了的地板占据

以下是解释性分解,**不是判据,也不改变任何判决**。

对每个 cell 取"没有任何作业"的碳作为与调度无关的地板,再与冻结臂的总碳相比,
得到调度可归因的份额:

    p0     0.142%
    p10    0.740%
    p25    2.201%
    p50    3.761%
    p75    7.022%
    p90   10.711%
    p100  17.893%

    份额 >= 15% 的 cell:12 / 1,728

功率常数解释了这个比例:

    每站空载        51.4 W x 1 host        三站 144 行 = 3,700.8 Wh
    最大动态负载    12 作业 x 8 PE x 9 行  =   182.9 Wh

即一个站点满载 16 PE 时的动态功率是 20.3 W,不到自身空载 51.4 W 的一半。
调度能移动的能量上限约占账本的 5%。

注册的 EVPI 门是"相对总碳 >= 15%"。在这套功率常数下,即使完美规划把全部动态碳降到零,
1,728 格里也只有 12 格在算术上可能达到 15%。这是**场景设计层面的量纲问题**,
不是判据松紧问题,因此本轮不做任何调整,如实记录。

各臂之间的逐格价差同样很小:中位 0.48%,p90 2.39%,最大 9.73%。

## 8. Phase B(exact 模型 + EVPI)判决:STOP

    cells                 1,728
    proven OPTIMAL        1,728    未解出 0
    EVPI >= 15%           0
    advancing cells       0
    advancing blocks      0
    wall                  689.55 s
    commit                0d05214,cohort 241764512eb5c658591f3a46,preflight PASS

EVPI(相对总碳,注册口径)分位数:

    p10   0.01%
    p25   0.33%
    p50   1.14%
    p75   2.26%
    p90   4.43%
    p100 11.51%

**Round 1-v3 记为 STOP_EVPI_GATE_NOT_MET。** 该 STOP 与 v1、v2 的 STOP 一样永久保留,
其数字不得回溯解释另外两轮,另外两轮的数字也不得替代本轮判决。

由于 1,728 格全部解到证明最优,这不是求解器未收敛的产物;在本场景的功率常数下,
注册的 15% 门在算术上从一开始就够不着。

## 9. 与判决并列的分解(解释性,不构成因果结论)

把 STOP 的分母换成"调度真正能移动的那部分碳",同一批 1,728 个解不变:

    EVPI / 总碳(注册)      p25 0.33%   p50  1.14%   p75  2.26%   p90  4.43%
    EVPI / 可移动碳(诊断)  p25 13.89%  p50 29.05%  p75 51.81%  p90 71.56%
    可移动碳占总碳(诊断)   p25 2.20%   p50  3.76%   p75  7.02%  p90 10.71%

    可移动口径下 >= 15% 的 cell   1,265 / 1,728
    注册口径下   >= 15% 的 cell       0 / 1,728

其余两个注册门在本 cohort 上并不构成瓶颈:

    exact 解里立即启动的作业比例   中位 62%,p10 38%,p90 100%
    落在注册 20–80% 带内的 cell    1,271 / 1,728
    pes_share >= 25% 的 cell       1,452 / 1,728

这三行只说明"在这套功率常数下,总碳这个分母被空载地板占据了 96%",
不说明预报在本场景有多大价值,也不说明注册门应当改动。是否另立新注册(改功率常数、
改每站主机数或暴露的 PE 数、或换分母)属于场景设计裁定,本轮不做,也不改本轮判决。
