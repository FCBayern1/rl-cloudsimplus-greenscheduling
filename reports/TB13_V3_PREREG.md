# TB13 v3 —— 独立预注册(2026-09-01,写于任何 v3 执行之前)

依 Codex 2026-09-01 裁定。**不改写 v1 / v2 判决**,两者的 STOP 与产物永久保留,均不用于 v3。

## 0. v2 的正式结论

> STOP_GENERATOR_EXHAUSTED。99 个 key 中 84 个发生 arrival-span clipping,实际 offered
> concurrency 中位 7.09、最大 120,严重偏离注册的 1–5。该结果是生成器语义失败,
> 不是预报价值证据。

根因在 `workload_v2.draw`:

    span = min(target_span, horizon − max(runtime) − wait_cap − 1)

为同时容纳目标并发、完整 slack 与短 horizon,到达跨度被压缩;18 个 key 更被压成单点
(全部作业同时到达第 0 行)。审计产物 `g1/tb13/axis_audit_v2_out/`。

## 1. 负载构造(消除 clipping)

    runtime      每个 workload 固定一半 6 行、一半 12 行,仅随机置换作业顺序
                 ⟹ sum(runtime) = 9 × n_jobs,  max(runtime) = 12
    S            = ceil(9 × n_jobs / concurrency)
    相容条件      horizon ≥ S + 12 + wait_cap
    arrival      把 [0, S) 分成 n_jobs 个连续小区间,每区间【确定性】抽一个到达
                 (不做整体均匀抽样,避免再次意外聚团)
    runtime→job  由冻结 seed 的置换决定
    deadline     = arrival + runtime + wait_cap

**不再存在 `min(target_span, capacity_span)` 或任何形式的 clipping。**

n_jobs 取偶数以使"一半 6 行、一半 12 行"精确成立:`{8, 10, 12}`。

## 2. 相容轴集合(机械得到,不得手删)

    horizon      {72, 96, 144}
    n_jobs       {8, 10, 12}
    concurrency  {1, 2, 3, 5}
    wait_cap     {6, 12, 24}

108 个笛卡尔组合中满足 `horizon ≥ S + 12 + wait_cap` 的为 **89** 组:

    按 horizon    72:24   96:29   144:36
    按 n_jobs      8:32   10:29    12:28
    按 concurrency 1:11    2:24     3:27    5:27
    按 wait_cap    6:31   12:30    24:28

144 行 = 24 小时,不超过 TimeCAP 视界。

## 3. preflight 逐格断言

    arrival_span > 1
    deadline = arrival + runtime + wait_cap ≤ horizon
    S == ceil(sum(runtime) / target_concurrency)

任一不成立即 STOP。

## 4. 六个季节窗口(确定性,完全不读绿电)

时区偏移映射**从冻结 map 机械读取,不写死**。TB13 的生成器与 Round 0 均不施加逐站偏移
(`instance_gen` 与 `round0` 中不存在 `time_zone_offset_rows`),故

    shift_map = {0: 0, 1: 0, 2: 0}     shift_min = 0   shift_max = 0
    H_max = 144
    W = H_max + shift_max − shift_min = 144

把 2021 trace `[0, N)` 分成六个连续等宽季节层,第 j 层

    L_j          = floor(j·N / 6)
    R_j          = floor((j+1)·N / 6)
    foot_start_j = L_j + floor((R_j − L_j − W) / 2)
    base_offset_j = foot_start_j − shift_min

全部涡轮实际读取的联合区间为 `[foot_start_j, foot_start_j + W)`。

    N = 52559

     j      L_j      R_j  foot_start  base_offset  foot_end
     0        0     8759        4307         4307      4451
     1     8759    17519       13067        13067     13211
     2    17519    26279       21827        21827     21971
     3    26279    35039       30587        30587     30731
     4    35039    43799       39347        39347     39491
     5    43799    52559       48107        48107     48251

    六个联合区间互不重叠     True
    全部落在 trace 内        True
    window artifact SHA      e1574c954c85dd0f     (g1/tb13/v3_windows.json)

全部 horizon、triplet、turbines-per-site **共用同一组六个 base offset**;
较短 horizon 嵌套在 144 行窗口中;**不按绿电强弱替换任何窗口**。

## 5. 保留不变的部分

    最严 budget_fraction = 0.10 的纯可行性门
    reservation_edf_blind 全格履约门
    workload 与风况 / budget 完全解耦(workload key 不含 green trace、triplet、
      divisor、offset、season、budget_fraction)
    冻结 seed 的字节语义、内容哈希、确定性求解参数(v2 Addendum A.1–A.4)
    Phase A 冻结单一盲臂之后,才准运行碳 oracle
    CONFIRMATION 涡轮零触碰;验收与预约代码源码级禁读绿电与碳字段
    Round 0 物理门的五个量与阈值(正相关 0.70–0.95、同时贫风非退化、
      最优 DC 变化 ≥10%、无 ρ 截断)

## 6. 执行阶梯

    v3 轴 / 窗口 preflight
    → Round 0 物理门(必须重跑:horizon 与窗口长度均已改变,旧 Round 0 保留但不用于 v3)
    → workload 可行性与 reservation-EDF 门
    → Phase A 冻结盲臂
    → Phase B exact EVPI

## 7. 与 v1 / v2 的关系

v1 的 `STOP_NO_VALID_BLIND`、v2 的 `STOP_GENERATOR_EXHAUSTED` 及其全部产物与诊断永久保留。
v3 的结果不得回溯解释前两轮,前两轮的数字亦不得替代 v3 的判决。

---

# Addendum B(append-only)—— 2026-09-01,规模上限与四项冻结

本节只增不改。上文与本节冲突处以本节为准。

## B.0 v3 的最坏规模(机械核实)

    Round 0 物理单元
      3 pes × 4 concurrency × 2 turbines/site × 5 divisor × 3 horizon
      × 6 triplet × 6 season                                    = 12,960
    Round 0 展开上限
      72 layers × 2 anchors × 3 divisor 邻域                     =    432
    Round 1 最坏
      432 × 最多 9 个相容 (n_jobs, wait_cap) × 4 budget          = 15,552

主文沿用的"1,728 上限"**漏乘了 n_jobs × wait_cap**,不得再用;且 horizon 上限已由 48 涨到 144,
直接全跑会超出机时预算。

## B.1 零时区是场景语义,不是实现细节

> TB13-v3 的三个 DC 注册为**共享天气时钟**,`shift_map = {0:0, 1:0, 2:0}`。
> 最终仿真器配置必须同样为零,并由接线哨兵验证。
> 将来若加入地理时差,视为**新场景**,必须重跑全部阶梯。

## B.2 分区到达算法(写死)

    lo_i = floor(i·S / n)
    hi_i = floor((i+1)·S / n)
    arrival_i ∈ [lo_i, hi_i)                     i = 0 … n−1

arrival 与 runtime 置换使用**域分离 seed**,以免代码调用顺序改变负载:

    seed_arrival  = sha256(payload + ":arrival:"  + str(k))[:8]  % 2**31
    seed_runtime  = sha256(payload + ":runtime:" + str(k))[:8]  % 2**31

`payload` 为 v2 Addendum A.1 的 canonical JSON。两条流互不影响。

## B.3 数据完整性门

    全部 DISCOVERY 涡轮的 2021 文件行数必须均等于 52,559
      已核实:24 个涡轮,唯一行数 {52559}
    六个窗口 × 三个 horizon 的真实切片必须逐文件不越界
    v3 workload key 上限 = 3 pes × 89 相容组合 = 267
      【不再断言 99】;正式数量按 Round 0 之后的 cohort 对 key 的投影机械计算

## B.4 Phase A / B 的机时 cohort(冻结)

保持旧预算 **1,728 cells**,但以**完整 block** 为抽样单位:

    一个 block = 同一 anchor + 同一 (n_jobs, wait_cap)
                 × 3 个 divisor 邻域 × 4 个 budget
               = 12 cells

    最多选择 144 个 block,即 144 × 12 = 1,728 cells

选择规则:

    只在 Round 0 通过的 anchor 上构造
    按 72 个 layer 轮转,保证层覆盖
    层内按 canonical block SHA 排序
    不拆 divisor 邻域,不拆四个 budget
    不读取绿电数值、盲臂碳或 EVPI
    少于 144 个 block 时全部使用
    零碳 preflight、Phase A、Phase B 使用【完全相同】的冻结 cohort

如此既保住完整的 3 load × 4 budget 邻域,又把最坏机时锁回 1,728 格。

## B.5 执行顺序(冻结)

    v3 轴 / 窗口门
    → Round 0-v3
    → 冻结最多 144 个完整 block
    → 零碳 preflight
    → Phase A
    → Phase B
