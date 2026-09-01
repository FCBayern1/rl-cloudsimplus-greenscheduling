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
