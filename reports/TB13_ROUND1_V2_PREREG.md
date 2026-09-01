# TB13 Round 1-v2 —— 独立预注册(2026-09-01,写于任何 v2 执行之前)

依 Codex 2026-09-01 裁定。**本文件不改写 v1 判决。** Round 1 v1 的
`STOP_NO_VALID_BLIND` 永久保留,其 Phase B 永不恢复。Round 0 的物理筛选结果复用,不重跑。

## 0. v1 结论(限定表述)

> Round 1 因无单一盲臂全格履约而停止;事后零碳诊断发现,一个唯一 workload 本身不可行,
> 另有两个唯一 workload 虽离线可行但四个现有在线盲均失败。所有强制失败均由预算门触发,
> 并在当时与容量冲突。

诊断产物 `g1/tb13/poststop_out/`,说明 `g1/tb13/round1_out/STOP_NOTE.md`。

## 1. 两个同时存在的缺陷

    生成器   未保证可行域:最严 budget_fraction 下存在离线也排不出日程的 workload
    盲臂族   缺少 contract-safe 的容量预约策略,离线可行的 workload 仍可能全臂失败

v2 一并关闭,不分先后。

## 2. workload key(固定)

    seed, horizon, pes_per_job, concurrency, n_jobs, wait_cap, runtime_set

**明确排除**:green trace、triplet、installed_divisor、offset/season、budget_fraction。
这些均不得进入 workload 的随机数发生器。

v1 的 `build_instance` 把 `axes["offset"]` 混进种子,导致每个季节重新采样,
1,296 格只得到 272 个唯一 workload。v2 消除该依赖。

## 3. 唯一 workload 的正确目标

Round 0 的 36 个展开物理实例含 **11** 种唯一 `(horizon, pes_per_job, concurrency)`:

    (36,2,5) (36,4,1) (36,4,3) (36,8,2) (36,8,3) (36,8,5)
    (48,2,5) (48,4,1) (48,4,2) (48,8,3) (48,8,5)

    seed-0 唯一 workload = 11 × 3 n_jobs × 3 wait_cap = 99
    seeds 0/1/2          = 297
    grid cells           = 36 × 3 × 3 × 4 = 1,296

同一 workload 复用于 4 个 budget 与其对应的全部绿电配置。

## 4. 生成器的确定性可行性验收

对每个 workload key 使用**冻结的确定性 seed 序列**:

    seed_k = sha256(canonical(workload_key) + ":" + str(k)) 的前 8 字节取模 2**31
    k 自 0 递增

在**最严 `budget_fraction = 0.10`** 下,同时要求:

    CP-SAT 纯可行性模型返回 FEASIBLE
    reservation_edf_blind 履约

    重试上限 MAX_RETRIES = 64
    耗尽即【生成器 STOP】,不得删轴、不得放宽 bf、不得改 wait_cap

验收通过的 workload 直接复用于四个 budget 与全部对应绿电配置,**不再重新采样**。

## 5. `reservation_edf_blind`

**与绿电完全无关**,以免生成器的验收间接依赖天气:

    只看已到达作业
    EDF 排序(deadline 升序,平局取较小 job id)
    维护持久的逐站、逐时段容量预约
    在预约表上选择【最早可行启动】
    站点 tie-break 固定为较小的 DC index
    预约的总等待不得超过预算

现有四盲**继续保留**。Phase A 仍从**所有全格有效臂**中按池化碳冻结**一个**。
新臂只保证候选集合至少有一个合同安全成员,**不保证最终会被选中**。

## 6. v2 零碳 preflight(不过则不跑任何碳优化)

机械得到全部下列结果方可继续:

    grid cells                        = 1,296
    unique seed-0 workloads           = 99
    全部实例 CP-SAT                    = FEASIBLE(UNKNOWN 不算通过)
    reservation_edf_blind             = 全格履约
    同一 workload hash 跨 budget 与绿电配置完全一致

preflight 不读碳、不算 EVPI、不选盲臂。

## 7. v2 执行顺序

    Round 0        复用 v1 的 36 个展开物理实例,不重跑物理筛选
    生成器验收     §4,失败即生成器 STOP
    零碳 preflight §6,不过即停
    Phase A        跑全部盲臂(含新臂),按池化碳冻结单一臂,写 freeze artifact 与 SHA
    Phase B        才解 exact oracle,对冻结臂算 EVPI,执行 OPTIMAL + EVPI ≥ 15% + 分布门

阈值、窗口、涡轮分割、EU-CRD 非退化门一律沿用主文与 Addendum A–C,**不作任何放宽**。

## 8. 与 v1 的关系

v1 的 STOP、诊断与全部产物永久保留,不删不改。v2 是独立注册,其结果不得回溯解释 v1,
亦不得用 v1 的数字替代 v2 的判决。

---

# Addendum A(append-only)—— 2026-09-01,四处机械歧义与两条追加门

本节只增不改。上文任何与本节冲突之处,以本节为准。

## A.1 冻结 seed 的精确字节语义

    payload = json.dumps(key, sort_keys=True, separators=(",", ":"))
    digest  = sha256((payload + ":" + str(k)).encode()).digest()
    seed    = int.from_bytes(digest[:8], "big") % 2**31

`key` 为 §2 的 workload key。`runtime_set` 固定序列化为 JSON list(非 tuple、非 set),
以免解释器版本或容器类型改变字节串。`k` 自 0 递增,以十进制无前导零写入。

## A.2 一致性哈希必须覆盖实际内容

只哈希 key 不足以证明复用 —— key 相同而生成过程有隐藏依赖时,内容仍可能不同。
内容哈希的 payload 至少包含有序的:

    arrival, runtime, pes, deadline, horizon

preflight 必须机械得到:

    99   个不同的内容哈希
    1,296 个引用(每格引用其 workload 的内容哈希)
    同一 key 在不同 budget_fraction 与不同风电配置下,内容哈希【完全相同】

## A.3 可行性验收必须与墙钟无关

否则"第一个被接受的 retry"会随机器而变,冻结的 seed 序列失去意义。冻结:

    num_search_workers      = 1
    CP-SAT random_seed      = 固定常数
    max_deterministic_time  = 固定值(不使用 max_time_in_seconds)
    模型                     无目标,纯约束
    FEASIBLE 或 OPTIMAL(即存在 witness)统一记为通过
    UNKNOWN 进入下一个 retry,【不得】当作不可行

## A.4 `reservation_edf_blind` 的精确语义

    同一时刻到达者按 (deadline, job_id) 升序处理
    到达时【立即】建立不可撤销的预约,此后不再改动
    搜索 start ∈ [arrival, latest_start]
    优先最早可行 start;同一 start 取最小 DC index
    总预约等待包含此前全部已建立的预约
    找不到任何可行预约即【立即失败】,不回溯、不重排
    源码不得出现 green / cb / cg / climatology / carbon 的读取

## A.5 零碳 preflight 追加两门

在 §6 五门之外,另加:

    CONFIRMATION 涡轮零触碰
    验收与预约代码在【源码层】禁止读取任何绿电或碳字段(由测试断言)

七门全过之前,**不运行任何盲臂碳比较,也不运行 exact carbon oracle**。

## A.6 实现顺序(冻结)

    workload 内容生成与缓存
    → schedule-only 可行性验收
    → reservation EDF
    → 99 / 1,296 零碳 preflight
    → Phase A

## A.7 仓库卫生

`g1/tb13/__pycache__/` 的 10 个 `.pyc` 此前虽被 `.gitignore` 命中,却仍是 tracked 文件,
导致运行测试后 provenance 变脏。已从索引移除(commit 836840f)。
