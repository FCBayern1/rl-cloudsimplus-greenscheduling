# 预注册：COMPRESSED 短视界 TimeCAP 合成正控（方案二）

日期：2026-09-01。分支：`gpu/compressed-timecap-s2`。起点 SHA：`d91d39ce6c5592b3ce224e6d0ff669a99692cae5`。
上位工单：`reports/WORKORDER_GPU_COMPRESSED_TIMECAP_SCHEME2.md`。

本文件在**任何碳结果产生之前**冻结第一批交付：窗口、workload、配置派生规则与全部机械判据。
第 7、9、10 节的门槛写死在这里；看到结果之后修改其中任何一条，都使该轮作废，必须以附录形式
追加修正并从头重跑。

## 0. 身份

> accelerated-weather synthetic mechanism positive control
>
> 加速天气的合成机制正控

一行风电 = 一个合成控制 epoch = COMPRESSED 下的一个仿真秒。**不是十分钟，不是一小时。**
144 行不得写成"144 秒预测"或"24 小时预测"。本线不进入 REAL_TIME 物理证据链，不改变
C-regime、TB12、TB13-v1/v2/v3/v4 的任何既有判决。

本轮**不触碰** `g1/tb13/`、`reports/TB13_V4_PREREG.md` 及其产物目录，**不编辑**
`drl-manager/Code/`。

## 1. 本批交付与冻结哈希

| 文件 | 作用 |
|---|---|
| `reports/COMPRESSED_TIMECAP_S2_PREREG.md` | 本文件 |
| `g1/compressed_timecap_s2/constants.py` | 全部冻结常量 |
| `g1/compressed_timecap_s2/select_windows.py` | 数据盲的六窗选择器 |
| `g1/compressed_timecap_s2/windows.json` | 六窗产物 |
| `g1/compressed_timecap_s2/workload.py` | 确定性 workload 生成与哈希 |
| `g1/compressed_timecap_s2/generate_configs.py` | 由冻结 base 程序化派生配置 |
| `g1/compressed_timecap_s2/config_cts2_stage_a.yml` | 108 个 Stage A block |
| `g1/compressed_timecap_s2/workloads.json` | 108 个 workload 的申报清单 |
| `cloudsimplus-gateway/src/main/resources/traces/cts2_*.csv` | 108 条 trace |
| `g1/compressed_timecap_s2/test_*.py` | 65 个测试 |
| `reports/COMPRESSED_TIMECAP_S2_JAR_MANIFEST.md` | jar / 源码 / 配置 manifest |

冻结哈希：

    windows.json  selection_hash      e216d4d35f0a7320e523335ae42bbed27138b451b51daa3bbc9f6c8e67bb793f
    windows.json  文件 SHA256          9f646c4621246ad497a7eedacbbc32ccdeb8365a242222c89839cf157e7db10c
    config_cts2_stage_a.yml SHA256    a5b924dde7f2c876bbb5afb6a995f9380135e18783c031e9d536a8d5afc05502

每条 trace 的 SHA256 逐条列在 `workloads.json` 的 `workloads[].content_sha256`。
`test_workload.py::TestTracesOnDisk` 会逐字节比对磁盘上的 trace 与生成器输出，
`test_generate_configs.py::test_emitted_config_matches_a_fresh_build` 会比对 YAML 与重建结果——
手工编辑任何一个产物都会让测试失败。

## 2. 数据隔离

    TimeCAP train / validation     只用 2020
    scheduler DISCOVERY            2021 的 w0 / w2 / w4
    scheduler CONFIRMATION         2021 的 w1 / w3 / w5
    2022                           禁用

实测行数（`windows.json::turbine_row_counts`，只数换行、不解析任何功率字段）：

    Turbine_{12,36,91,95,96}_2020    32224 行
    Turbine_{12,36,91,95,96}_2021    52559 行
    Turbine_{12,36,91,95,96}_2022        2 行   -> 禁用，与工单一致

2020 与 2021 是不同文件，训练集与评测集因此在文件层面不相交；六个调度窗口全部落在 2021，
彼此不重叠（第 3 节）。配置里 `wind_csv_year = 2021`，而 base block 原值是 2020——如果不改，
调度评测就会跑在留给 TimeCAP 训练的那一年上，整个隔离前提失效。该键由
`test_generate_configs.py::test_scheduler_year_is_the_eval_year_not_the_training_year` 钉死。

## 3. 六窗选择：确定性、数据盲

### 3.1 盲性

`select_windows.py` 只允许读取：文件是否存在、行数、年份、每 DC 时区偏移、以及本预注册的常量。
**不读取任何一个功率值。** 行数由数一遍换行得到，从不按逗号切分。
`test_select_windows.py::test_selector_never_parses_a_power_value` 对源码做静态检查
（禁止出现 `power_kw`、`DictReader`、`np.percentile` 等），保证这条盲性不会在后续修改中悄悄失效。

理由：按绿电强弱选窗，等于让天气挑考卷；DISCOVERY / CONFIRMATION 拆分的全部意义就是排除这一点。

### 3.2 footprint

COMPRESSED 下 `row_seconds = 1.0`，DC `d` 在仿真时钟 `t` 读到的行是

    row(d, t) = episode_offset + tz_rows[d] + simulation_warmup_rows + floor(t / 1.0)

`simulation_warmup_rows` 在 base block 未设置，按 Java 默认为 0。footprint 逐项：

    clock0_rows            13      CloudSim 启动开销（首次观测时时钟已在 13 s）
    max_episode_steps    3792      全网格最长 cell 的解析上界
    warmup_rows             0
    max_tz_rows           108      DC_APAC 的时区偏移，五站中最大
    guard_rows             64      吸收 clock0 重测、floor 取整、额外终止事件
    footprint_rows       3977      = 13 + 3792 + 0 + 108 + 64

`max_episode_steps` 覆盖 episode 与 terminal drain：单 cell 的 episode 长度 =
（最后一个作业的最迟合法完成时刻）+ `DRAIN_STEPS = 120`。最长 cell 是
`runtime=72, wait=72, concurrency=1, n_jobs=50`，解析上界 `49*72 + 72 + 72 + 120 = 3792` 步。
窗口选择用解析上界（不依赖任何 RNG 抽样），实际配置用实测长度，两者关系
`realised <= bound` 由测试钉死；实测最长 3361 步、最短 242 步。

### 3.3 选择规则（冻结）

    range      = 52559 - 3977 = 48582        每个可达 offset 都在年内，靠构造而非裁剪
    block      = range // 6 = 8097           footprint 3977 <= 8097，装得下
    centre_j   = (2j+1) * range // 12        j = 0..5
    候选 k     = 1 .. K，K = (range-1) // 1009 = 48
    取 k_j     = 使 |1009*k - centre_j| 最小的 k，同分取更小的 k
    offset_j   = 1009 * k_j
    拆分       DISCOVERY = block 0/2/4，CONFIRMATION = block 1/3/5

**为什么限制在 k ≤ 48（不 wrap）：** 窗口由 `evaluate.py --reset-skip k` 寻址，而该实现会在测量
前真的执行 k 次 `env.reset()`，每次都是一次完整的 Java `resetSimulation()`。用模逆解出"精确落在
块心"的 offset，会得到 k ≈ 46000，那不是一个能跑的实验。1009*k 在 wrap 之前严格单调，因此
k ≤ 48 既保证可运行（最多 44 次预热 reset），又只让每个窗口偏离块心几百行。

**为什么等距 + 交错：** 本脚本的第一版按 k 递增贪心接受，得到 k = 1,5,9,13,17,21，六个窗口全部
落在前半年、间距仅 4036 行（略大于一个 footprint）。那样 CONFIRMATION 取到的是紧邻 DISCOVERY 的
天气，确认不了任何东西。等距使六窗横跨全年，交错使两个集合季节匹配——确认失败就不能再用
"两组落在一年的不同部分"来解释。

### 3.4 实际窗口

| 窗 | split | k (`--reset-skip`) | offset | 读取行区间 | 与块心偏差 |
|---|---|---|---|---|---|
| w0 | DISCOVERY | 4 | 4036 | [4036, 8012] | −12 |
| w1 | CONFIRMATION | 12 | 12108 | [12108, 16084] | −37 |
| w2 | DISCOVERY | 20 | 20180 | [20180, 24156] | −62 |
| w3 | CONFIRMATION | 28 | 28252 | [28252, 32228] | −87 |
| w4 | DISCOVERY | 36 | 36324 | [36324, 40300] | −112 |
| w5 | CONFIRMATION | 44 | 44396 | [44396, 48372] | −137 |

相邻窗口间距 8072 行 > 2 × footprint；最大读取行 48372 < 52559。
互不重叠与不越界由 `test_select_windows.py` 的
`test_read_intervals_are_pairwise_disjoint` / `test_no_window_runs_off_the_end_of_the_year` 钉死。

### 3.5 clock zero 复核（Stage A 开跑前的硬前置）

`CLOCK0_SEC = 13.0` 继承自 C-regime 在 COMPRESSED 下的实测，本预注册直接沿用。Stage A 的
第一次碳运行之前，必须用一次一次性探针复核首次观测时的仿真时钟，并把实测值写进 Stage A 产物。
若实测偏离 13 超过 `guard_rows = 64`，记 `STOP_CLOCK_ZERO`，重算 footprint 并重新选窗，
不得直接开跑。

## 4. Workload：闭合条件与生成公式

### 4.1 闭合条件

工单第 4 节的条件是 `(s_i - a_i) + r_i <= 144`，不是"等待不超过 144"。生成器把它做成恒等式：

    deadline_i      = arrival_i + wait_cap + runtime_i
    latest_start_i  = deadline_i - runtime_i = arrival_i + wait_cap
    => (s_i - a_i) + r_i <= wait_cap + runtime_i <= wait_cap + runtime_rows <= 144

因为 `runtime_i` 抽样时以注册的 `runtime_rows` 为硬上界，所以注册值可以直接代入闭合式。
`test_workload.py::test_closure_holds_per_job_not_just_on_average` 对全部 108 个 cell 的每个作业逐条检查。

这正是 TB12 窗口探针踩过的坑的反面：那一轮窗口比作业短六倍，"这个作业塞得进预测吗"对每个作业
在每个 epoch 都是同一个答案（否），预测因此不可能改变任何决策。

### 4.2 冻结网格

    runtime_rows       {24, 48, 72}
    wait_cap_rows      {24, 48, 72, 96, 120}
    admissible         runtime_rows + wait_cap_rows <= 144   -> 12 个 (r,w) 组合
    concurrency        {1, 3, 5}
    n_jobs             {20, 35, 50}
    cells              12 x 3 x 3 = 108

### 4.3 生成公式（冻结）

    seed(cell, stream)  = int(sha256("cts2|20260901|<cell key>|<stream>")[:16], 16)
    runtime_i           ~ U{ceil(0.75*r) .. r}                       stream "runtime"
    pes_i               ~ U{2, 4}                                    stream "pes"
    spacing             = mean(runtime) / concurrency
    arrival_i           = round(i * spacing)                         i = 0 .. n-1
    mi_i                = runtime_i * pes_i * 40000 * 1.0
    deadline_i          = arrival_i + wait_cap + runtime_i
    file_size           = 300,  output_size = 150
    cell key            r{runtime}w{wait}c{concurrency}n{n_jobs}
    trace 名             cts2_<cell key>.csv
    content SHA256      对最终 CSV 字节取，而不是对内存数组取

两条独立随机流做了域分离，改动 PES 字母表不会平移无关 cell 的 runtime 抽样。
到达序列由目标并发和 runtime 反解得到，**没有任何 clipping**：`arrival_0 = 0`、`arrival_{n-1} > 0`、
到达时刻不全等，三条都由测试钉死。实测 offered concurrency = `sum(runtime)/arrival_span`，
按构造等于 `c * n/(n-1)`（n 个作业铺在 n−1 个间隔上），测试允许 5% 相对偏差。

每个 workload 在 `workloads.json` 中逐项申报工单第 4 节要求的全部字段：
`arrival_span_rows`、`offered_concurrency`、`runtime_min/max/mean`、`wait_cap_rows`、
`deadline_reachable`、`content_sha256`，另加 `latest_start_*`、`last_possible_finish`、
`episode_steps`。

### 4.4 执行物理

`cloudlet_cpu_utilization = 1.0`，于是 `runtime_rows = MI / (PES * VM_PE_MIPS * util)` 精确成立，
COMPRESSED 下 1 行 = 1 秒。base block 继承的是 0.5 这个 legacy 默认值，它会把每个作业拉长到两倍
注册 runtime，闭合条件直接作废——这一项由 `test_execution_physics_is_full_utilization` 钉死。

`split_large_cloudlets` 保持 base 的 `true` 不动（不做无谓偏离），但
`max_cloudlet_pes = 8` 且全网格 `pes <= 4`，所以不可能发生拆分；由
`test_no_job_can_be_split` 钉死。

`obs_cloudlet_mi_high = 14400000 = ceil(1.25 * 11520000)`，全网格统一（不逐 cell 调），
否则 r=72 / pes=4 的作业会被 base 的 1e7 上界静默截断。

## 5. latest_start backstop（显式配置）

    defer_deadline_force_mode   latest_start        显式，不继承 legacy
    defer_deadline_slack_sec    1.0                 行
    defer_urgency_window_sec    144.0               行

Java 侧规则是 `now + MI/(PES*MIPS*util) + slack >= deadline` 时强派。代入
`deadline = a + w + r` 得强派时刻 `= a + w - slack`，即恰好落在注册的最迟合法启动点前一行。

如果继承 legacy（固定前置 `now + 600 >= deadline`），本网格里每个作业的 `deadline - 600` 都小于
它自己的到达时刻——也就是说**所有作业在到达的瞬间就被强派**，144 行考场在调度器做出任何决策
之前就已经结束，而运行看上去会完全健康。`test_backstop_fires_at_the_registered_latest_start`
同时验证两件事：latest_start 下强派点等于注册最迟启动点，以及 legacy 规则会在到达前触发。

## 6. 配置派生与精确差分白名单

全部 block 由**同一个冻结 base** 程序化派生，禁止手抄：

    base config   g1/config_C_2020.yml
    base block    experiment_g1eval_matchedvan

### 6.1 Stage A 顶层白名单

    experiment_name  simulation_name  cloudlet_trace_file  max_episode_length
    green_episode_offset_range  wind_csv_year  green_interpolation_mode  green_power_scale
    cloudlet_cpu_utilization  defer_deadline_force_mode  defer_deadline_slack_sec
    defer_urgency_window_sec  obs_cloudlet_mi_high  forecast_mode  green_oracle_mode
    timecap（删除）  datacenters

`datacenters[*]` 内部白名单只有 `green_interpolation_mode`、`green_power_scale` 两个键。
拓扑（五 DC、三站有风机两站无、brown 因子、时区偏移）逐字段与 base 相同，由
`test_topology_is_untouched` 钉死。

白名单是双向合同：白名单外的键漂移是失败；白名单内的键没有被真正赋值也是失败
（`test_every_whitelisted_key_actually_changes_or_is_pinned`）。

固定语义：`time_scaling_mode = COMPRESSED`、`green_interpolation_mode = STEP`（顶层与逐 DC）、
`green_power_scale = 1.0`（COMPRESSED 下缩放由 `compressed_power_divisor = 1500.0` 承担，
这里显式钉成 1.0 防止悄悄二次缩放）。

Stage A block **删除 `timecap` 子块**，并设 `forecast_mode = none` / `green_oracle_mode = godeye`
（本仓库已注册的盲臂配对）。Stage A 的臂自己读风电 CSV，任何 checkpoint 都不得被加载。

### 6.2 Stage C 臂间白名单

    forecast_mode  green_oracle_mode  timecap  experiment_name  simulation_name

`noforecast` 臂就是被改名的 Stage A 场景本身（同 trace、同 episode 长度、同 backstop、同窗口）——
matched 就是 matched，`test_no_forecast_arm_is_the_stage_a_scenario_renamed` 要求它与 Stage A block
的差异**恰好只有两个名字键**。

负控开关的键名与取值现在就冻结：`timecap.forecast_perturbation ∈ {none, shuffle, anti}`。
**gateway 目前没有实现该开关**（`grep shuffle/anti` 在 Java 侧为空）。Stage C 不得在以下两件事完成
之前启动：(a) 实现该开关；(b) 有测试证明 shuffle 与 anti 确实改变了送达调度器的预测序列。
现在写下键名，是为了让它在看到任何结果之前就定死。

## 7. Stage A 机械判据

### 7.1 每格执行合同（全绿才算该 cell 有效）

    completion_rate_mi            >= 0.995
    ontime_mi_share               >= 0.995
    deadline_forced_count         == 0
    planner_n_stale_dropped       == 0
    n_unplanned_start             == 0
    n_wrong_dc                    == 0
    n_dispatched_never_started    == 0
    running_pes_over_cap          == 0
    同窗各臂 workload / weather / power / row signature 完全一致

### 7.2 臂与环境

Stage A 的臂全部来自已注册的 planner 家族，共用同一 planner、同一容量账本、同一空间碳模型，
只在"被给予的未来信息"上不同：

    盲臂候选   nowait_planner  reactive_wait_planner  persistence_planner  climatology_planner
    oracle144  oracle144_planner        PLANNER_HORIZON_STEPS=144
    full oracle curve_planner

环境变量（每次运行都必须写进产物）：

    EVAL_CONFIG_PATH   g1/compressed_timecap_s2/config_cts2_stage_a.yml
    ORACLE_EXPERIMENT  experiment_cts2_<cell key>
    ORACLE_YEAR        2021
    ORACLE_OFFSET_ROWS <该窗 offset>          必须与 --reset-skip k 指向同一个 offset
    PLANNER_TAIL_MODEL climatology
    PLANNER_HORIZON_STEPS  144（仅 oracle144）
    --reset-skip       <该窗 k>

`ORACLE_OFFSET_ROWS` 与 `--reset-skip` 不一致会让 planner 与仿真器读不同的天气，且不会报错。
每个 cell-窗口-臂的产物必须同时记录这两个值，并在归约时校验 `(1009*k) mod range == offset`。

### 7.3 oracle144 的尾部模型

尾部冻结为 **climatology**：每站取"严格早于本 episode 窗口"的历史均值（planner 内部的
`self.clim`，`PLANNER_TAIL_MODEL=climatology`，即家族默认）。它知道气候、不知道天气，因此是因果的。
`oracle144` = 前 144 行真值 + 该尾部；与之配对的盲臂使用**完全相同**的尾部模型。
不得对 oracle144 使用 `zero` 或 `persistence` 尾部，也不得让两臂尾部不同。

### 7.4 冻结盲臂（单一臂，先于 oracle 结果冻结）

在看到任何 oracle 结果之前，用 DISCOVERY 三窗、全部合同有效实例的 **pooled 总碳**，
从四个候选盲臂中选出**唯一一个**，作为整条线的盲臂。

    pooled_intensity(arm) = sum_over(w in DISCOVERY, valid cells) C_total
                            / sum_over(同上) MI_completed

取 pooled_intensity 最小者；同分按臂名字典序。**不得逐实例挑最低碳的盲臂。**
冻结结果写入 `g1/compressed_timecap_s2/frozen_blind.json` 并单独提交，之后才允许运行 oracle 臂。

### 7.5 主门（每个 cell）

pooled 一律用"和之比"，不是"比之均值"：

    I(arm)   = sum_w C_total(arm, w) / sum_w MI_completed(arm, w)      w in DISCOVERY 三窗
    reduction(arm) = 1 - I(arm) / I(blind)
    capture  = (I(blind) - I(oracle144)) / (I(blind) - I(full_oracle))

该 cell 通过，当且仅当以下四条同时成立：

    1. reduction(oracle144)  >= 0.05
    2. 三个 DISCOVERY 窗中至少 2 个方向有利（该窗 I(oracle144) < I(blind)）
    3. capture >= 0.50
    4. I(full_oracle) <= I(blind)                     full oracle 不劣于冻结盲臂

**分母保护：** 若 `I(blind) - I(full_oracle) <= 0`，capture 不计算、该 cell 直接不通过，
不得取绝对值、不得翻转符号、不得改用别的分母。
合同不全绿的 cell 也直接不通过（不进入 pooled，也不计入方向计数）。

### 7.6 稳定区域与中心 cell

**相邻的定义（冻结）：** 四个轴按注册顺序取值——
`runtime_rows (24,48,72)`、`wait_cap_rows (24,48,72,96,120)`、`concurrency (1,3,5)`、`n_jobs (20,35,50)`。
两个 cell 相邻，当且仅当它们**恰好在一个轴上相差一级**（该轴有序列表中相邻的两个取值），
且**其余三个键完全相同**。

在通过的 cell 上按该关系建图，取其连通分量。要求**至少存在一个规模 >= 3 的连通分量**；
不存在则视为无稳定区域。不得只取孤立最大值。

**中心 cell（不按效果选）：** canonical cell JSON =
`json.dumps(cell, sort_keys=True, separators=(",", ":"))`，取其 SHA256。
若有多个规模 >= 3 的连通分量，先选"分量内最小 cell-SHA256"最小的那个分量；
再在该分量内取 SHA256 最小的 cell 作为中心 cell。**任何一步都不看碳降幅。**

### 7.7 STOP

无稳定区域 -> 记 `STOP_ORACLE144_GATE`，停止本工单，不训练 TimeCAP、不运行 RL，
完整负结果提交并推送。

### 7.8 运行规模（预算申报）

    盲臂筛选   108 cells x 3 DISCOVERY 窗 x 4 候选盲臂 = 1296 次仿真
    oracle 臂  108 cells x 3 DISCOVERY 窗 x 2 臂       =  648 次仿真
    合计                                                 1944 次

episode 长度 242 ~ 3361 步，另加每次运行最多 44 次预热 reset。这是 CPU / 仿真器工作，
GPU 空闲也不得跳过它先训练。若实测总时长超出预算，缩减网格属于**修正预注册**，
必须在看到任何碳结果之前以附录形式追加，不得在中途按结果裁剪。

## 8. Stage B：TimeCAP 数据管线 STOP 门（本批不实现）

本批**不做**完整数据边界适配器，也不训练——理由是 Stage A 若失败，训练管线根本不会被使用。
但语义现在写死：

    TimeCAP train / validation 只使用 2020 文件
    scheduler 评测只使用 2021 文件
    2022 一律不得进入训练或评测

Stage A PASS 之后，才实现不修改 `drl-manager/Code/` 的外层审计 / 适配器，机械证明：

    任一 96+144 样本不跨 turbine 边界
    任一样本不跨 year 边界
    train / validation / scheduler evaluation 无行重叠
    scaler 只在 train 上拟合
    k=0 标签语义与 provider 的 k=0 读取语义一致
    输入列顺序与 predictor 的 13 个特征逐位一致
    Patv 单位一致

任一条不过 -> `STOP_DATA_PIPELINE`，不训练。新 dataset wrapper 只能放在
`drl-manager/timecap_prediction/` 或 `g1/compressed_timecap_s2/`。

现有 `prepare_turbine_data.py` 会把多个 turbine/year 直接拼接，而 legacy `Dataset_Custom` 按固定
行号划分 train/val/test，样本可能跨越拼接边界——不得假设它天然满足隔离。

训练身份（Stage A PASS 后才生效）：`seq_len=96`、`pred_len=144`、target `Patv`、`seed=20260901`、
单卡优先、**只按 validation loss 选 checkpoint**，scheduler carbon 不得参与选 epoch / 学习率 / checkpoint。

## 9. Stage C 机械判据

同一中心 cell、同一三窗、同一 workload、同一动作空间、同一尾部，运行五臂：
`matched no-forecast` / `TimeCAP clean` / `TimeCAP shuffle` / `TimeCAP anti` / `oracle144`（冻结参照，不重算）。

主门（DISCOVERY）：

    1. 1 - I(clean)/I(nofc) >= 0.05
    2. 三窗至少 2/3 方向有利
    3. 全部完成与容量合同通过（第 7.1 节）
    4. clean 的动作序列相对 no-forecast 非全等
    5. 负控破坏，量化如下

**"明显破坏"的量化（执行前冻结）：**

    gain(x) = I(nofc) - I(x)
    要求 gain(shuffle) <= 0.5 * gain(clean)   且   I(shuffle) >= I(clean)
    要求 gain(anti)    <= 0.5 * gain(clean)   且   I(anti)    >= I(clean)

即两个负控各自至少回吐 clean 相对 no-forecast 收益的 50%，且都不得比 clean 更优。
看到负控结果之后不得改这条阈值。

CONFIRMATION 只在 DISCOVERY 主门全过、checkpoint 与场景均已冻结后**读取一次**：
要求 pooled 方向有利且至少 2/3 窗同向。**不在确认集重新选 checkpoint、场景或盲臂**，
也不重新挑窗口或重跑 Stage A。

失败 -> 记 `STOP_TIMECAP_VALUE_GATE`，不运行 EU-CRD / RL。

## 10. Stage D 边界

仅 Stage C PASS 后，先做 1 seed / 50k smoke，健康门见工单第 11 节。
50k 通过后另写 append-only 长训预注册。本预注册**不授权**看到 smoke 后调奖励、改门限、
移动判定点或直接延长训练。

## 11. 修正纪律

本文件的任何门槛、公式、窗口、seed、网格一旦提交即冻结。修改只能以**附录**形式追加，
写明修改时间、原因、以及"此前已看到哪些结果"。在看到碳结果之后放宽任何一条判据，
该轮作废。

## 12. 成功与失败各自能声称什么

全线通过，只能声称：在一个明确标注的 accelerated-weather synthetic C-regime 正控中，
144 行 TimeCAP 预测包含可执行的调度价值，负控会破坏该价值。
**不得**声称真实十分钟 SDWPF 时间尺度、现实 24 小时预测或生产数据中心具有同等收益。

Stage A 失败 = 当前被冻结的短视界合成设计仍没有为 144 行预测提供决策价值，不是 TimeCAP 训练失败。
Stage A 通过而 Stage C 失败 = 完美 144 行未来有价值，但当前 TimeCAP 没兑现。
Stage C 通过而 RL 失败 = 断点在学习 / 信用分配。

这个三段定位是本工单最重要的产出，任何一段的 STOP 都是有效结果。

---

# 附录 A：Stage A′ 加扰 oracle 预测质量阶梯（冻结）

追加时间：2026-09-01，GPU 侧。
来源：`reports/WORKORDER_S2_ADDENDUM_A_PERTURBED_ORACLE.md`（origin/main `ed4e449`）。

**追加时已看到的结果：无。** 截至本附录提交，本线尚未运行任何碳评测，Stage A 未开跑。
因此本附录不受"看到结果后放宽判据"条款的限制。

## A.0 动机（抄自修正案 §1）

原结构把两个命题捆在一起：RL 能否使用质量为 q 的预测（命题 A），与 TimeCAP 能否产出质量 q
（命题 B）。Stage B 重训只给一个 q 点，失败时分不清死因。加扰 oracle 把预测质量做成受控自变量，
先拿到整条剂量–响应曲线。

## A.1 冻结阶梯

    tier          语义
    godeye        sigma = 0，与 oracle144 逐位相同（阶梯零点，有测试钉死）
    s05/s15/s30/s60   相对噪声 5% / 15% / 30% / 60%
    timecap_cal   sigma/rho/alpha 取自现有 TimeCAP checkpoint 验证残差的标定产物
    shuffle       整段冻结置换：边缘分布保留，时序摧毁（负控）
    anti          时间反转：边缘分布保留，相位反转（负控）

臂：`perturbed_oracle_planner`（`PerturbedOraclePlannerGlobalScheduler`），
只覆写 `_green_view` 一个方法，与 `oracle144_planner` 之差纯粹是信息质量之差。

## A.2 误差模型（执行前冻结，不得看碳后调）

    view[tau] = max(0, G[tau] + lead_scale(tau-t) * sigma_rel * scale_ref * eps[tau])
    eps         每 (site, tier, episode) 一个冻结 AR(1) 场，rho = 0.8
                误差模式跨决策步持续，不逐步重采样
    lead_scale  alpha + (1-alpha) * lead / H，alpha = 0.25
    scale_ref   该站真值序列在 **本 episode 内** 的平均绝对值
    确定性      一切由 (序列字节, site, tier) 经 sha256 域分离派生
    结算        永远用真值曲线；只有规划器的眼睛被腐蚀

### A.2.1 GPU 侧对交付代码所做的三处修正（本附录一并冻结）

交付的 `forecast_perturb.py` 是对长度 600 的序列写并测的，而 planner 实际传入的是
`self.G[d]`——`CurveInformedPlanner` 无论 episode 多长都把它填满 20000 个网格步
（本网格 episode 实际只有 242 ~ 3361 步）。在这个形状下有三个问题，已修复并各配测试：

1. **剂量轴被 episode 之外的行设定。** `scale_ref` 原先取整条 20000 行的均值。本方案六窗
   间距只有 8072 行，20000 行的量程会跨过其余五个窗口——DISCOVERY 窗的噪声幅度会部分由
   CONFIRMATION 的天气决定。在 2020 数据上实测（**未读取任何 2021 评测行**），
   episode 均值 / 20000 行均值的比值在 0.49 ~ 1.15 之间，31% 的 offset 偏差超过 20%，
   最差接近 2 倍。这会让单调性判据比较的几档根本不在同一条剂量轴上。
   现固定为 `scale_ref = mean(|series[:span]|)`，`span = 该 cell 的 max_episode_length`。
2. **shuffle / anti 的语义与注册不符。** 原实现在整条 20000 行上置换 / 反转，于是
   `anti` 返回的是约 20000 行之外的天气而不是"本 episode 反转"，且实际读取远超本窗口的
   注册读取区间。现全部约束在 `extent = span + horizon` 内，边缘分布保留的对象就是本 episode。
3. **速度：Stage A′ 原本跑不动。** `_costs_all` 每 (作业, 站点) 调一次 `_green_view`，
   而原实现每次调用都重建 AR(1) 场（对整条序列的 Python 循环）并重算一次 160 KB 的 sha256，
   生产形状下实测 **5.732 ms/次**，五个站点即每个被规划作业 28 ms，一个 episode 光造噪声就是小时级。
   现改为每 (site, tier, episode) 构建一次 `FrozenField`（`reset()` 时清空，绝不跨 episode 复用），
   实测 **0.0068 ms/次，加速约 847 倍**，数值逐位不变。

修正只改变可运行性与"腐蚀被约束在本 episode"这一条，不改变阶梯语义、不改变任何判据。
新增 9 个测试（共 26 个），`PLANNER_PERTURB_SPAN` 可显式覆盖 span，
telemetry 增加 `planner_perturb_tier / span / cal / sigma_rel / cal_sha`，
每份 Stage A′ 产物必须回写这些字段。

## A.3 Stage A′ 判据（执行前冻结）

在 Stage A 的**同一冻结网格、同一合同、同一冻结盲臂**上，每格增跑八个 tier。
pooled 与 capture 的定义沿用正文第 7.5 节（和之比，不是比之均值）。

    单调性     pooled 收益（相对冻结盲臂）沿 godeye -> s05 -> s15 -> s30 -> s60 不增
               允许相邻两档打平，不允许任何一档反超 godeye
    负控归零   shuffle 与 anti 各自回吐 godeye 收益的 >= 50%，且不得优于 godeye
    现实档     timecap_cal 保留 godeye 收益的 >= 50%  -> RL 值得做
               < 50% 但 > 0                          -> 记边缘
               <= 0                                  -> STOP_REALISTIC_QUALITY
    合同       全部 tier 每格合同全绿（正文第 7.1 节七项，与 Stage A 相同）

`STOP_REALISTIC_QUALITY` 的含义：完美预报有价值，但已知可达的预测质量兑现不了它。
此时不训 TimeCAP、不跑 RL，负结果照常提交。这是花 CPU 小时就能买到的最值钱的负结果。

## A.4 对正文的影响

- **正文第 6.2 节的 Stage C 阻塞解除。** 负控从 gateway 移到规划器：shuffle / anti 由
  `forecast_perturb` 在 Python 侧实现并已测，不再需要 Java 的 `timecap.forecast_perturbation`。
  正文中"Stage C 不得在 gateway 实现该开关前启动"一条，改由本附录的阶梯替代；
  若日后仍要走真 TimeCAP 输入，原条款照常生效。
- **Stage B 推迟条款不在本附录冻结。** 修正案标注"待 Codex 批准"，工单是冻结文件，
  GPU 侧不单方面改阶段结构。本附录只冻结 Stage A′ 的基建与判据，它对原工单是纯增量。
- **正文第 7、8 节不变**，Stage A 仍先跑，Stage A′ 紧随其后，纯 CPU。

## A.5 窗口读取范围的澄清（正文第 3.4 节补充）

正文称六窗"互不重叠"，指的是**碳被结算的那些行**，即仿真器实际读取的
`footprint_rows = 3977`。规划器另有一套前瞻网格：`CurveInformedPlanner` 把 `self.G` 建满
20000 步，因此它读取的行远超 footprint。这是整个 planner 家族**共有且相同**的行为，
不由 Stage A′ 引入，且这些行从不参与结算。

本附录把加扰限制在 `span + horizon` 内，就是为了不让这条前瞻通道把其他窗口的天气
带进剂量轴。`self.G` 的建表宽度本身不在本轮修改范围内；若后续要求规划器前瞻也严格落在窗口内，
需另开修正案并重选窗口。

## A.6 运行规模（预算申报）

    Stage A'   108 cells x 3 DISCOVERY 窗 x 8 tier = 2592 次仿真

其中 `godeye` 与 Stage A 的 `oracle144` 逐位相同（有测试钉死），因此那 324 次同时充当
两批运行之间的完整性交叉校验；若两者不一致，说明环境或窗口寻址发生了漂移，该批作废。

## A.7 措辞约束（加严）

一切产物只能称 **synthetic forecast-quality ladder**（合成预测质量阶梯），
**不得**写成 TimeCAP 实验。`timecap_cal` 档只能称"标定到现有 checkpoint 已测残差水平的合成档"，
不得称为"TimeCAP 的表现"。正文第 0 节与第 12 节的措辞约束继续适用。

## A.8 Stage A′ 开跑前仍缺的

1. **残差标定未做。** `timecap_cal` 档需要 `residual_calibration.py` 产出的
   `timecap_cal.json`。修正案 §5 第 3 步标注 label-offset 是审计点（工单 §6 的 k=0 语义），
   必须先审计再定值。在该产物存在并提交 SHA 之前，`timecap_cal` 档不可运行；
   其余七档不受影响，可先跑。
2. 正文第 6 节的两项 Stage A 前置（clock zero 复核、冻结盲臂）仍然有效，且优先于 Stage A′。
