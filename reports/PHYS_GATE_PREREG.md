# 物理时间基门 —— 判据冻结(2026-08-31 14:05,写于任何臂的碳被读取之前)

依 Codex 2026-08-31 裁定。加速天气基下的全部结果已标记为 legacy,退出物理证据链。

## 1. 时间基

    wind_row(t) = registered_offset + tz + floor((clock0 + t − origin) / 600)
    time_scaling_mode        REAL_TIME      (rowSeconds = 600)
    green_interpolation_mode STEP           (行 i 在其 600 秒单元内恒定)
    green_power_scale        1/1500         (模式无关乘子,替代只在 COMPRESSED 生效的 divisor)
    origin                   0              (与 Java 一致,天气自时钟零点流逝)
    clock0                   每格从观测的 current_clock 读取,非估计

规划器不再持有 `warmup = 13` 魔数;风电网格在首次决策时按真实时钟铺开,与 Java 同源。

**启动相位门**:`floor((clock0 − origin) / 600) != 0` 即失败,拒绝静默平移窗口。

**分段签名**:`planner_rows_signature` 哈希完整分段表 `(dc, row_index, overlap_seconds)`,
非仅首末行。同窗不同臂签名必须相同。

    一格读 21 行(12000 步),首行服务 587 秒,窗口占 129 行(21 + 最大时区偏移 108)

## 2. 窗口选择(规则先于结果)

    候选量      三站实际分段时长加权总绿能
    分位        经验分位 p10 / p50 / p90
    取点        距目标分位最近的实际候选窗口
    tie-break   最小 offset
    约束        三窗的全部逐涡轮读取区间互不重叠
    若需替换    在所有合法三元组中最小化三项分位秩误差之和,再以 offset 字典序破平

    name     k   offset   green Wh   rank pct   read rows
    low   2528    33552     332.60       10.0   [33552, 33626]
    mid   3279    27161    3033.92       50.0   [27161, 27235]
    high   869    22771    9832.38       90.0   [22771, 22845]
    disjoint: True     首选即合法,未触发替换
    selection hash: fa8a53115b82cfbb

不用 min/max:它们估计的是年度极端事件而非贫/中/富风制度,且无风端机制中性,放进主判决只会
机械稀释方向门。

**主结果称为「等权三制度池化」**,不得称为 2021 年度平均 —— p10/p50/p90 是分层抽样,不按全年频率加权。

**负控(不参与成功判决)**:最小绿能窗 k=43 offset=43387 green=0.00 Wh,机制中性。
主门若通过,补跑 full oracle 与冻结最强盲两格,预期趋同 —— 证明规划器不会在没有绿电时凭空造收益。
**不进 2/3 方向分母。**

## 3. 五臂

    curve_planner          真实未来曲线(信息臂,不参与盲冻结)
    nowait_planner         同一空间成本模型,不许等待
    persistence_planner    未来 = 当前电表读数
    climatology_planner    窗口之前的历史均值,因果
    reactive_wait_planner  因果停止规则,不预约未来容量

五臂共用规划器、容量账、backstop、逐 ID 合同与时间基。**唯一差异是未来信息源与因果规则。**

**逐格核对**:同 `planner_rows_signature`、同 `planner_clock0`、同 workload/涡轮/年份哈希。

## 4. 最强盲与判据

四个盲臂中三窗**等权池化** terminal carbon/completed-MI 最低者冻结为最强盲,不得逐窗更换。

    (C_curve − C_strongest_blind) / C_strongest_blind  ≤  −5%
    至少 2/3 窗口方向有利
    六项执行合同全绿(Addendum B.6),完成率与准时率 ≥ 0.995

不到 5% 则 C-regime 在物理时间基下**干净 STOP**;过 5% 则直接进 TimeCAP clean/shuffle/anti
(24 小时视界已覆盖 2 小时 episode,不需视界扫描),但须先完成 TimeCAP 标签语义与 96 行因果预载审计。

## 5. 哈希

    jar          6d23d8790d3a4d997eb5867c180c0030c5ced264b794d18e32b39ed10de261b5
    phys config  79f7e6fd9e36c3e9e447643cdcce24e9b9d25ad8dff5422c8edfaea88ddeb492
    2021 涡轮    T12 e6512599834c352a  T36 14452bd95440401e  T91 6279066901ab3053
                 T95 32ac9baf98d1ac88  T96 ff5df78a51c09f1d
    workload     traces/probe_C_2xjob_dl6500.csv,8000 作业
