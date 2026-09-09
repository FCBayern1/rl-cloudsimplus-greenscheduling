# GPU 工单：COMPRESSED 短视界 TimeCAP 合成正控（方案二）

日期：2026-09-01。执行位置：GPU 服务器。目标分支：
`gpu/compressed-timecap-s2`。

## 0. 一句话任务

从 C-regime 的五 DC 拓扑、风机和功率账出发，建立一个**时间压缩、但在 144 个风电行内闭合**
的合成工作负载。先证明 144 行完美未来相对最强因果盲有价值；只有该零训练门通过，才在 GPU
上重训一个独立 TimeCAP checkpoint，再运行 clean / shuffle / anti / no-forecast，最后才允许
EU-CRD / RL。

本线的身份固定为：

> accelerated-weather synthetic mechanism positive control
>
> 加速天气的合成机制正控

它不进入 REAL_TIME 物理证据链，不改变 C-regime、TB12、TB13-v1/v2/v3/v4 的任何既有判决。

## 1. 当前仓库状态与并行边界

GPU 开工前必须从远端 `main` 拉取本文件所在提交，并记录起点 SHA。

本机另有一条独立 TB13-v4 物理线，当前已经完成：

    功率硬门              PASS
    Round 0-v4 / cohort   已冻结
    零碳 preflight        11/11 PASS
    Phase A               immediate_current_only 已冻结
    Phase B               尚未在本工单授权范围内执行

GPU 线不得改动或覆盖以下目录：

    g1/tb13/
    reports/TB13_V4_PREREG.md
    g1/tb13/round0_v4_out/
    g1/tb13/zero_emission_v4_out/
    g1/tb13/round1_v4_out/

GPU 使用独立分支和独立产物目录：

    git fetch origin
    git switch -c gpu/compressed-timecap-s2 origin/main

    g1/compressed_timecap_s2/
    reports/COMPRESSED_TIMECAP_S2_PREREG.md
    reports/COMPRESSED_TIMECAP_S2_STATE.md

禁止在本轮编辑 `drl-manager/Code/`。这是仓库的 legacy / external 区域。

## 2. 为什么不能直接重训当前 TimeCAP

当前 TimeCAP 的任务是按 CSV 行学习：

    seq_len  = 96 行
    pred_len = 144 行

把仿真器从 REAL_TIME 改成 COMPRESSED，不会改变网络看到的训练样本；它只会把一行从“十分钟”
重新解释成“一个仿真秒”。因此，在相同数据和 96→144 任务上重新训练，不会扩大信息视界。

旧 COMPRESSED C-regime 已有决定性诊断：

    full curve oracle vs nowait     约 -68%（旧功率语义，不得复用数值）
    oracle144 vs climatology        pooled +1.96%，仅 1/3 窗有利
    oracle144 vs nowait             pooled +25.22%，0/3 窗有利
    对 full-oracle 天花板捕获       -3.1%

对应报告：`reports/HORIZON_GATE_VERDICT.md`。它只证明当前规划器和尾部语义下 143 步完美近场
没有产生价值，不是“所有 144 步策略都不可能”的定理。但它足以阻止我们直接烧 GPU 重训同一个
144 行任务。

本工单因此先重建**决策时间尺度**，不是先调神经网络。

## 3. 本轮不得改变的语义

以下语义在方案二中固定：

    time_scaling_mode          COMPRESSED
    green_interpolation_mode   STEP
    一行风电                   一个合成控制 epoch
    TimeCAP seq_len            96 行
    TimeCAP pred_len           144 行
    topology                   C-regime 的五 DC；三站有风机、两站无风机
    wind source                真实 SDWPF 行序列，但时间被明确压缩
    primary metric             仿真器终端总碳 / completed MI
    SLA                        completion_rate_mi 与 ontime_mi_share
    action                     同时允许空间路由与 defer
    negative controls          no-forecast / shuffle / anti

COMPRESSED 结果只能用“行”“epoch”描述。不得把 144 行写成现实中的 144 秒预测，也不得把完整
SDWPF 曲线的未来长度写成现实可获得的预测视界。

旧的 `-68.36%`、旧 jar 和旧 profile 功率语义全部保留，但不得作为本轮输入结果或通过证据。
本轮必须使用修复 profile 功率接线后的新 jar，并冻结 source commit / jar SHA / config SHA。

## 4. 方案二的核心闭合条件

TimeCAP 必须能看到一个候选动作从开始到完成所需的全部未来：

\[
    (s_i-a_i)+r_i \le 144
\]

其中 `a_i` 是到达行，`s_i` 是候选启动行，`r_i` 是运行行数。仅要求“等待不超过 144”不够；
如果作业在第 120 行启动、还要运行 72 行，碳积分仍需要看到第 192 行。

GPU 执行者必须建立 append-only 的 C-regime 衍生 workload/config，不能修改旧 C-regime block。
允许搜索的一级时间轴冻结为：

    runtime_rows          {24, 48, 72}
    wait_cap_rows         {24, 48, 72, 96, 120}
    admissible pair       runtime_rows + wait_cap_rows <= 144
    target concurrency    {1, 3, 5}
    n_jobs                {20, 35, 50}

这些是合成 epoch，不作现实小时解释。到达序列必须从目标并发和 runtime 反解，实测 offered
concurrency 必须记录；禁止用 clipping 把全部到达压到 epoch 0。每个生成 workload 都必须报告：

    arrival span
    offered concurrency
    runtime distribution
    wait cap
    deadline reachability
    workload content SHA256

backstop 必须显式使用 runtime-aware `latest_start` 语义，不能继承隐式 `legacy=600 seconds`；否则
144 行考场会在调度器决策前被 600 秒规则强派。活动键和值必须写进新配置并由测试 pin 死。

## 5. 数据隔离与窗口

先于任何碳结果起草并提交 `reports/COMPRESSED_TIMECAP_S2_PREREG.md`，至少冻结：

1. DISCOVERY 和 CONFIRMATION 的窗口清单；
2. 每个窗口的 offset、实际访问行、互不重叠检查；
3. workload seed、生成公式和哈希；
4. 风机 / 年份 / TimeCAP 训练数据的隔离；
5. jar、配置和源码 manifest；
6. 本文件第 7、9、10 节的机械判据。

推荐隔离方案：

    TimeCAP train / validation   只使用 2020 数据
    scheduler DISCOVERY          2021 的三个冻结窗口
    scheduler CONFIRMATION       2021 的另外三个、与 DISCOVERY 不重叠的冻结窗口

2022 风机文件多数只有两行零值，不得纳入训练或评测。CONFIRMATION 窗口不得按绿电强弱或碳结果
事后替换。若 2021 无法提供六个互不重叠窗口，先 STOP 报告，不得回头读取碳后缩减窗口。

## 6. TimeCAP 数据管线的开跑前审计

现有 `prepare_turbine_data.py` 会把多个 turbine/year 直接拼接，而 legacy `Dataset_Custom` 使用固定
行号划 train/val/test。窗口可能跨越两台风机或两个年份的拼接边界。GPU 线不得假设它天然满足
数据隔离。

在训练前必须交付一个不修改 `drl-manager/Code/` 的外层审计/适配器，机械证明：

    任一 96+144 样本不跨 turbine 边界
    任一样本不跨 year 边界
    train / validation / scheduler evaluation 无行重叠
    scaler 只在 train 上拟合
    k=0 标签语义与 provider 的 k=0 读取语义一致
    输入列顺序与 predictor 的 13 个特征逐位一致
    Patv 单位一致

审计不过即 `STOP_DATA_PIPELINE`，不训练。若必须建立新的 dataset wrapper，应放在
`drl-manager/timecap_prediction/` 或 `g1/compressed_timecap_s2/`，不得编辑 `drl-manager/Code/`。

## 7. Stage A：零训练场景门（先跑，GPU 暂不占用）

对冻结网格的每个 cell，在 DISCOVERY 三窗运行完全相同合同下的：

    strongest causal blind
    full curve oracle
    oracle144（前 144 行真值；超出部分使用与盲臂相同的冻结尾部）

最强盲必须在看到 oracle 结果之前，按 DISCOVERY 全部合同有效实例的 pooled 总碳冻结为**单一臂**；
不得逐实例选择最低碳盲臂。

每格执行合同：

    completion_rate_mi            >= 0.995
    ontime_mi_share               >= 0.995
    deadline_forced_count         == 0
    planner_n_stale_dropped       == 0
    n_unplanned_start             == 0
    n_wrong_dc                    == 0
    n_dispatched_never_started    == 0
    running_pes_over_cap          == 0
    同窗各臂 workload / weather / power / row signature 完全一致

`oracle144` 主门全部满足才算该 cell 通过：

    pooled total-carbon reduction vs frozen blind   >= 5%
    三个 DISCOVERY 窗至少 2/3 方向有利
    capture = (C_blind - C_oracle144)
              / (C_blind - C_full_oracle)            >= 50%
    full oracle 不劣于 frozen blind

分母非正、full oracle 不优或合同不全绿时，该 cell 不得通过。最终必须出现至少三个在预注册参数
邻接关系下相邻的通过 cell；不得只取孤立最大值。稳定区域有多个时，按 canonical cell JSON 的
SHA256 最小值冻结中心 cell，不按效果大小选择。

若 Stage A 无稳定区域：

> `STOP_ORACLE144_GATE`

停止本工单，不训练 TimeCAP、不运行 RL。完整负结果提交并推送。

## 8. Stage A 实现与复用边界

优先复用：

    drl-manager/src/baselines/global_schedulers.py
      CurveInformedPlannerGlobalScheduler
      HorizonLimitedOraclePlannerGlobalScheduler
    drl-manager/src/baselines/evaluate.py
    drl-manager/tests/test_curve_planner_ledger.py
    reports/HORIZON_GATE_VERDICT.md

新增配置必须从一个冻结 base block 程序化派生，并用精确差分测试锁死。不得复制一批手写 YAML
后靠肉眼比较。任何 Java gateway 改动都必须执行：

    cd cloudsimplus-gateway
    ./gradlew -q compileJava
    ./gradlew test

Stage A 是 CPU / 仿真器工作。即使 GPU 空闲，也不得跳过它先训练。

## 9. Stage B：TimeCAP 重训（仅 Stage A PASS 后）

训练身份固定为新 checkpoint，不覆盖现有 TimeCAP：

    model input history     96 rows
    prediction target       next 144 rows
    target                  Patv
    seed                    20260901
    primary run             single GPU
    checkpoint selection    validation loss only
    scheduler carbon        不得参与选 epoch、学习率或 checkpoint

训练命令的基本入口位于 `drl-manager/timecap_prediction/train_timecap.py`。GPU 执行者必须先运行
一轮 1 epoch smoke，确认 CUDA、checkpoint、`model_args.json` 和 predictor 回读，再启动正式训练。

示例形态（实际路径必须来自预注册 manifest）：

    cd drl-manager
    .venv/bin/python -m timecap_prediction.train_timecap \
      --data-csv <frozen-train-csv> \
      --res-dir <new-s2-result-dir> \
      --epochs 30 --batch-size 64 --lr 5e-5 --patience 5 --gpu 0

不要因为服务器有多张 GPU 就直接启用 `--multi-gpu`；现有脚本的分布式入口必须先做 1 epoch
`torchrun` smoke 和 checkpoint 单写者检查。单卡能在预算内完成时，优先单卡以减少新变量。

训练完成必须保存：

    checkpoint SHA256
    model_args.json SHA256
    train-data manifest SHA256
    code commit
    CUDA / torch / GPU 型号
    best epoch 与完整 validation curve
    test MSE / MAE（只报告，不用来回调 scheduler）

大 checkpoint 不要求提交 Git；存入持久卷并在 Git 中提交路径、大小和 SHA256 manifest。

## 10. Stage C：预测价值门

使用 Stage A 冻结的同一中心场景、同一窗口、同一 workload、同一动作空间和同一尾部，运行：

    matched no-forecast
    TimeCAP clean
    TimeCAP shuffle
    TimeCAP anti
    oracle144（冻结参照，不重算场景）

主门：

    clean vs matched no-forecast pooled 总碳下降   >= 5%
    DISCOVERY 三窗至少 2/3 方向有利
    全部完成与容量合同通过
    clean 动作相对 no-forecast 非全等
    clean 收益必须被 shuffle 和 anti 明显破坏

“明显破坏”须在 `COMPRESSED_TIMECAP_S2_PREREG.md` 中于执行前量化，推荐冻结为：两个负控各自
至少回吐 clean 对 no-forecast 收益的 50%，且不得比 clean 更优。不得在看到负控结果后改阈值。

CONFIRMATION 只在 DISCOVERY 主门全过、checkpoint 与场景均冻结后读取一次。确认集要求 pooled
方向有利且至少 2/3 窗同向；不在确认集重新选 checkpoint、场景或盲臂。

若 Stage C 失败，记录 `STOP_TIMECAP_VALUE_GATE`，不运行 EU-CRD / RL。

## 11. Stage D：EU-CRD / RL（仅 Stage C PASS 后）

先做 1 seed / 50k smoke，不直接开完整训练。最少包含：

    matched no-forecast
    TimeCAP clean + matched vanilla credit
    TimeCAP clean + EU-CRD

50k 健康门：

    奖励改善时物理总碳必须同向下降
    cap / SLA / completion 合同全绿
    clean 与 no-forecast 的 argmax 动作必须出现差异
    forecast 特征非恒定、非饱和
    EU-CRD 的 delta-r 通道非恒零
    不允许全 route / 全 defer 以“门通过”冒充学习

50k 通过后，另写 append-only 长训预注册，再决定配对种子和训练预算。本工单不授权看到 smoke
后调奖励、改门限、移动判定点或直接延长训练。

## 12. 提交纪律与交付格式

每个阶段必须在运行前提交定义、运行后提交产物，建议提交顺序：

    1. prereg + config generator + tests
    2. Stage A DISCOVERY artifacts + verdict
    3. TimeCAP data audit + train manifest
    4. checkpoint manifest + training report
    5. Stage C DISCOVERY artifacts + verdict
    6. Stage C CONFIRMATION artifacts（若获准）
    7. 50k smoke prereg / artifacts（若获准）

每次提交前：

    git status --short
    git diff --check
    运行最近的 Python 测试
    若动 Java，运行 compileJava + Java tests

只添加本任务明确产生的文件，不得使用 `git add .`。仓库里已有未跟踪的 micro traces、
`Turbine_900_*`、`g1/micro/` 等用户文件，禁止顺手提交或删除。

GPU 分支完成后：

    git push -u origin gpu/compressed-timecap-s2

最终回报必须用一段话先给出当前停在哪道门，再附：commit、artifact SHA、逐格合同、GPU 小时、
checkpoint manifest 和下一步是否获准。不得只汇报“训练完成”。

## 13. 成功与失败各自能声称什么

若全线通过，只能声称：

> 在一个明确标注的 accelerated-weather synthetic C-regime 正控中，144 行 TimeCAP 预测包含
> 可执行的调度价值，负控会破坏该价值；若后续 RL 也通过，则 EU-CRD 能在该合成机制正控中
> 利用预测信息。

不得声称真实十分钟 SDWPF 时间尺度、现实 24 小时预测或生产数据中心具有同等收益。

若 Stage A 失败，说明当前被冻结的短视界合成设计仍没有为 144 行预测提供决策价值；不是
TimeCAP 训练失败。若 Stage A 通过但 Stage C 失败，才说明“完美 144 行未来有价值，但当前
TimeCAP 没兑现”。若 Stage C 通过而 RL 失败，断点才位于学习 / 信用分配。

这个三段定位是本工单最重要的产出，任何一段的 STOP 都是有效结果。
