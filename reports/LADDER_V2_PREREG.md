# Ladder-v2 预注册:CONFIRMATION 主判决的加扰阶梯

按 Codex 2026-09-02 裁定起草。A′ 的 STOP_LADDER_GATE 永久有效;旧 A′ 数据不按新口径改判,
只作为选择 v2 估计量的 DISCOVERY 依据。v2 的**正式主判决使用从未读取的 CONFIRMATION 三窗
k=25/33/41(offset 25225/33297/41369)**,一次性运行,不得预览。

执行位置:本机(8 核主机,`joshua@…`,main 权威线)——执行位置变更与 A′ 执行均已获追认,
本文件即所要求的 append-only 记录(主机、commit、时间以 git 记录为准)。

## 1. R3:全档强制 lead-0 真值(语义不变量,不是第九档)

当前行已实现,属于观测不属于预测。v2 全部档位统一:

    lead 0            使用真实当前值,逐位等于真值
    污染范围          仅 lead 1 到 143
    重新规划          每步的新当前行恢复真值(视图按 t 重算,自动满足)
    shuffle / anti    同样不得改变当前行
    不变量            green_now、workload、动作空间、结算真值全档完全相同

## 2. R2:checkpoint_residual_surrogate_v2(DC 级机械标定,禁止手写)

只用 2020 校准数据,按部署路径聚合:DC0 = T12+T36,DC1 = T95+T91,DC2 = T96。
锚点跨 DC 同步(stride 480,label-offset 0 沿 k=0 审计)。机械产出:

    sigma_rel_d       各 DC 残差 std ÷ 该 DC 真值平均绝对水平
    ar1_rho           各 DC 沿 lead 轴的滞后 1 自相关,取三 DC 中位数
    lead_alpha        各 DC lead0/lead143 残差 std 之比,取三 DC 中位数
    相关矩阵          三 DC 标准化残差的 3x3 相关阵(同锚点同 lead 展平)
    c                 三个非对角元的中位数,不四舍五入

生成模型(单因子):

    eps_d = sqrt(c) · eps_common + sqrt(1 - c) · eps_d_independent

eps_common 由 episode 级 key(全站真值字节)派生,各站共享;独立场按站派生。构造上
边际方差 = 1、AR(1) 保持、两两相关 = c 精确成立;**复现容差**(预注册冻结):生成误差的
三站两两相关与实测矩阵逐元素偏差 max|r_ij − c| ≤ 0.10。超差则不加参数拟合,改用
2020 同步残差窗口的 deterministic residual replay(另立 addendum 后执行)。
该档命名 `checkpoint_residual_surrogate_v2`,一切文字不得称其为 TimeCAP 本身。

## 3. R1:估计量(逐格归一中位保留率)

每格 i、每臂 q,先跨三个 CONFIRMATION 窗聚合为碳强度:

    I_{q,i} = (Σ_w carbon_w) / (Σ_w completed_MI_w)
    completed_MI_w = total_finished_cloudlets_w × mi_per_job(cell)

保留率与判决统计量:

    R_i(q) = (I_blind,i − I_q,i) / (I_blind,i − I_godeye,i)
    分母必须为正;分母 ≤ 0 的格从中位数中排除并逐格报告,
    排除格超过冻结区域的 20% 则本轮 INVALID
    判决量 = 冻结 97 格区域上的 median_i R_i(q)

盲臂 = Stage A 冻结的 nowait_planner,在同三窗同格重跑(它此前从未跑过 CONFIRMATION)。

## 4. 判据(数值保持,口径为 §3)

    单调         中位保留率沿 godeye→s05→s15→s30→s60 单调不增
    负控         shuffle 与 anti 的中位保留率均 ≤ 50%
    现实档       checkpoint_residual_surrogate_v2 的中位保留率 ≥ 50%
    合同         全部臂 × 97 格 × 3 窗:completion_rate_mi ≥ 0.995、ontime_mi_share ≥ 0.995、
                 六个零字段为零(无台账臂缺列按零)
    零点         godeye 与真值视图逐位一致(单元测试钉死)

任一门失败 → Scheme 2 STOP;不得以"机械等待仍有收益"为由绕过负控直接训练 RL。

## 5. R4:Stage D 的主张(预告,另立预注册)

"信任校准"表述驳回。v2 全门通过后,Stage D 预注册的主张为**质量鲁棒信用分配**:

> 在相同预测质量与完全匹配的训练条件下,EU-CRD 是否比 Vanilla 更能保留干净预测价值,
> 并减少预测腐坏造成的碳损失。

至少五臂:matched no-forecast / Vanilla+good / EU-CRD+good / Vanilla+corrupted /
EU-CRD+corrupted。正面闭环六条(EU-CRD good 对 no-forecast 总碳 ≥5% 降、good 档不得靠
忽略预测装鲁棒、corrupted−good 增量 EU-CRD < Vanilla、配对种子同向、SLA/cap/完成/奖励-物理
方向门全绿、crd_dr 非零且 c_t 不贴边且 rho 有方差)。历史 G1 中 EU-CRD 在 shuffle 下劣于
Vanilla 的观察记录在案,不预设相反机制。逐档独立训练的结果只得称 forecast-quality
robustness;trust calibration 须另立"冻结质量混合分布训练 + 跨档评估"实验方可声称。

## 6. 执行阶梯

    本预注册提交
    → 纯函数实现 + 测试(godeye 逐位闭合 / lead0 全档不变 / 仅污染 lead 1–143 /
      DC 级协方差复现 / 结算真值 / 2020 标定与 2021 CONFIRMATION 无交集)
    → DC 级标定产物提交(2020,机械)
    → 冻结 97 格区域 × k=25/33/41 × (盲臂 + 8 档) 一次性运行
    → 判读器(先于运行提交)机械执行 §4
    → 全过才起草 Stage D 50k 预注册

## Addendum A(append-only,记录性,不改判据)

标定产物的跨机复现性经 3060 复核:provenance 字段(锚点数、stride、label offset、
风机、全部输入 SHA、scale_ref)逐字节一致,拟合标量随 BLAS 线程数在 ~1e-7 相对量级漂移,
同线程数重跑逐字节相同。处置:两个标定脚本自此钉死单线程推理并把线程数写入产物,
今后的再生成跨机字节可复算;**既有两件产物不再生成**——timecap_cal.json 是已关闭的
A′ 的冻结输入,dc_residual_cal.json 是在跑的 v2 一次性扫描的冻结输入,二者的跨机声明
一律为"provenance 逐字节一致 + 拟合值相对偏差 < 1e-6",不得声称字节可复现。

两个开放项照登,不得视为已解决:①k=0 标签语义仅验证"可跑通且锚点数一致",其语义
审计(工单 §6)仍开放;②3060 的 1-epoch smoke 数字仅证明管道通,不可引用;其 CUDA
因内核模块不匹配未验,待修复后补真 GPU smoke。
