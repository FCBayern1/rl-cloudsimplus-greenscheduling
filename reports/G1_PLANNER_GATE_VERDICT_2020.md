# 终判 —— 2020 跨年确认集(2026-08-31 10:35)

依 `reports/PLANNER_GATE_PREREG.md` 及 Addendum A–D 机械执行。**判据在数据之前冻结,未作任何修改。**

    commit   271a031(实现冻结于 2a86e3f + Addendum D 所记两文件改动)
    jar      6d23d8790d3a4d997eb5867c180c0030c5ced264b794d18e32b39ed10de261b5
    config   g1/config_C_2020.yml(wind_csv_year=2020, offset_range=24669)
    windows  low k=27 off 2574 / mid k=71 off 22301 / high k=13 off 13117

## 有效性合同:六格全绿

`deadline_forced_count`、`planner_n_stale_dropped`、`planner_n_unplanned_start`、
`planner_n_wrong_dc`、`planner_n_dispatched_never_started`、`planner_running_pes_over_cap`、
`planner_occ_max_over_cap` 六格全部为零;`completion_rate_mi` 与 `ontime_mi_share` 均为 1.0000。

## 结果

    window   curve_planner   nowait_planner(冻结最强盲)   Δ
    low        0.045522        0.155585                  −70.74%
    mid        0.051796        0.144392                  −64.13%
    high       0.041286        0.138067                  −70.10%
    pooled     0.046201        0.146015                  −68.36%

单位为 terminal carbon / completed MI,取自仿真器真实终端碳账,非规划器内部成本。

## 三条判据

    1  全部合同有效                    MET
    2  池化降幅 ≥ 5%                   MET   −68.36%
    3  ≥ 2/3 窗口方向有利               MET   3/3

**GATE PASSED。**

## 结论范围(预先声明于 Addendum D.6,此处重申)

本结果**只**证明:在 C-regime 上,**完美未来信息**是物理可兑现的 —— 一个共享规划器、
共享容量账、共享空间碳模型,仅在未来信息源上不同的臂,能在合法 legacy backstop 约束下
把终端碳降低约 68%,且完成合同全绿。

**不得**据此声称:

- 真实预测器(TimeCAP)有效 —— 下一步必须换成真实预测重验
- EU-CRD 有效 —— 本轮与该方法无关
- RL 策略应当能兑现这一价值 —— 本轮不含任何学习组件

## 与 2021 公开校准集的一致性

    校准集(2021)  curve 0.053233  blind 0.120579   −55.85%   3/3 同向
    确认集(2020)  curve 0.046201  blind 0.146015   −68.36%   3/3 同向

方向与量级在两年之间一致,确认集上更强。

## 归因限制(不可省略)

冻结最强盲 `nowait_planner` **不允许等待**,故 curve 与它之差同时含「未来信息」与
「等待这一动作本身」。若要拆分,应另比可等待但信息源不同的臂。2021 校准集上:

    curve 0.053233   reactive_wait 0.137379   persistence 0.141976   climatology 0.148092

即便对最强的**可等待**盲(reactive_wait 0.137379),curve 仍低 61.3%。但该对比属于校准集,
且未在 2020 上重复,**不作为终判的一部分**。若需在确认集上定量拆分,须另行预注册并运行。
