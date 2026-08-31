# 2021 校准判决 —— 最强盲冻结(2026-08-31 03:05)

> **LEGACY ACCELERATED-WEATHER DIAGNOSTIC (2026-08-31)**
>
> 本文件的全部数字产生于 `time_scaling_mode: COMPRESSED`,该映射令 `wind_row = sim_step + offset`,
> 即**风电以 600 倍于仿真数据中心时钟的速度前进**(SDWPF 行距 10 分钟,仿真步长 1 秒)。
> 论文正文同时声称 SDWPF 为 10 分钟 SCADA、workload 运行于 7200 仿真秒、控制间隔 1 仿真秒,
> 三者与该映射不自洽:deadline、runtime 与能量积分按秒计,而天气按 600 倍走。
>
> 据此,本文件**退出物理证据链**,仅作加速天气基下的诊断保留。不得作为 C-regime 的物理判决引用。
> 修复后的时间基为 `wind_row(t) = offset + floor(t / 600)`,须在该基上重跑。


依 `reports/PLANNER_GATE_PREREG.md` 及 Addendum A/B/C。**写于 2020 数据被任何进程读取之前。**

    commit  2a86e3f
    jar     e1aeba94e154eb01152482da097df5cbcb1f4018b4e32497596aae96684067d5
    config  ca4fcf76ed4b97b8d972613e604dadd5453fef8cb90f0a90b285fe6f17a84d73

## 有效性合同:15 格全绿

每格 `deadline_forced_count`、`planner_n_stale_dropped`、`planner_n_unplanned_start`、
`planner_n_wrong_dc`、`planner_n_dispatched_never_started`、`planner_running_pes_over_cap`、
`planner_occ_max_over_cap` 全部为零;`completion_rate_mi` 与 `ontime_mi_share` 均为 1.0000;
workload 均为 8000。

    arm                     pooled     high       low        mid
    curve_planner           0.053233   0.053041   0.055669   0.050989
    nowait_planner          0.120579   0.112166   0.125985   0.123587
    reactive_wait_planner   0.137379   0.128202   0.142277   0.141660
    persistence_planner     0.141976   0.142996   0.144303   0.138628
    climatology_planner     0.148092   0.140675   0.148954   0.154647

## 冻结最强盲

盲臂中三窗池化 terminal carbon/MI 最低者为 **`nowait_planner`**(0.120579)。
按预注册,冻结为终判对手,**不得逐窗更换**。

`curve_planner` 是信息臂,不参与盲冻结。

## 校准集上的差值(不是判决)

curve 相对冻结最强盲为 −55.85%,三窗同向 3/3。**这是校准集,不构成 5% 判决。**
判决只在 2020 三窗上进行一次。

## 一处必须在解读时保留的限制

`nowait_planner` 与 `curve_planner` 的差异同时包含两部分:未来信息,以及等待这一动作本身。
`nowait` 不允许等待,故二者之差是「信息 + 协调」的合计。若需拆分,应另比
`curve` 与 `persistence` / `reactive_wait`(两者均可等待,仅信息源不同)。
校准集上该对比为 curve 0.053233 对 persistence 0.141976 与 reactive_wait 0.137379。
终判仍按预注册以冻结最强盲为准。
