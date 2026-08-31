# 视界门 —— 判据冻结(2026-08-31 12:02,写于读取 oracle144 数字之前)

依 Codex 2026-08-31 裁定。本文件在 `oracle144_planner` 三格结果被读取之前写定,
之后只做机械读取,不得修改分母、公式或合同。

## 1. 这道门要分开的两件事

TimeCAP 若不过门,只有两种原因:**预测不够准**,或**预测视界先天太短**。
`oracle144` 用完美预测截断到真实预测器的视界,再接共用因果尾部,一次把两者分开。
若连它都过不了,则任何预测器都不可能过,**直接停止,不接 TimeCAP**。

## 2. 视界的准确表述

`forecast[k]` 声称描述绝对行 `origin_row + k`(编码预测器逐位验证,见
`g1/audit_forecast_time_alignment.py`)。**`k = 0` 对应当前已观测行**,
故真正的未来视界是 `k = 1 … 143`,即 **143 步**。

报告中一律按 143 步未来视界表述。`pred_len = 144` 是输出点数,不是未来步数。
`csv_row = sim_step + offset` 为 1:1 映射,`simulation_timestep = 1.0`,
故 143 点即 143 个规划步。可等待窗口约 5898 步,覆盖率 **2.4%**。

## 3. 两道门,必须同时满足

**主配对门(信息增量)**

    (C_oracle144 − C_climatology) / C_climatology  ≤  −5%

分母为 `climatology_planner`。二者动作空间、容量账、backstop、尾部模型**完全相同**,
唯一差异是前 143 步是否为真实未来。差值即短视界未来信息的增量。

**竞争力门(是否打得过最强盲)**

    (C_oracle144 − C_nowait) / C_nowait  ≤  −5%

分母为 2021 校准集冻结的最强盲 `nowait_planner`。防止只赢过一个较弱的同构尾部。

**共同要求**

    至少 2/3 窗口方向有利(两门各自计)
    六项执行合同全绿(Addendum B.6),完成率与准时率 ≥ 0.995

## 4. 分支

    两门均过        短视界信息足够且具竞争力 → 接 TimeCAP clean/shuffle/anti
    仅主配对门过    短视界含信息,但当前混合控制器不具实用竞争力 → 暂不接 TimeCAP
    主配对门不过    视界先天太短 → 停止,不烧 TimeCAP,也不烧 RL

## 5. 尾部模型

    tail_model = climatology(窗口之前的历史均值,因果)
    horizon    = 144 输出点 = 143 步未来

`clean / shuffle / anti` 未来若接入,必须使用**完全相同**的尾部模型与视界长度。

## 6. 参与比较的结果文件哈希(SHA256 前 16 位)

    climatology_planner_low_m0.csv    f15a618f39e59c5a
    climatology_planner_mid_m0.csv    f95b2aeef7260aa4
    climatology_planner_high_m0.csv   1312fc03308154c1
    nowait_planner_low_m0.csv         debe03b56e31fb4f
    nowait_planner_mid_m0.csv         2995c7f6f10620a8
    nowait_planner_high_m0.csv        1427583f1899178b
    oracle144_planner_{low,mid,high}_m0.csv   写定本文件时尚未生成,读取时补记

    jar     6d23d8790d3a4d997eb5867c180c0030c5ced264b794d18e32b39ed10de261b5
    config  ca4fcf76ed4b97b8d972613e604dadd5453fef8cb90f0a90b285fe6f17a84d73
    windows 2021 校准三窗 low k=19 / mid k=56 / high k=34(Addendum A)

## 7. 接线审计的遗留记录

    年份一致性        timecap.csv_year == wind_csv_year,不等则构造期失败
    时间对齐          确定性哨兵通过,forecast[k] 逐位声称 origin_row + k
    完整曲线          get_raw_forecast_per_dc 提供轨迹,非均值摘要
    无未来泄漏        历史缓冲仅含 ≤ t 的行;实测最大相关 0.9830 < 1
    预测质量诊断      DC0 r≈0.97 / DC1 r≈0.95 / DC2 r≈0.80 —— 仅诊断,不参与判决
    预报覆盖 DC0-2    DC3/DC4 无涡轮,与规划器一致
