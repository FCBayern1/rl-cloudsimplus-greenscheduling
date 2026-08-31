# curve_oracle 单窗口 smoke —— 作废(2026-08-30)

**结论**:本轮仅证明 curve-informed scheduler 能接入环境并完成近乎全部工作,不构成 oracle headroom 测量。
除截断和比较器不对称外,代码审查发现 padding 虚假承诺、deferred 作业重复承诺、计划生命周期缺失及容量/功率
近似四项 P0,因此 **−47.92% 整体作废**。该臂当前应称为 **curve-informed heuristic v0 — invalid**,不是
oracle,不进 5% 判决,不作为 headroom,不进论文。

## 跑了什么
C-regime,冻结 jar `aba6f0ed…`,低绿窗 k=19(`ORACLE_OFFSET_ROWS=19171`),单 episode,seed 20260823。

    blind   green_queue_balanced + drain        (无 defer)
    oracle  curve_oracle + drain + --global-defer

## 出的数(作废,仅存档)
                        blind      oracle      Δ
    carbon/MI          0.129579   0.067503   −47.92%
    completion_rate_mi 1.0000     0.9998
    ontime_mi_share    1.0000     1.0000
    green_used_wh      581.4      460.7      −20.76%
    green_waste_wh     1363.8     1782.4     +30.70%
    total_energy_wh    1020.9      833.9     −18.32%
    episode_length     6105       7200       (oracle 撞 7200 上限)
    routed/received    8000       7997       (workload 两臂同为 8000)
    deadline_forced    0          49
    received_dc[0..4]  [4930,2,3066,1,1] → [7344,148,25,412,68]

## 四个 P0(已逐条读代码核实)

1. **padding 被当成真实作业规划** —— `schedule()` 遍历全部 128 槽,`p = max(1.0, pes[j])` 把 pes=0 抬成 1,
   `r = max(1, ...)` 把 mi=0 抬成 1,随后无条件 `committed[d, s:s+r] += p`。空槽每步向未来容量账写虚构负载。
   `global_schedulers.py:1305-1330`
2. **同一 deferred 作业重复承诺** —— 类中不存在 `committed` 的任何减法或撤销路径;deferred 作业下一步重新
   出现在队首即被重新规划并再次累加,旧承诺永不撤销。
3. **计划没有被执行** —— 只记未来容量占用,不记「作业 i 应在时刻 s 去 DC d」。每步从零重规划;Java backstop
   强派后 Python 侧旧承诺也不清除。
4. **容量与功率只是近似** —— `cap_pes = h*64.0`、`static_w = 332.0*h/tot`、`dyn_per_pe = (214−51.4)/64`
   对所有 DC 一律相同,而本考场 DC0/1/4 是 rs500a、DC2/3 是 rs700a,idle power-down 也不同。

附:`ORACLE_HORIZON=400` 是有限视界,`starts` 网格为 `r//8` 粒度。二者都不是完整 deadline curve oracle,
必须据实改名或扩到作业可行等待窗。

## 我在初版报告里写错、现已纠正的两条

- **DC0 是最干净的棕电站点,不是最脏。** `experiment_g1eval_matchedvan`(`config_C.yml:44212`)的因子是
  `[0.08, 0.35, 0.55, 0.75, 0.92]`。我最初 grep 打到的 0.7 属于文件顶部另一实验的模板块(82–280 行)。
  因此 92% 路由至 DC0 与碳下降**不冲突**,反而提示 −48% 的多数可能来自空间碳因子(DC2 的 0.55 → DC0 的 0.08)、
  主机型号打包效率与更低的总能耗,而非未来信息带来的绿电捕获。
- **`created=7997` 不是 workload。** `total_created_cloudlets` 在 Java 侧表示「已路由/已被 DC 接收」
  (`rllib_green_energy_logger.py:719`),真实 workload 是 `total_cloudlets=8000`,两臂相同。准确表述:
  oracle 在截断时已路由并完成 7997/8000,仍有 3 个作业未完成。终端账不完整,但两臂 workload 并无差异。

## 仍然成立的两条比较器缺陷

- **盲态被写弱**:blind 无任何时间杠杆,oracle 有 defer,差距里混入时间杠杆本身。
  `green_queue_balanced` 不是合格主盲:不使用 brown factor、不按 PES 扣容量、无等待动作、目标函数与 oracle 不同。
- **`--global-defer` 的包装是覆盖式的**:`DeferringGlobalScheduler` 只加 defer 不撤 defer,默认
  `forecast_thresh=0.3` 而本考场归一化 forecast 落在 ~[0,0.05],推测整轮未开火,但日志未打印 defer 计数,未证实。

## 最强联合因果盲(Codex 裁定)

oracle 与盲臂必须**共用同一规划器、容量账、空间碳成本与计划生命周期**,唯一替换未来信息源:

    oracle_curve    真实未来曲线
    persistence     未来等于现在
    climatology     只用历史校准出的因果分布
    reactive_wait   当前绿电相对作业需求的等待门
    nowait          同一空间成本模型,立即执行

先在校准窗口按「完成合同合格后 terminal carbon 最低」冻结最强盲,再去 held-out 比较。

## 终止口径(Codex 裁定)—— 双时点合同

1. **注册决策边界 H=7200**:报 completion@7200、carbon@7200、pending/running MI@7200、deadline-forced@7200。
2. **统一 terminal drain**:max horizon 扩到 10000–12000;7200 之后禁止新增可选 defer,剩余作业按各臂已承诺
   目标或共同冻结 fallback 路由,只 drain 到自然完成。
3. **主账用 terminal 指标**。有效对要求:两臂 workload 均为 8000;terminal routed=finished=8000 或 MI 完成率
   ≥99.5%;terminal ontime MI ≥99.5%;同报 terminal carbon、total energy 与完成时刻。

不让两臂机械跑满固定步数 —— 那会给先完成的一臂添加无业务意义的空闲能耗。

## 能耗审计(要做,但理由不是「违反物理」)

选择低碳棕电站点、减少总能耗,本来就可能比多用绿电更低碳。要做的是逐 DC 分解并验恒等:

    C = Σ_d ( E_green,d · c_green,d + E_brown,d · c_brown,d )

同报每 DC 的 green/brown Wh、静态/动态 Wh、active-host 时间、完成 MI、received/finished、加权棕碳贡献。
若该恒等式与 global carbon 逐位吻合,账本大概率没坏;届时 −48% 的来源可明确拆成**空间碳因子 / 主机能效与打包 /
未来信息导致的时间移动**三部分。

## 执行顺序(冻结)

1. 修 oracle 计划生命周期与 padding
2. 加零训练计划账单元测试
3. 逐 DC 能量—碳恒等审计
4. 构造同规划器、仅信息源不同的盲候选
5. 用双时点 / terminal 口径重跑同一窗口
6. 通过接线与有效性门后,数字才允许碰 5% 止损线
