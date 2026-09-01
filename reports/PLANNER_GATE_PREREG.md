# 规划器门 —— 预注册(冻结于 2026-08-30,任何确认运行之前)

> **LEGACY ACCELERATED-WEATHER DIAGNOSTIC (2026-08-31)**
>
> 本文件的判据与窗口属于 `time_scaling_mode: COMPRESSED` 的加速天气考场,
> 其中 `wind_row = sim_step + offset`,风电以 600 倍于仿真时钟前进。
> 该考场已被时间基审计判为与论文语义不自洽,相关判决已退出物理证据链。
> 保留作历史记录,不得作为 C-regime 的物理依据。物理时间基的预注册见
> `reports/PHYS_GATE_PREREG.md`,判决见 `reports/C_REGIME_PHYSICAL_TIMEBASE_VERDICT.md`。


依 Codex 2026-08-30 裁定。窗口、margin 网格、有效性合同与哈希在此冻结,
之后不得修改判据、不得换窗口、不得在看到结果后重选盲臂。

计算脚本 `g1/prereg_windows.py`,机读输出 `g1/prereg_windows.json`。

---

## 1. 被测对象

**信息臂**(不参与盲冻结)

    curve_planner            真实未来曲线

**盲候选**(四臂,共用规划器 / 容量账 / 空间碳成本 / 计划生命周期)

    nowait_planner           同一空间成本模型,不许等待
    persistence_planner      未来 = 当前电表读数
    climatology_planner      窗口之前的历史均值,不读被测窗口
    reactive_wait_planner    因果停止规则,不预约未来容量;
                             够绿则走,不够则等,到 margin 无条件路由

五臂唯一的差异点是 `_green_view(d)`、`ALLOW_DEFER` 与 `RESERVES` 三处,
有单元测试断言五臂的 `cap / cb / cg / mips / dyn_per_pe` 逐位相同。

**命名约束(Codex 硬性)**:只能称 curve-informed feasible planner。
不得称 oracle、最优策略或物理 headroom 上界。功率模型本轮为近似档。

## 2. 窗口

偏移遵循仿真器自身的日程 `offset(k) = (1009k) mod range`(`MultiDatacenterSimulationCore:200`)。
每窗跨 `warmup 13 + max_tz 108 + episode 7200 = 7321` 行。

**校准集 —— 2021,range 44950(配置值),已核对逐位一致**

    low    k=19   offset 19171   末行 26492   ok
    mid    k=56   offset 11554   末行 18875   ok
    high   k=34   offset 34306   末行 41627   ok
    三窗互不重叠

**确认集 —— 2020,range 未定,见 §6 待裁**。两个候选的三窗都能装下且互不重叠。

**2022 不可用**:`windProduction/simplified` 下 134 个 2022 文件**全部只有 2 行、功率 0**。见 §6。

## 3. margin 网格(预注册,不得事后调)

    backstop_margin ∈ {2, 4, 8, 16, 32, 64} 步

在 **2021 三窗**上,取使**全部五臂同时满足有效性合同**的**最小** margin,冻结之。
进入 2020 后若出现 forced,**直接判无效**,不再调 margin。

## 4. 有效性合同(每一格都必须全绿)

    deadline_forced_count == 0
    terminal completion_mi        >= 99.5%
    terminal ontime_mi            >= 99.5%
    完整 workload 被创建,两臂工作量一致
    n_stale_dropped == 0
    无容量越界
    决策点(H=7200)与 terminal 碳均报告
    预测占用与逐 DC 实测负载一致(漂移哨兵,见 §5)

## 5. 双时点终止

    决策边界      H = 7200 步。此后对所有臂同时关闭等待
    终端上限      12000 步(先在校准集验证足够)
    边界后处理    已开预约按各臂【已承诺的站点】立即拉起,只去掉等待,不改站点
    主账          terminal 指标

**漂移哨兵**:`active` 是规划器推算的运行账,不是 Java 的实际启动事件。
每格必须比对规划占用与逐 DC 实测负载。出现队列或启动漂移则本轮无效,
不得用账本自洽掩盖。

## 6. 逐 DC 能量—碳恒等审计

    C = Σ_d ( E_green,d · c_green,d + E_brown,d · c_brown,d )

`g1/energy_carbon_identity.py`,已在 smoke 的 blind trace 上逐位闭合
(carbon / green / brown / total energy 全部 rel=0.000%,waste 0.006%)。
每格都要跑,并报每 DC 的 green/brown Wh、spill、碳份额、received。

## 7. 终判口径(2020 三窗)

    主量        池化 terminal carbon / completed-MI
    通过条件    curve 相对冻结最强盲降幅 ≥ 5%
                且 ≥ 2/3 窗口方向有利
                且所有窗口两臂合同有效
    同时报告    decision-horizon 与 terminal 两套数字,防尾部逃账

**比较必须用仿真器真实 terminal carbon,不得用规划器内部成本。**

失败时只能否定当前规划器实例,**不得据此宣布该场景没有预测价值**。
通过后才做逐 host 型号 + idle power-down 的精确功率模型。

## 8. 冻结哈希

    config_C.yml   4ec9b8ef6b7927d4e4d3af8c1e5200b64370662e31f759575c19ebf18e9bb414
    gateway jar    824bd8573d0b328d91d344fbb7c71f6b451615932ad4c156aa3a355ade1461bb

    turbine        2021 (前 16)        2020 (前 16)
    T12            e6512599834c352a    162e80e50b31a720
    T36            14452bd95440401e    e92090af7abd76ec
    T91            6279066901ab3053    c45df4f3d2d0ba66
    T95            32ac9baf98d1ac88    ee5e68df0004ab16
    T96            ff5df78a51c09f1d    7ac6ce1a70321b17

    每涡轮行数     2021 = 52559    2020 = 32224    2022 = 2(桩)

冻结 G1 jar `/home/joshua/frozen/g1_gateway/lib` 只读未动,与本轮无关。

## 9. 执行顺序

    1  实现并测试 reactive_wait                                   已完成,21 项测试全过
    2  写死窗口 / margin 网格 / 合同 / 哈希                        本文件
    3  跑 2021 三窗 × 五臂,冻结 margin 与最强盲                   待跑
    4  有效性全绿后,首次运行 2020 三窗 curve vs frozen blind      待 §6 裁定
    5  机械执行 5% 判决
    6  通过才做精确功率模型并考虑 2022;失败则封存当前 planner 方向

---

## 待 Codex 裁定的两处(不挡第 3 步)

**(A) 2020 的 `green_episode_offset_range`**

你给的三个 offset(2574 / 22301 / 13117)对应 range = **24669**。
按 2021 的同等预留量推算得 range = **24615**(2021: 52559 − 44950 = 7609 行预留;
32224 − 7609 = 24615),对应 offset 为 2628 / 22409 / 13117。
两者差 54 行。两个候选的三窗都能装下且互不重叠,机械上都可行。
**请指定用哪一个**,我不自选。

**(B) 2022 无数据**

`Turbine_*_2022.csv` 共 134 个文件,**每个只有 2 行,功率全 0**。
2022 目前不是"暂不读取"的问题,而是**根本没有数据**。
若确认集之后要用 2022,需要先补数据;否则第 6 步里 2022 那一项应从计划中撤下。

---

# Addendum A —— 2026-08-30,Codex 拍板,写于任何 2020 运行之前

**在提交本 addendum 时,2020 的任何运行结果均未被读取。** 该年数据从未被本轮任何进程加载过。
因此这不构成结果后调参。

## A.1 2020 确认集冻结

    green_episode_offset_range = 24669

    low    k=27   offset  2574
    mid    k=71   offset 22301
    high   k=13   offset 13117

依据(Codex 原话):`24669 = 32224 − 13 warmup − 7200 episode − 54 timezone − 288 forecast reserve`,
按真实语义算出的精确安全上界;24615 只是机械继承 2021 为保持旧 offset 而多留的 54 行,无独立物理依据。

**一处更正,不改变裁定。** 本考场配置的时区偏移是 `[0, 18, 54, 72, 108]`,**最大为 108**(DC4),不是 54。
用 108 重算上界即得 24615 —— 也就是说那 54 行的差额并非 2021 的历史余量,而正是 DC3/DC4 的时区偏移。

裁定仍然成立,因为**三个选定窗口离上界很远**,按 tz=108 与 288 预留逐窗核验:

    k=27   末行 10183 ≤ 32224   ok
    k=71   末行 29910 ≤ 32224   ok
    k=13   末行 20726 ≤ 32224   ok

上界之争只在 offset 接近 range 上沿时才会咬到,三窗都不在那里。故按裁定执行 24669 与上述三个 offset。

## A.2 2022 彻底撤下

`Turbine_*_2022.csv` 共 134 个文件,每个仅 2 行、功率不变、多数为 0。本轮**不以任何形式使用 2022**:
不作 sanity check、不补零、不循环扩展、**文档中不得称其为 held-out**。
将来若取得真实 2022 数据,另立独立预注册的新实验。

新增 `g1/preflight_wind_data.py`,机械拒绝行数不足、功率不变或均值为零的涡轮年。实测:

    2021  T12/36/91/95/96  全部 ok   rows=52559
    2020  T12/36/91/95/96  全部 ok   rows=32224
    2022  T12/36/91/95/96  全部 REJECT(仅 2 行;功率不变;多数均值为零)

## A.3 本轮数据角色(冻结)

    2021 三窗   公开校准集,用于选 margin 与冻结最强盲
    2020 三窗   跨年确认集,用于唯一一次 5% 判决
    2022        不使用

结论适用范围明确限定为现有 2020/2021 SDWPF 数据。

## A.4 有效容量的三项锁

漂移哨兵上线即抓到容量口径错误。空载时仿真器报的可用 PE 为 `[480, 384, 296, 240, 144]`,
而配置算术给出 `[600, 480, 296, 240, 184]`,更早的 `host×64` 给出 `[640, 512, 320, 256, 192]`。
真实容量是 `min(VM PE 配置数, 主机 PE 装机数)`:rs500a 为 48 PE/host,故 DC0=10×48=480、
DC1=8×48=384、DC4=3×48=144;DC2/DC3 的 VM 总数低于主机容量,恰好对上,掩盖了问题。

改为向仿真器读取,并按裁定加三项锁(均有单元测试):

    只在【零运行作业】的初始化时点读取一次 —— t≠0 或任一 DC 利用率 >0 即抛异常
    每 episode reset 后重新读取并重新冻结 —— reset 清标志并回落到配置值
    冻结值必须等于 [480,384,296,240,144],否则立即停止

`planner_cap_config` 与 `planner_cap_observed` 同时进结果表。

## A.5 校准执行顺序(冻结)

    1  margin 网格 {2,4,8,16,32,64} 依次跑 2021 三窗 × 五臂
    2  取使【所有格】同时满足 forced=0、drift=0、stale=0 与完成合同的【最小】margin
    3  在有效盲臂中按三窗池化 terminal carbon/MI 冻结最强盲
    4  提交冻结 artifact 与 jar/config/data 哈希
    5  才首次运行 2020 三窗
    6  机械执行「池化降幅 ≥5%、至少 2/3 同向、全部合同有效」的终判

---

# Addendum B —— 2026-08-30 深夜,测量通道重建

Addendum A 之后,第一格 curve/low/m2 四项合同失败。追查发现失败的不是规划器,而是哨兵所用的观测量。
本 addendum 记录通道重建、口径修正与新哈希。**所有内容写于 2020 数据被任何进程读取之前。**

## B.1 `dc_available_pes` 不可用作执行占用

`DatacenterInstance.getTotalAvailablePes()`(:241)是 `Σ vm.getFreePesNumber()`。
实测:单作业路由后 PE 被占,**作业完成后永不归还**(第 6 步占用,第 200 步已完成,第 1400 步仍显示占用),
而 `dc_utilizations` 全程为 0。它是 VM 分配计数器,不是执行占用。

据此撤回三条曾发出的结论:滞后固定 8 步、单作业 runtime 拉伸 34×/198×、runtime 模型错一个数量级。
三者均建立在把该字段当执行占用之上。

## B.2 evaluator-only 执行事件通道

Java 侧新增,走 `info`(String map,CSV 编码,沿用 `per_slot_reward_csv` 先例),不进 observation_space:

    exec_started_csv    id : dc : pes : startTime
    exec_finished_csv   id : dc : finishTime : elapsed : length : pes
    exec_running_csv    id : dc : pes
    exec_queued_csv     id : dc
    dc_free_vm_pes_csv  逐 DC 的 Σ getFreePesNumber()(保留作对照,不作判据)
    dc_running_pes_csv  逐 DC 的执行中 PE ← drain 预算与容量判据用此
    cloudlet_cpu_utilization_effective   JVM 实际生效值,供一致性哨兵

finish 事件由 running 集合差分推导,不用 `getCloudletsFinishedLastStep()` ——
实测该列表对 8 个确实运行并在第 55–200 步完成的作业全程为空。

## B.3 runtime 标定

对仿真器自身的 finish 事件,12/12 作业:

    elapsed / (length / mips)          均值 2.0166   sd 0.0156   与 PES 无关
    elapsed / (length / (pes × mips))  均值 3.5310   sd 0.8792   随 PES 在 2/4 间跳

强制 `cloudlet_cpu_utilization = 0.25` 后,第一列变为 4.0201(sd 0.0114)。拉伸精确等于 `1/u`。

**根因**:Java 以 0.5 为默认读取该键(`SimulationSettings:438`),而 G1 实验块从未设置它。
`load_config` 对该键返回 `None`。此前所有运行都在 u=0.5 下执行,作业实际耗时是标称值的两倍。

统一口径:

    runtime       = ceil(length / (mips × u))     与 PES 无关
    dynamic_power = pes × dyn_per_pe × u
    static        按更长区间积分

功率必须同步:只延长 runtime 而按满载积分动态功率,会把动态能量算成两倍,并错判绿电是否覆盖该作业。

## B.4 显式冻结 u=0.5 与 A/B 证明

G1 实验块显式写入 `cloudlet_cpu_utilization: 0.5`,替代隐式 Java 默认。语义不变,已用 A/B 证明:
同窗同种子、`green_queue_balanced`(不涉及规划器,隔离仿真器自身物理)、1500 步,显式配置对删除该键的配置——

    total_carbon_kg      0.030797266378540793   逐位相同
    green_used_wh        120.89839177435131     逐位相同
    brown_used_wh        103.6290856562037      逐位相同
    total_energy_wh      224.527477430555       逐位相同
    green_waste_wh       315.34759246527784     逐位相同
    completion_rate_mi / ontime_mi_share / finished / created / forced /
    mean_completion_time / episode_length       全部逐位相同

60 列中 50 列逐位相同;差异的 10 列全部是墙钟与内存仪表(`*_decision_us_*`、`episode_wall_s`、
`peak_cpu_rss_mb`),两次运行必然不同,非仿真输出。

## B.5 Java latest-start backstop 同步

原式 `length / mipsPerPe` 缺 u,低估约一倍、开火过晚。已加 `cpuUtilization` 参数并保留旧重载(默认 1.0)
以免影响其他调用点。四条 Java 测试覆盖:半利用率使假定运行时翻倍、不随 PES 缩短、旧重载不变、
非法 u 回落满速而非除零。

## B.6 合同改为逐 ID 闭合

废止 `planner_occ − (cap − dc_available_pes)`。新判据(全部进结果表):

    planner_n_unplanned_start          == 0    启动了规划器从未派工的作业
    planner_n_wrong_dc                 == 0    启动站点 ≠ 承诺站点
    planner_n_dispatched_never_started == 0    派工后从未启动
    planner_n_running_unknown          == 0    执行中但不在规划器账内
    planner_running_pes_over_cap       == 0    逐 DC 执行 PE 越容量
    deadline_forced_count              == 0
    planner_n_stale_dropped            == 0
    terminal completion_mi / ontime_mi >= 99.5%

drain 每步每 DC 预算改为 `cap − dc_running_pes − 本步 shadow`。

## B.7 边界 drain 语义(Addendum A 之后的修正)

删除边界批量拉起(原实现一次拉起 4542 个预约,摧毁已可行的计划)。现行:
预约保留原站点与原启动时间,到点才执行;边界后不再规划、不再重优化;
未预约积压走共用可行性 drain,按 latest-start 排序,装不下则等下一步;无固定作业数限速。

## B.8 合成微考场验证

`g1/micro_validate.py`,14 项全过:

    runtime ceil 五个边界值
    零绿电   → 选棕电因子最低的站点,按实测时长占用
    满绿电   → 风大的 DC4(棕 0.92)胜过无风的 DC0(棕 0.08)
    单脉冲   → 作业被排进唯一有风的窗口(40–60,实得 start=46)
    容量越界被记录、无主启动被计数
    动态功率随 u 缩放,且能量在更长窗口上守恒

## B.9 新哈希(替代 §8)

    config_C.yml   267e9402e4bd1fd2be0b69b8d5d81c4a0ddcd296de71a912daddce8495c2d762
    gateway jar    3b0fc140d5472eafc013b52bea3660b51396113770469823f6fc966cd75a36b4

涡轮数据哈希与行数不变(见 §8),2021/2020 数据未被触碰。

## B.10 V3.2 观测缺陷(记录,本轮不修)

`hierarchical_multidc_env.py:1782` 的 `cf_mode` 分支用 `MI/(PES×MIPS)`,同函数 :1787 用 `MI/MIPS`,
两者写进同一个 `slack_sec` 观测特征。实测证明除 PES 口径错误(比值随 PES 在 2/4 间跳,非常数)。
按裁定:旧 checkpoint 按 legacy 语义复现,不再用它支持"物理 runtime 正确"的论断,
未来修复另开 v3.3 并重训。本轮 C-regime 未启用,不阻塞。

## B.11 受 u=0.5 影响的组件表

    历史 G1/TB12/T1/T2 的碳与完成率     数字有效,是 u=0.5 物理下的真实结果
    作业时长的绝对叙述                  需重算,实际为标称值的 2 倍
    「作业相对风窗太短」的机制解释        必须重算
    latest-start / backstop 行为        不对称影响,原式低估一倍
    forecast-fit / 承诺窗口类论断        需复核
    论文若声称使用 1.0                   配置—描述不一致,需改正文或降级证据

暂不重跑历史实验。

---

# Addendum D —— 2026-08-31,风电年份可选(held-out 运行期间补写的文档)

## D.0 时间口径,如实记录

    03:07:12   源码 diff 落盘(GreenEnergyProvider.java, SimulationSettings.java)
    03:09:42   jar 构建完成
    03:20:33   2021 冻结格回归证据落盘
    09:58      2020 六格启动
    10:2x      本 addendum 撰写

**实现修复先于 held-out 运行,正式文档在启动后补齐。** 实现证据(源码、jar、回归结果)在
09:58 之前全部落盘,但**当时尚未提交 git**,且本文档在六格启动后才写。

期间未查看任何碳结果。启动后仅读取过:逐格步数、gateway 日志中的 CSV 解析行、
以及一次独立的绿电序列相关性哨兵(单独的 env,非六格之一)。

**本轮结果不得因结果不利而重跑。** 六格一次跑到底,无论结论如何。

## D.1 为何需要这次改动

`GreenEnergyProvider.findTurbineFile` 的模式列表把 `Turbine_<id>_2021.csv` 写在首位,
而该文件恒存在,故任何配置都无法解析到其他年份。`csv_year` 只属于 timecap 预报器,
Java 侧不存在该键。**预注册的 2020 跨年确认集在旧实现下无法运行。**
这是执行能力缺口,不是判据变更。

## D.2 精确 diff

新增 `SimulationSettings.windCsvYear`(键 `wind_csv_year`,**默认 2021**),
`GreenEnergyProvider` 增静态 `preferredYear`(默认 2021)与 setter/getter,
模式列表首位改为首选年份、原有顺序整体后移作为回退,fallback 串同步。
`MultiDatacenterSimulationCore` 构造时在任何 provider 构造前设置该偏好,
并在 info 中回报 `wind_csv_year_effective`。

    SimulationSettings.java     +5
    GreenEnergyProvider.java    +23 −2

默认 2021 时,模式列表首位与原列表首位相同,解析结果逐位不变。

## D.3 兼容性证据

**回归格**:用新 jar 重跑已冻结的 2021 格 `nowait_planner_low`(k=19, offset 19171),
与旧 jar 结果比对 **82 个物理列全部逐位相同**,含
`total_carbon_kg = 0.1259849089118572`。仅墙钟与内存仪表不同。

**Java 测试** 117 项,0 失败,含新增 `WindYearSelectionTest`(默认为 2021、偏好被采纳)。

## D.4 哈希

    旧 jar   e1aeba94e154eb01152482da097df5cbcb1f4018b4e32497596aae96684067d5
    新 jar   6d23d8790d3a4d997eb5867c180c0030c5ced264b794d18e32b39ed10de261b5
    构建自   commit 2a86e3f + 上述两文件未提交改动(见 D.0)
    config   2021 校准 ca4fcf76ed4b97b8d972613e604dadd5453fef8cb90f0a90b285fe6f17a84d73
    config   2020 确认 g1/config_C_2020.yml(由前者派生,仅改 wind_csv_year=2020
             与 green_episode_offset_range=24669)

    2020 涡轮文件 SHA256
      T12  162e80e50b31a720e210008729cfbfbbff6bbdc4cc399814195f4e5613291daf
      T36  e92090af7abd76ecaba1b6f62045b4d2b788d243f49781bac74f944a525b7c1a
      T91  c45df4f3d2d0ba66713941e3ab94208f1a96b8a4db11f2d7bb039e958fccb61e
      T95  ee5e68df0004ab1640d146b255257c87bb5635bf02cb33cf948abf6c2625f79c
      T96  7ac6ce1a70321b172427edc4fe143ec188b062c34712dec3a47da86fa9d69112

## D.5 六格启动时的接线

    jar        6d23d879…(runner 硬校验)
    year       wind_csv_year = 2020
    offsets    low k=27 off 2574 / mid k=71 off 22301 / high k=13 off 13117(Addendum A)
    arms       curve_planner 对冻结最强盲 nowait_planner
    contract   Addendum B.6 逐 ID 闭合,不变

**相关性哨兵**:以 2020 配置取仿真器 600 步绿电序列,与两年轨迹分别求最大相关 ——
2020 得 **1.0000**,2021 仅 0.7813。接线确认正确。

该哨兵曾出现一次单点比对不符,查明为探针自身的 off-by-one(`--reset-skip k` 的被测集是
索引 k,而探针循环 k 次后测的是索引 k−1,差一个 1009 步长)。runner 无误。

## D.6 结论范围(预先声明)

即使 2020 三格全部通过 5% 判据,**只能**证明「完美未来信息在 C-regime 上可兑现」。
下一步仍须换成真实 TimeCAP 预测重验,**不得**据此声称预测器或 EU-CRD 已经有效。
