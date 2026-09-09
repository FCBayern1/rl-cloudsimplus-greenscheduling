# 规划器重建完成报告 —— 请裁定余下三处

裁定全部照办。三层账本 + 稳定 ID 已落地,`curve_oracle` 这个名字已从代码里删除。
六步顺序里 **1–4 步做完并验证,5–6 步已具备条件但尚未跑**。

---

## 一、Java 侧:evaluator-only 稳定 ID(已批准的新 jar)

- `GlobalObservationState` 新增 `batchCloudletIds`(`long[]`,空槽 `-1`,防御性拷贝,长度参与既有校验)
- `MultiDatacenterSimulationCore` 用 `cloudlet.getId()` 填充
- 新增 `BatchCloudletIdChannelTest`:空槽必须报 `-1` 而非 `0`(`0` 是合法 id,padding 绝不可被规划)、
  防御性拷贝双向、长度不匹配抛异常
- **Java 全套 108 测试 0 失败**;新 jar 在 `cloudsimplus-gateway/build/install/`,
  `/home/joshua/frozen/g1_gateway/lib` 只读未动

ID **不进 `observation_space`**,而是走 `info["planner"]`,因此 checkpoint schema 不变。
旧 jar 上该块返回 `None`,不静默降级。

## 二、Python 侧通道 + 第五个 P0(你没提到、我实测到的)

`info["planner"]` 出规划器需要的**原始量**:
`ids / mi / pes / time_to_deadline(秒)/ deadline_present / is_deferred / wait_age / current_clock`。
走 info 而非观测,是因为 v3.1 特征是归一化的且被 `obs_v31_features` 开关挡着,规划器要的是秒。

**新发现的 P0-5:规划器从来没看见过 deadline。**
`evaluate.py::_convert_global_obs_for_scheduler` 只转发 `batch_cloudlet_pes` 与 `batch_cloudlet_mi`,
不转发 `batch_cloudlet_time_to_deadline`。实测:

    keys passed through: ['batch_cloudlet_mi', 'batch_cloudlet_pes']
    ttd present? False
    ttd the oracle actually sees -> [1e+09 1e+09 1e+09 1e+09]   unique: {1e+09}

于是 `latest = t + max(0, int(1e9) - r)` 恒为无穷,`hi = min(latest, t+400)` 永远等于 `t+400`。
**deadline 约束在旧规划里从未生效**,它解的根本不是那个受约束的问题,`deadline_forced_count=49` 由此而来。

新实现在缺 planner 块时**抛异常**,错误信息点名这个哨兵,不再填默认值。

**端到端实测**(新 jar,真实 gateway):

    planner keys: [ids, mi, pes, time_to_deadline, deadline_present, is_deferred, wait_age, clock]
    [step 0] real_slots=5/128  ids=[0,1,2,3,4,-1]  ttd=[6499, 6500, 6500, 6501, 6501]
    [step 5] real_slots=17/128 ids=[0,1,2,3,4,5]   ttd=[6494, 6495, ...]

id 跨步稳定,ttd 真实倒数。顺带量化了 P0-1:**开局 128 槽里只有 5 个是真作业**,
旧代码把另外 123 个 padding 全写进了未来容量账。

## 三、规划器重建

- **三层账本**:`scratch`(每步重建)/ `reservations`(按 job id 跨步持久)/ `active`(已派工),
  `occ` 为唯一容量真值,读写只走 `_hold` / `_release`,两个字典与网格不会漂移
- **已持有预约的作业不重规划**,计划由构造保证稳定
- **容量改真实 VM PE** `[600, 480, 296, 240, 184]`(旧 `host×64` = `[640, 512, 320, 256, 192]`,高估 4–8%)
- **补 `reset()`** —— 基类是 `pass`,原实现未覆盖,跨 episode 泄漏时钟与占用网格
- **视界默认改为作业自身的可行等待窗**(`ORACLE_HORIZON=0`),不再是固定 400 步
- **改名** `CurveInformedPlannerGlobalScheduler` / 注册键 `curve_planner`;**oracle 别名已删除**

## 四、盲候选族(共用规划器 / 容量账 / 空间碳模型,只换未来信息源)

    curve_planner         真实未来曲线
    persistence_planner   未来 = 当前电表读数
    climatology_planner   窗口【之前】的历史均值,不读被测窗口
    nowait_planner        同一空间成本模型,不许等待

唯一的差异点收敛到一个方法 `_green_view(d)` 与一个 `allow_defer` 开关。
有测试断言四臂的 `cap / cb / cg / mips / dyn_per_pe` 逐位相同。

**你列表里的 `reactive_wait` 我没实现** —— 见下面第(三)问。

## 五、测试

`tests/test_curve_planner_ledger.py` **17 项全过**,含你补的两条:

- 作业离开当前 128 槽若干步再回来,预约仍在且未漂移,`n_plan` 不增(不重复承诺)
- `reset()` 清时钟与三本账

**变异校验**(测试写在修复之后,故做此校验):恢复「padding 当真作业」→ 挂 4 项;
恢复「每步清空预约」→ 挂持久性那项。

`tests/test_energy_carbon_identity.py` 3 项全过。

全量 Python 套件 1218 passed / 18 failed;逐条比对干净树,**18 项在改动前后完全同集**,
两个方向不同的条目已隔离:一个是全套顺序依赖(单跑两边都过),
另一个是 `gateway jar not stale` 的 mtime 假阳(stash/pop 顶高源文件 mtime),重建 jar 后转绿。**无回归**。

## 六、逐 DC 能量—碳恒等审计(第 3 步)——**账本闭合**

`g1/energy_carbon_identity.py`,对 smoke 的 blind 逐步 trace:

    site  brown_f   green_Wh   brown_Wh   spill_Wh  carbon_kg   share   recv
       0     0.08     403.38     251.65     477.87   0.024166   18.6%   4930
       1     0.35       0.00       0.29     784.44   0.000103    0.1%      2
       2     0.55     178.07     185.91     101.40   0.104029   80.3%   3066
       3     0.75       0.00       1.20       0.00   0.000902    0.7%      1
       4     0.92       0.00       0.41       0.00   0.000380    0.3%      1

    total carbon kg   reconstructed=0.129579  reported=0.129579  rel=0.000%  MATCH
    green used Wh     reconstructed=581.4487  reported=581.4487  rel=0.000%  MATCH
    brown used Wh     reconstructed=439.4662  reported=439.4662  rel=0.000%  MATCH
    total energy Wh   reconstructed=1020.9149 reported=1020.9149 rel=0.000%  MATCH
    green waste Wh    reconstructed=1363.7123 reported=1363.7874 rel=0.006%  close

**恒等式逐位闭合**(waste 差 0.006%,边界步舍入)。账本没坏,碳差可以做归因。

**且它直接指向一个结论**:blind 臂的碳有 **80.3% 来自 DC2**(棕 0.55,收 3066 作业),
DC0(棕 0.08,收 4930 作业)只占 18.6%。这与你的判断一致 ——
作废的那个 −47.92% 很可能主要是「把活从 DC2 挪到 DC0」的**纯空间碳因子效应,不需要任何未来信息**。
`nowait_planner` 正是用来把这部分吃掉的对照臂。

## 七、双时点合同(第 5 步,已实现未跑)

按你的裁定实现,`PLANNER_DECISION_HORIZON` 控制,默认关闭:

- 边界之前可等
- 边界之后**对所有臂同时关闭等待**;已开预约按**各臂已承诺的站点**立即拉起(不改站点,只去掉等待),
  新到作业 `latest = now`
- 有三项测试:边界后不再开新等待 / 已开预约在边界被拉起且站点不变 / 默认关闭

`max_episode_length` 用 evaluate 现成的 `--override` 抬,不改配置文件。

---

# 请裁定的三件事

## (一)功率模型:走「近似 + 降级命名」还是先对齐?

我按你给的两档里选了**近似档**,已照你的措辞改名为 curve-informed feasible planner,
四臂共用同一功率模型。当前近似仍是:`static = 332W × host占比`、`dyn_per_pe = (214−51.4)/64` 线性、
候选起点网格 `r/16`。**这一档是否即为本轮定稿**,还是要先做逐 host 型号 + idle power-down 的对齐?
本考场 DC0/1/4 是 rs500a、DC2/3 是 rs700a,型号差异真实存在。

## (二)最强盲怎么冻结:校准窗口取哪个?

你要求「先在校准窗口按完成合同合格后 terminal carbon 最低冻结最强盲,再去 held-out 比较」。
请指定**校准窗口的 k**(作废那轮用的是低绿窗 `k=19` / `ORACLE_OFFSET_ROWS=19171`),
以及 held-out 用哪几个 k。我不自选,以免用判决窗口选盲。

## (三)`reactive_wait` 要不要补?

你列了五个信息源,我实现了四个。`reactive_wait`(按当前绿电相对作业需求的等待门)
在共用规划器的框架里其实是 `persistence` 的一个特例(未来=现在 ⟹ 只在当前够绿时才起跑),
所以我怀疑它与 `persistence_planner` 会给出相同或极近的策略。
**要我单独实现它作为独立臂,还是认定其被 `persistence` 覆盖?**

## 附:两个次要参数,若无异议我按默认走

- `deadline_forced_count == 0` 这道有效性门,我用**从最晚可行起点回退 2 步**的余量去抢在 backstop 前面
  (`ORACLE_BACKSTOP_MARGIN`,可调)。若首跑仍有 forced,我会调大余量而非改门。
- 双时点的 terminal 上限我打算取 **12000**(你给的是 10000–12000)。
