# V3.1 工单(2026-08-13 下发;已按第四轮复审修订)

背景必读:`docs/V3_FORECAST_DIAGNOSIS.md`(§0 目标、§6d 执行序列、§6e 战略)。
全部不占仿真机器。

**归属已定(08-13)**:本文件是规格的唯一权威来源;归属/顺序/交付口径见两份任务书——
Claude:工单 1→4→5(`docs/V31_TODO_CLAUDE.md`);
Codex:工单 3→2(`docs/V31_TODO_CODEX.md`)。
文件所有权分锁与合并顺序(Claude 先合,Codex rebase)以任务书为准。

## ⚠️ 验证状态(领工前必读,别把修复当成已确立的事实)

**被验证过的是缺陷,不是修复。**各修复针对的缺陷全部代码级证实(路由补贴、逐次收费、
槽位无 slack、两臂 local 各异、两把尺子),但修复的**有效性几乎全部未验证**:
- 统一定价是唯一部分验证的,结果是个警告:oracle_sp 修掉了反号,但逐局配对 30/30
  输盲臂——修复真缺陷 ≠ 接近目标;
- 势函数化只有理论保证(不变坏),没有本系统实证;
- z 标准化的前提(方差假设)本身待判(等 sp_s2 配对),且 SustainCluster 默认 `normalize:false`;
- 新观测有反向风险(高噪声下的新干扰通道);
- local 差异是否**载重**正是 drainfix 在测的。

**为什么仍然现在实现**:①修复对两臂对称,V3.1 臂间消融仍只差 forecast_mode;
②全部门控默认关+回归锁;③100k P1 冒烟验证的是**打包配方**能否学出预报响应,
不验证单个修复。
**⚠️ 但"修复错了最坏是干净的 null"不成立**(第四轮复审纠正):若配方导致 all-defer、
backstop 主导或碳信号被归一化抹掉,得到的是**失效考场**,不是有解释力的 null——
这就是工单 1 必须过奖励真值表测试(见下)而不只是公式符号测试的原因。
**开关决定**:drain 固定是**无条件的**(结构清理,不取决于 drainfix 结果——drainfix 只
复核旧判决);z-score 变体**不由 sp_s2 决定**(sp_s2 只判跨种子复现性,定位不了方差 vs
干扰项);其余开关明早看完四份判决再锁。打包配方若成功无法归因单个修复——可接受
(目标是存在性证明),要归因事后做减法消融。

## ⛔ 所有工单的共同禁令

- **机器队列在跑**(sp s2 训练 → P3 评测 → track0 → drainfix,全自动到明早)。
  **禁止**启动任何 `baselines.evaluate`、任何训练、任何 Java 网关;**禁止任何 pkill**
  (训练的 JVM 同样匹配 `MainMultiDC`,误杀会冻死训练)。
- **禁止编辑正在运行的 bash 脚本**(`local_eval_rt/run_v3_*.sh` 四个都算)。
- JUnit(gradle test)和 Python 单测可以跑,都是 CPU 秒级。
- 每项新代码必须配测试(CLAUDE.md 硬性要求);确实无法测的要在交付说明里写明原因。
- 所有行为改动必须**配置门控,默认关**——旧实验的行为要字节不变。
  这不是风格问题:P3/主表的预注册口径正在依赖旧行为。
- **文件所有权**(工单 1、2 都改 `MultiDatacenterSimulationCore.java`,工单 1 还改
  `SimulationSettings.java`——逻辑独立但**不是文件级独立**):并行 agent 必须各用
  worktree/分支,或者按此分锁:工单 1 独占 `accumulatePerActionReward` +
  `computeDcCostFeatures` + defer 分支(L590-770 段)与 `SimulationSettings.java`;
  工单 2 独占 `GlobalBroker` / `GlobalObservationState` + Core 里的观测组装段;
  两单都要碰的 import/字段声明区,后合并者负责解决冲突。

**执行优先级(第四轮复审定)**:工单 3(含评测一致性)→ 工单 2(含槽位
deferred flag/count)→ 工单 1 的 no_offset(可直接写)→ incremental_urgency
(按修订规格)→ 碳归一(先做采样仪表 + scale_only,centered_zscore 待议)→
工单 4 模板/preflight(开关全关)→ 工单 5。

---

## 工单 1:奖励手术(Java)★关键路径

**动机**:诊断文档 §6c——路由拿 +2 固定奖金而 defer 拿不到(同一 softmax 被推向立即
路由);defer 逐次重复收费;现货碳价方差巨大(泄漏定价赢 36/36 局的真实机制是方差缩减)。

**改动**(全部新配置键门控,默认保持旧行为;⚠️ 规格经第四轮复审修订,不要按初版实现):

1. `per_action_completion_mode: bonus(默认)|no_offset`
   - `no_offset` = `−w_p·(1−probComplete)`。**正名**:代数上 `−w(1−p) = wp − w`,
     它不改变各 DC 之间的 completion 排序,唯一作用是给所有 route 动作统一去掉 +w 的
     常数——这是**移除 route offset**,不是新的 completion 学习信号,命名与注释按此写。
   - 位置:`MultiDatacenterSimulationCore.accumulatePerActionReward()`(~L715-768)。
2. `defer_cost_mode: flat(默认)|incremental_urgency`
   - **不再声称 Ng 势函数策略不变性**(标准形式是 γΦ(S′)−Φ(S),我们 γ=0.999 且是
     替换旧 cost 而非在无塑形 MDP 上加塑形,理论前提不满足)。改为诚实命名
     `incremental_urgency_cost`。
   - 紧迫度函数 **`U(s) = clip(1 − s/W, 0, 1)²`**(⚠️ 初版的 `min(1,(1−s/W)²)` 有 bug:
     s>W 时回升,s=2W 时=1,与"远离 deadline 应为 0"正相反)。
   - **重逢结算语义**(必须实现,否则任务 route 走时会逃掉最后一段等待成本):
     每个 cloudlet 保存 `lastChargedUrgency`;任务**每次重新出现在 batch 里**——
     无论这次被 route 还是再次 defer——先结清 `−w_s·[U(now) − lastCharged]`,再更新。
     这样总收费 telescope 成 U(末)−U(初),与重逢次数、重逢间隔无关。
   - base cost 强制 0。位置:同文件 L605-626 defer 分支 + route 分支入口。
3. 碳项归一:`per_action_carbon_norm: fixed(默认)|scale_only|centered_zscore`
   - **`scale_only` = `marginalKg/σ`,保留物理零点,是安全档**;
   - `centered_zscore` = `clip((marginalKg−μ)/σ,−5,5)`。⚠️ 代数上
     `−w(m−μ)/σ = −w·m/σ + wμ/σ`,那个 **+wμ/σ 只落在 route 上,是重新引入的
     route/defer offset**——它不是归一化副作用,是一个**设计决定**:低于均值碳价的
     路由拿正奖励,等于定义了"碳阈值策略"。若开启必须在配置注释里明说这一点。
   - **概念修正**:z-score 改的是尺度不是信噪比(真正降方差的是平均/基线/控制变量);
     它解决的是"碳项相对其他奖励项被压扁"的尺度问题。μ/σ 来自配置常量
     (工单 4 标定脚本离线产出,全臂共用),不做在线统计。
     三档都记录 raw / normalized / clip rate。
4. `SimulationSettings.java` 加对应参数解析(照 `windowCarbonSource` 的现成模式)。

**测试**(照 `WindowCarbonRewardMathTest.java` 的模式新建):
- **奖励真值表(核心,替代纯公式测试;route/defer 的耦合正是 all-defer 复发的风口)**:
  在推荐开关组合下逐行断言——
  ① 高绿电+容量充足:r(route_green) > r(defer);
  ② 当前棕电+绿电将至+slack 充足:r(defer) > r(route_brown);
  ③ deadline 紧迫(s→0):r(route) > r(defer);
  ④ 拥塞 DC:r(route_其他DC) 或 r(defer) > r(route_拥塞DC)。
  真值表不过 → 该开关组合禁止进入训练。
- no_offset:p=1 → 0,p=0.5 → −w_p/2;
- incremental_urgency:k 次重逢序列总收费 = U(末)−U(初) 与 k 无关;U(2W)=0;
  最后一次是 route 时该段照收(防逃单);
- scale_only 零点不变;centered_zscore 均值→0、±6σ 截 ±5;
- **回归**:全部新键取默认值时输出与改动前逐位一致。

**验收**:JUnit 全绿(含真值表);`git diff` 无非门控行为改动。

---

## 工单 2:观测补齐 + 仪表(Java + Python)

**动机**:§6c——批槽位没有 slack/年龄,策略连奖励最优的 defer 规则都无法**表示**;
deferred 与新到达同队不可分(状态混叠);local 节流比例现在算不出来。

**改动**(统一由 `obs_v31_features: false(默认)|true` 门控):

1. Java `GlobalBroker`:deferred 标记 + 计数,**生命周期锁死**(第四轮复审规格):
   - 只有**显式 defer 动作**打标记;routing-failure 的 requeue **不**打(实现者需枚举
     全部 requeue 路径,包括 L603 "forced routing failed → fall through" 那条,逐条归类);
   - 重复 defer:全局 deferred_count **不重复加**,该 cloudlet 的 defer_count **加**;
   - 被 route(含 forced route)后从 deferred 聚合中删除;reset 清空;
   - 新增 `getGlobalDeferredCount()` / `getGlobalDeferredMi()`。
2. Java `GlobalObservationState`:批槽位数组**四个**(不是两个):
   `batch_cloudlet_wait_age`、`batch_cloudlet_time_to_deadline`、
   **`batch_cloudlet_is_deferred`、`batch_cloudlet_defer_count`**——只有全局聚合时,
   槽位 actor 仍不知道"当前槽里这个任务是不是我以前 defer 的"。
   不预合成 slack(deadline−now−runtime 在异构 MIPS 下随候选 DC 变;分开给
   time_to_deadline / est_runtime(=已有 mi 可推)/ wait_age,让模型自己组合)。
3. Python `hierarchical_multidc_env.py`(obs space ~L544 起)。**边界规格**:
   运行中 time_to_deadline 会变**负**(过期未派),wait_age 可达局长 ~7200s——
   **不能用 [0,3000]**。输出归一化+显式截断的值(如 time_to_deadline/W 截 [−1, 4]),
   并提供 deadline-present mask。初始 slack 的实测参考:p99≈2983s、max≈2999s。
   (obs 压扁旧坑:green_power high=5e6、cloudlet_mi bound——bound 错一次废一轮训练。)
4. 仪表(不进观测,进 monitor CSV):`local_dispatch_requested` /
   `local_dispatch_placed` / `local_waiting_after_dispatch`,逐 DC 逐步累计。

**测试**:
- Java 单测:defer 两次 → 全局 count 计 1、槽位 defer_count=2;route 后聚合删除;
  routing-failure requeue 不打标;reset 清空;MI 记账正确;
- Python:obs space 含新键、shape/bound 正确、负 slack 与超长 wait_age 不越界、
  `obs_v31_features:false` 时与现状完全一致;
- 老 checkpoint 加载不受影响(默认关时 obs 字节不变)。

**验收**:两套测试绿;默认关时 `probe_forecast_sensitivity.py` 对旧 ck 的输出不变。

---

## 工单 3:训练侧固定 local drain(Python)

**动机**:§2b 三层结论——local 与各臂 global 共同演化出不同 release 率,是已证实的
混杂源;评测侧 `--local drain` 已有(evaluate.py `local_override`,08-13 加),训练侧没有。

**改动**(`fixed_local_scheduler: none(默认)|drain` 门控):

1. `hierarchical_multidc_pettingzoo.py`:调底层 `step()` 前,把所有 local agent 的动作
   覆盖为**最大合法 dispatch**(按 action mask 取最大可行值,不是硬编码 64);
   **fail-fast**:`fixed_local_scheduler=drain` 而 `local_dispatch_mode != dispatch_rate`
   时启动即报错——vm_placement 动作空间下"最大动作"不是 drain,静默跑会产出垃圾臂;
2. `train_rlmodule_gtrxl.py`(~L581/706):该模式下 `policies_to_train` 只留
   `global_policy`;local module 保留以满足 RLlib 接口,输出被覆盖;
3. local 的样本仍会进 buffer——确认不训它就行,不必阻止采样(改采样管道风险大);
4. **训练-评测一致性**(第四轮复审补,必做):evaluate.py 直接建 base env,不走
   pettingzoo wrapper——V3.1 checkpoint 若用 `--local rllib` 评测,会启用那个**没训过**
   的 local module。规矩:V3.1 的一切评测用 `--local drain`(local_override 已接好);
   加保险——evaluator 读到 checkpoint config 里 `fixed_local_scheduler=drain` 时,
   对 `--local rllib` 直接**拒绝**并提示,而不是默默跑。

**测试**:wrapper 单测(mock 底层 env,断言 local 动作被覆盖为 mask 内最大值;
`policies_to_train == ['global_policy']`);fail-fast 单测(vm_placement + drain → 抛错);
evaluator 拒绝逻辑单测;默认 none 时行为不变的回归断言。

**验收**:测试绿;100 步 dry-run(可用 mock env,不起网关)不抛错。
**注**:drain 固定是无条件的设计决定(清理无关 co-learning、砍 8× local 样本、
两臂共享同一局部动力学);drainfix 的结果只复核旧 V3 判决,**不决定**这一项。

---

## 工单 4:配置块 + 预检查升级 + 标定脚本 + 预注册(我这边,或单独一个 agent)

1. `config_C.yml` 新增 `experiment_v3_1_oracle` / `experiment_v3_1_noforecast`
   **模板**:基于 v3 块,两臂只差 `forecast_mode`,统一 `window_carbon_source:
   persistence`;**候选开关今天全部保持关闭**(⚠️ 初版此处写"打开所有开关",与
   验证状态节矛盾,以本行为准)——最终认证配置明早看完四份判决 + 工单 1 真值表
   结果后生成;
2. `preflight_scenario.py` 加四项检查(这次咬我们的每一口都变成一道门):
   ①两臂 `window_carbon_source` 相同;②两臂 experiment 块 diff 白名单 = {forecast_mode};
   ③obs 新特征的 bound ≥ trace 实际 p99;④认证臂 `fixed_local_scheduler=drain`;
3. 标定脚本 `calibrate_reward_norm.py`:用固定随机策略(种子定死)跑 2000 步
   marginalKg 采样 → 输出 μ/σ artifact;**依赖工单 1 的键名**,脚本先写好,机器空了再跑。
   **artifact 必须记录**(第四轮复审规格):trace/scenario hash、seed、reference policy
   标识、样本数、逐 DC 分布、raw mean/std/分位数、clip rate、单位。
   采样口径:对**所有候选 (task, DC) 对**采样而非策略实选的,避免策略选择偏移分布;
   defer 不以 carbon=0 混入;
4. 预注册文档 `docs/V31_PREREG.md`:P1/P2/P3 判据、护栏(backstop/超时筛/defer 率监控)、
   条件 critic 三诊断(defer TD residual / 条件 EV / 分桶 calibration)、
   λ 动用条件,跑之前定稿。

---

## 工单 5(可选,低优先):反号习得时点扫描

对 v3_oracle_s1/s2 的 ck1…ck10 逐个跑 `probe_forecast_sensitivity.py`(纯 CPU,
分钟级/ck,不碰网关,可与训练共存),画 `P(defer|将至)−P(defer|将去)` 随训练的轨迹。
回答 §7.3:反号是早期就定型还是后期漂移。产出一张表即可。

---

## 依赖与时序

```
今天:工单1-4 并行实现+测试(全不占机器)
明早:四份判决落地(sp s2 配对 / P3 / track0 / drainfix)
  ├─ track0 <10% → V3.1 场景参数要先改(工单1-3 的代码照用,考场换参数)
  ├─ drainfix 翻转 → local 混杂是主因,V3.1 的 drain 固定更加关键
  └─ 正常路径 → 标定脚本跑 μ/σ → V3.1 双臂开训(15h/seed)→ 100k 冒烟判 P1
```

工单 1-3 的产出对**任何**后续考场都适用(是配方修复不是场景修复),
所以即使 track0 判考场不合格,这些工作也不白做。
