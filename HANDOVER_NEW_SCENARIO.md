# 交接:第十考场(新场景搜索)—— 给 GPU 机上接手的 agent

写于 2026-08-19。**先读完本文再动手。**本文自足:不依赖 `docs/`(仓库故意不跟踪)。

---

## 0. 一句话任务

前九个考场都没能证明「知道未来」有不可替代的调度价值。SQT2(第九个)已封档为负结果,
但它把**失败原因定位到了机制层**。你的任务是:**造第十个考场,并且必须先在离线阶段
证明它不会重蹈 SQT2 的覆辙,再烧任何仿真/训练算力。**

---

## 1. 背景:论文与战役状态

- 论文 = EU-CRD(ICLR 2027,摘要 9-18,正文 9-25),主表考场是 **C-regime**:
  `experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap`(Vanilla)/
  `..._timecap_eucrd_v4`(EU-CRD),5 DC、8000 作业、SDWPF 2021 风电、TimeCAP 预报器。
- 论文的主张是**信任管理**(腐蚀预报下的遏制),**不依赖**"预报有价值"这个命题。
  所以第十考场是加分项,不是论文的必需品。**不要为了它推迟论文证据工作。**
- 已冻结的两个负结果:
  - **V3.2(真实风)**:oracle/shuffle/anti 三臂碳排一致(−1.8%),预报**内容**贡献为零;
    解释是空间替代(5-10 个 DC 里总有一个此刻是绿的,不需要知道未来)。
  - **SQT2(构造方波)**:见第 2 节。

---

## 2. SQT2 到底测出了什么(这是你设计新考场的全部依据)

考场:同步方波绿电(全绿 DC 同时进深槽,堵死空间逃逸),槽长双峰
(80% 短槽 U[300,1500]s / 20% 长槽 U[2700,4500]s,时间权重各半),
作业 slack 双峰(60% 紧类 U[200,900]s)。四门阶梯共享同一冻结空间底座:

```
nowait(不等) → naive(有预算就等到底) → hazard(闭式后验,q* 冻结) → clairvoyant(知道真实剩余槽长)
```

### 2.1 核心分解(可直接写进论文)

```
nowait → naive        −10.44% 碳(10/10 同号)   ← 「会等」这个动作本身的价值
naive  → clairvoyant   ≈ +0.5% ~ −1%           ← 「知道该等多久」的边际价值
```

**杠杆价值 ≈ 9-12%,预报内容价值 ≈ 1%。**

### 2.2 四个致命机制(新考场必须逐条规避)

**① 无悔等待(主因,已被数学证明)**
`clair_forgone ≡ 0`:在全部 10 个锚点上,naive 的等待终点**恰好等于** clairvoyant 的
放弃边界,因此**每一个作业的绿/棕归宿逐个相同**。等错了不付出任何代价
(容量宽裕 2.9× + 棕碳因子时不变 + latest-start backstop 免费兜底
= 一个**免费的美式期权**)。信息再完美也无法改变任何一个决策。

**② 最强盲基线是 naive —— 零信息的那个**(⚠️ 这条最反直觉,也最重要)
```
vs nowait:  naive −10.44%  >  hazard@0.25 −8.28%  >  @0.40 −7.74%  >  @0.50 −6.45%  >  @0.60 −5.45%
clairvoyant vs hazard@0.50: −3.39%(10/10)   clairvoyant vs naive: +0.50%(仅 3/10)
```
hazard 用了槽龄 + 注册的槽长分布 + 闭式后验(离线分类正确率 77.3%,n=82472),
**却全线输给"无脑等到最后一刻"**,而且越会挑(q 越大)越差。

> **推论:削弱盲策略的先验(例如让槽长更难预测)对新考场毫无帮助**,
> 因为 naive 根本不用先验。**唯一有效的方向是让「等待」本身有代价。**

**③ 容量收紧无效(已实测,别再试)**
两个会等的臂**在同一时刻(槽末)释放同一批作业**,拥堵必然同时砸中两者。
实测 894 PE(基线 2520 的 35%)时:nowait 按期率 0.9747 纹丝不动,
naive 0.958-0.980,clairvoyant 0.963-0.981,hazard 0.964-0.983 —— clairvoyant
只是帕累托中间态,不占优;继续压到 700/504 PE 时控制臂自己先崩。
**拥堵惩罚的是「等」,不是「等错」。**

**④ 控制臂的按期率天花板**
基线容量、零延迟、零溢出、零强派下 nowait 的 `ontime_mi_share` 只有 **0.9756**
(逐锚恒定 → 不是噪声)。机制:峰值并发 157 个作业 vs DC0 只有 175 个 VM,
紧作业 slack p01 = 211s,CloudSim 时间片共享一挤就踩线。
**教训:任何绝对合同阈值,必须先测控制臂的天花板再定。**

---

## 3. 新考场的四个必要条件(缺一即重蹈覆辙)

1. **clairvoyant 的信息必须改变具体作业的决策**(不只是改变时点);
2. **盲策略不能靠槽龄/先验复现该决策**,**也不能靠「最大化等待」复现**(条件 ②);
3. **等错必须有作业级、非对称的代价**;
4. **该代价不能同时伤害所有等待策略**(否则就是容量收紧的重演)。

### 推荐机制:时变电网碳强度(brown carbon factor 随时间变化)

| | 拥堵代价(已失败) | **时变碳价(推荐)** |
|---|---|---|
| 代价落在谁身上 | 所有等待者(同时释放) | **只落在"等进更脏时段"的人** |
| 避开它需要什么 | 少等就行(零信息可得) | **必须知道未来价格/绿电曲线** |
| 是否伤害控制臂 | 是(nowait 也崩) | 否(nowait 不等,不承担) |

它同时满足条件 3 和 4,而且比方波深槽**更贴近真实电网**,论文里更好辩护。
注意:此时「等待」变成双侧风险 —— 盲策略"看到黑就等"会主动踩坑,
这正是 SQT2 缺的那一半。

### ⚠️ 第一步不是写代码,是离线证明

**在建任何场景、跑任何仿真之前**,先用纯离线计算回答:

> 在这个设计下,一个**零信息最大化等待**策略(naive)与 clairvoyant 的
> 逐作业碳排差是多少?这个差是否 ≥ 预注册门槛(建议 ≥10%,与历史一致)?

做法参照 SQT2 的离线分解(见 §2.1 的 `clair_forgone` 计算方式):对每个作业,
在确定性的绿电/碳价曲线上,分别计算"立即跑""等到绿""等到 backstop 强派"的碳排,
统计三者不同的作业占多少 MI。**若离线算不出差距,就不要建这个场景。**
SQT2 烧掉几天,正是因为先建后测。

---

## 4. 协议纪律(继承自 SQT2,不可协商)

1. **预注册**:所有阈值在跑之前写死进代码/文档,事后不得因结果调整;
2. **cal / held-out 分裂**:两套独立种子的序列与 trace,held-out **只跑一次**;
3. **冻结空间底座**:所有时间臂共享逐位相同的空间动作(SQT2 用冻结 blind PPO
   route-only + 容量溢出盾);任何臂不得有专属的路由规则(见 §5 坑 ⑤);
4. **最强盲对手在 calibration 上冻结**,判据是**合同下的碳**,不是分类正确率;
5. **合同**:completion@horizon / terminal / ontime 三合同;**绝对阈值必须先测
   控制臂天花板**(条件 ④),够不到就改成相对条款(臂 ≥ 同锚控制臂 − ε);
6. **零效应窗口也是数据**,永远不得剔除;
7. **任何场景参数变更 = 新场景 ID + 全套重认证**(preflight、容量审计、
   hazard 冻结、cal/ho 重分裂),不得覆盖旧 artifact。

---

## 5. 运维坑(全部踩过,逐条会浪费你半天到一天)

1. **CPU 利用率**:`CloudletDescriptor` 默认 `cloudlet_cpu_utilization: 0.5` →
   作业实际执行**拉伸 ~2.5×**。任何认证数学假设 `runtime = MI/(PES×MIPS)` 的场景,
   **必须在实验块显式写 `cloudlet_cpu_utilization: 1.0`**(SQT2 四个块已写)。
   这一条曾污染 SQT2 全套 slack 标定、B_eff、backstop 与容量审计。
2. **torch 线程**:评测/探针脚本必须设
   `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1`。
   不设时 3 个 python 各吃 255% CPU,把 Java 仿真饿死 —— 实测**吞吐差 8.5 倍**。
3. **并行互杀**:老脚本里的 `pkill -9 -f "exe.edu.cspg.MainMultiDC"` 会杀掉
   **其它 tmux 会话**的网关。并行跑时必须删掉这行(症状:
   `Py4JNetworkError: Answer from Java side is empty`)。
4. **awk 的 nan**:`awk '$2==$2+0'` **不会**过滤字符串 "nan"(求和会变 nan)。
   用 `$2 ~ /^-?[0-9]+(\.[0-9]+)?$/`。
5. **发散检查点**:`drl-manager/logs/creg_eucrd_local_s3` 是**已发散**臂
   (clean 部署完成率 16.7%,论文附录 H 明文剔除)。**任何鲁棒性读数都不能用它。**
   健康的 C-regime 检查点在 `isambard_backup/rl-runs-full/`
   (EU-CRD: `creg_eucrd_s1` 1.0000 / `creg_eucrd_s2` 0.9999;
   Vanilla: `creg_van_s1/s2/s3` 全 1.0000)。⚠️ 该目录 **gitignore,不在 GitHub 上**,
   需要时从本机 rsync。
6. **EVAL_CONFIG_PATH**:所有评测/训练必须 `export EVAL_CONFIG_PATH=<repo>/config_C.yml`,
   否则静默回落到 config.yml 并丢掉整个实验块(症状:`KeyError: 'datacenters'`)。
7. **episode offset**:绿电序列偏移是 `(1009*k) mod range`。锚点若取 0..9,
   只会覆盖序列开头几个百分点。SQT2 用 `ANCHORS=(0,20,40,59,79,99,119,138,158,178)`
   + 快进 reset 来铺满全域并覆盖 ≥20 个不同槽实例。
8. **孤儿 JVM**:tmux 会话被杀后网关会残留且 **SIGTERM 杀不掉**(gradle wrapper 挡着),
   必须对整条进程链 `kill -9`。残留会持续吃 CPU 并拖慢后续所有跑。
9. **pgrep 转义**:双引号里的 `pgrep -f "a\|b"` 中 `\|` 是字面竖线,匹配不到 →
   会误报"进程已死"。
10. **legacy py4j parser**:日志里 "rebuild the jar to use the fast path" 是常态
    (`getStepAsFlatMap` 从未合入本分支 Java),不是故障。

---

## 6. 可直接复用的代码(全部在 GitHub 上)

| 用途 | 文件 |
|---|---|
| 场景/trace 生成(含 cal/ho 双种子分裂) | `drl-manager/gen_sqt2.py`, `gen_sqt2_trace.py` |
| 场景认证(暴露比例、预算类检查、对称键) | `drl-manager/preflight_scenario.py` |
| 绿窗容量正式审计(≥1.2×) | `drl-manager/sqt2_capacity_audit.py` |
| 盲阈值离线冻结 | `drl-manager/sqt2_hazard_calibrate.py` |
| 最强盲对手竞赛冻结(碳/SLA 判据) | `drl-manager/sqt2_blind_freeze.py` |
| 四门阶梯 + 冻结底座 + 溢出盾 + 三合同判决 | `drl-manager/sqt2_prescreen.py` |
| 容量前沿搜索(按比例缩放保 DC 异构) | `drl-manager/sqt2_cap_probe.py` |
| 训练入口 | `drl-manager/entrypoint_rlmodule_gtrxl.py --config config_C.yml --experiment X --total-timesteps 600000 --num-workers 6 --seed S --output-dir logs/D` |
| 评测入口 | `python -m src.baselines.evaluate --experiment X --global rllib --local rllib --checkpoint CK --new-api --shared-local --global-defer --episodes N --seed S` |
| 预报腐蚀(评测期,env 变量) | `FORECAST_PERTURB_MODE` ∈ {blend, noise, shuffle, anti, **panti, bias, pshuffle**} + `FORECAST_PERTURB_EPS`;后三个是渐变模式,eps=1 与 anti/shuffle 逐位等价(测试锁定) |
| 部署审计器(滚动 Pearson χ) | `TRUST_GATE_SOURCE=resid` + `TRUST_GATE_MODE=log\|gate\|repair` + `TRUST_GATE_LOG=<csv>` |

测试:`drl-manager/tests/test_sqt2_*.py`、`test_gen_sqt2.py`、`test_graded_corruption.py`
(共 90+ 条)。**新增任何机制都要配测试**(仓库规则,见 `CLAUDE.md`)。

---

## 7. 建议的执行顺序

1. 读本文 + `CLAUDE.md`(项目规则:必配测试、必报开销);
2. **离线可行性证明**(§3 末),不过关就停,把结论写回来;
3. 过关 → 写场景生成器 + 实验块(记得 `cloudlet_cpu_utilization: 1.0`)+ 全套认证;
4. cal 上冻结最强盲对手与阈值 → 四门阶梯 shakedown;
5. 只有 shakedown 全绿,才跑 **held-out 一次性判决**;
6. 判决过线,才谈训练 RL(还需表示能力与奖励一致性两道审计)。

**不要跳步。**前九个考场里有六个是因为跳步才浪费了算力。

---

## 8. 本机(非 GPU 机)当前在跑什么

三条论文证据评测链(2026-08-19 14:14 起,预计当晚 22-23 点出):
`sweep_van` / `sweep_eucrd`(腐蚀强度退化曲线,两臂)、`audgrid`(审计器网格)。
GPU 机的工作与之**互不冲突**。
