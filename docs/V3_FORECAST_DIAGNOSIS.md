# v3 预报失效诊断:目标、现状、已排除项、活假设

状态截至 2026-08-13 11:30。写给接手分析的 agent。
不熟悉这个项目的话,**§0 是必读的**,它说明我们在做什么、v3 为什么存在、什么算成功。
赶时间的话读 §0 + §1 的三行表就够决定从哪下手。

---

## 0. 我们在做什么(先读这节)

### 0.1 整体目标

碳感知的数据中心调度。系统是**分层 DRL**:一个全局路由器把作业分派给多个地理分布的 DC
(或者按住不发),每个 DC 一个局部智能体决定何时把队列里的作业放上机器。
绿电(实测风电)只在部分 DC 有,且随时间大幅起伏,所以**同一个作业在不同地点、
不同时刻执行,碳排放差很多**。目标是在**不牺牲完成率**的前提下把碳排放降下来。

**唯一的比较原则**(2026-07-27 冻结,见 `paper_materials/SIMPLE_PLAN.md`):
同考场 + 同奖励配方 + argmax 解码 + iso-completion。跨配方、跨考场的数字永不同表。
**碳只在完成率 ≥99.5% 时可比** —— 掉活儿会压低碳,这是本项目最常见的假胜利来源。

### 0.2 正在投的论文:EU-CRD(ICLR 2027,摘要 9-18,正文 9-25)

> **EU-CRD: Managing Forecast Trust in Carbon-Aware Reinforcement Learning
> Scheduling via Epistemic Credit Redistribution**

论文的主张是关于**信任管理**,不是关于"预报有用":
DRL 调度器把预报直接吃进观测,训练时预报大多准确,策略于是学会**无条件信任**它;
标准训练把预报的贡献折进单一回报里,策略永远学不会"只在该信的时候信"。
部署时预报质量一降,这份信任就变成负债。
EU-CRD 用反事实估计来衡量预报对每个决策实际帮了多少、就只给这么多信用,
并且只在模型对这个估计有把握时才采用它,否则退回无预报信号;
外加一个零推理开销的部署期审计器。

**论文的主表数据不依赖 v3**,来自 C-regime 与 rwtight 两个考场,已经采完。

### 0.3 v3 是干什么的,以及为什么它重要

v3 **不是**论文的必需品,它是为了补一个**地基性的弱点**。

EU-CRD 的前提是"预报是有价值的,所以无条件信任一个坏掉的预报会造成损失"。
但如果在我们自己的考场上,**一个完全看不到预报的盲臂在同完成率下能追平甚至打平
看得到预报的臂**,那这个前提在审稿人眼里就站不住:
既然预报本来就不值钱,管理对它的信任又有多大意义?

所以我们需要一个考场,能干净地证明:**在同完成率下,有预报的策略碳排放显著低于无预报的策略。**
这就是 v3 的全部目的。

**v3 是第七次尝试。**前六次全部 null(`memory/project_workload_root_cause.md`、
`project_forecast_not_spatial_loadbearing.md`)。为了不再白烧算力,v3 带了一个
13 项预检查门(`drl-manager/preflight_scenario.py`),全部通过才开训。

**成功判据**(预注册,跑之前定死):
godeye(全知预报)vs noforecast(盲臂),argmax 解码,iso-completion,**碳差 ≥ 10–13%**
(13% 是实测噪声底的上界,见 `memory/project_eval_noise_floor.md`)。
timecap(真实预报模型)不进认证,只作信息。

### 0.4 这份文档要回答的问题

v3 的判据**没有达成**。这份文档记录:达成到什么程度、我们排除了哪些解释、
还剩哪些活假设、以及每条假设的证据强度。

**给接手的 agent**:目标不是"让 v3 出个好看的数",而是**搞清楚为什么预报在这里不载重**。
如果结论是"这个机制在我们的系统里就是不载重",那也是一个要如实写进论文局限的结果 ——
按场景设计的机制而不是按结果去调,这是研究设计和造假的分界线。
本文每条结论后面都附了产生它的命令或文件,**请优先复核证据本身,不要接受本文的转述**。
凡是标注"未定量"或"待验证"的地方,就是真的没证实。

---

## 0b. v3 场景是什么

配置键 `experiment_v3_oracle` / `experiment_v3_noforecast`(`config_C.yml`)。

**拓扑**:8 个 DC,其中 **4 个有绿电**(DC 0/1/2/5,turbine 9012+9036 / 9095+9091 / 9096 /
9101+9103),4 个纯棕电(棕电碳因子 0.55–0.92,给路由制造真实的地理碳差)。
`compressed_power_divisor = 5103.4`,`idle_host_power_down = true`(服务器功率非比例 → 整合杠杆)。

**绿电**:SDWPF 实测风电,stretch-6 生成序列。COMPRESSED 模式,**1 CSV 行 = 1 仿真秒**。
闭卷:`green_episode_offset_range = 4800`,每局按 (1009·k) mod 4800 平移绿电窗口,
防止策略背诵"第几步风大"。**预检查强制 4 个绿电 DC 同步**——这是有意的,
目的是掐死空间替代、逼出时间杠杆。

**负载**(`traces/v3b_n1200.csv`):

| | 值 |
|---|---|
| 作业数 | 1200 |
| 到达时间 | 1 – 5562 s |
| 执行时长 | p5 166s / **中位 372s** / p95 1000s(= 占用的风电行数) |
| 松弛 slack | p5 1290s / 中位 2066s / p95 2918s |
| slack / 时长 | 中位 **5.3×**(有充分延迟空间) |
| 局长 | ≈ 6756–7020 步 |

**动作**:全局路由器每步对 128 个槽位(`global_routing_batch_size=128`)各选一个
`{0..7} ∪ {defer}`,共 9 个选项,factored 空间 `{0..8}^128`。
局部智能体 `local_dispatch_mode = dispatch_rate`,动作是 `Discrete(65)`
("这一步放出多少作业",0 = 全按住)。deadline backstop 对所有臂一视同仁。

**两臂只差**:`forecast_mode: full|none`,外加(**这是缺陷,见 §5**)
`window_carbon_source: actual|persistence`。

---

## 1. 结论

**一句话**:策略读预报、按预报行动,但把它用在了一个被设计成空的杠杆上(空间),
在真正的杠杆上(时间)学出了反号的反应,而承接时间杠杆的局部策略整个训练一步没挪。

四条常见解释都有实测反驳(§3):不是解码、不是奖励错位、不是场景几何、不是表征丢信息。

**三个可独立追查的问题,按证据强度排序:**

| # | 问题 | 证据 | 状态 |
|---|---|---|---|
| **A** | 消融是双变量的:`window_carbon_source` 让 oracle 额外拿到一个无条件的价格折扣 | 两臂放置完全相同(绿电占比 92.1/90.5/91.5%)、物理碳相同,per_action 却差 37% | 验证臂 `oracle_sp` 已训完 seed1,**结果见 §6b:bug 是真的,但修掉之后策略干脆不用预报了** |
| **B** | 预报被用在空杠杆上:空间敏感度 0.47–0.53,时间杠杆反号 | 探针 §3.5 / §4,盲臂阴性对照 0.016 通过 | 机制清楚,**为什么学反**待查(§7.3) |
| **C** | 局部熵全程平坦(97% max)但权重在动(ck8→ck10 L2 9–10%);argmax 下学出**臂各异**的每 DC 固定 release 率 → 训练期打散启动时刻 + 评测期 local co-learning 混杂 | §2b(第三轮复审修正),12 格动作日志解析 | drain 去混杂评测已排队;V3.1 固定 drain(§6d 步6) |

A 和 B 已有闭环证据链,C 是今天(08-13)才从 `result.json` 里看出来的,**最值得先开工**。

---

## 2. v3 实验结果(argmax,10 局,12 格全齐,08-13 03:56 收工)

| 臂 | seed | ck | carbon/MI | 完成率 | green% |
|---|---|---|---|---|---|
| oracle | 1 | ck8 | 0.2494 | 98.62% | 50.23% |
| **oracle** | **1** | **ck9** | **0.2477** | **99.60%** | 51.09% ← 唯一 iso 合格 |
| oracle | 1 | ck10 | 0.2687 | 98.12% | 52.63% |
| oracle | 2 | ck8 | 0.2730 | 97.78% | 55.22% |
| oracle | 2 | ck9 | 0.2829 | 95.02% | 54.83% |
| oracle | 2 | ck10 | 0.2752 | 97.52% | 54.98% |
| 盲臂 | 1 | ck8 | 0.2382 | 91.84% | 54.43% |
| 盲臂 | 1 | ck9 | 0.2276 | 97.70% | 51.07% |
| 盲臂 | 1 | ck10 | 0.2317 | 94.61% | 53.10% |
| 盲臂 | 2 | ck10 | 0.2691 | 89.93% | 60.11% |
| 盲臂 | 2 | ck11 | 0.2593 | 91.52% | 56.31% |
| 盲臂 | 2 | ck12 | 0.2574 | 93.60% | 56.94% |

聚合(各 6 格中位数):

| 臂 | carbon/MI | 完成率 |
|---|---|---|
| oracle | 0.2708 | **97.95%** |
| 盲臂 | **0.2478** | 92.72% |

**⚠️ 这不是"盲臂赢"。**盲臂碳低 8.5%,但完成率低 **5.2pp** —— 掉活儿会压低碳,
这正是 iso-completion 规则要防的事。按预注册规则(≥99.5%)筛,**合格的只有一格:
oracle s1 ck9(0.2477 @ 99.60%),盲臂一格都没有**。

所以 v3 能安全声称的是:**预报买到的是完成率,不是碳。**
这和更早的记录一致(`memory/project_forecast_ablation.md`:
"forecast value = SPATIAL load-balancing keeping 100% completion;
noforecast dumps on 1 green DC, overloads, drops 18.5% work")——
v3 复现了已知模式,不是一个新的 null。

⚠️ 曾经的错误读法(留档以免重犯):只拿盲臂 s1 ck9(97.70%)对 oracle s2 ck8(97.78%)
得出"盲臂低 17%",那是单格对单格,而且把盲臂最好的一格对上了 oracle 较差的种子。
12 格齐了以后这个读法不成立。噪声底见 `memory/project_eval_noise_floor.md`(跨 ck 10–13%)。

### 2c. ⭐ drainfix 判决反转(08-14 02:00,6 格 @ 统一 drain local)

同 checkpoint、同 offsets,两臂统一 `--local drain` 重评(去掉 co-learned local 混杂):

| 格 | @DRAIN | 原判(@rllib local) | 变化 |
|---|---|---|---|
| oracle s1 ck9 | 0.2548 @ 99.70% ✓iso | 0.2477 @ 99.60% ✓iso | 基本不动 |
| oracle s1 ck10 | 0.2622 @ 98.96% | 0.2687 @ 98.12% | 微升 |
| oracle s2 ck10 | 0.2707 @ 97.95% | 0.2752 @ 97.52% | 微升 |
| **盲臂 s1 ck9** | **0.2197 @ 99.77% ✓iso** | 0.2276 @ **97.70%** ✗ | **完成率 +2.1pp,进 iso** |
| 盲臂 s1 ck10 | 0.2270 @ 99.42% | 0.2317 @ 94.61% | +4.8pp |
| 盲臂 s2 ck12 | 0.2531 @ 98.64% | 0.2574 @ 93.60% | +5.0pp |

**三个事实:**
1. **盲臂的完成率赤字是它的 co-learned local 造成的,不是 global 路由**——换 drain 后
   盲臂完成率 +2.1~+5.0pp,oracle 几乎不动(它的 local 本来就 drain-ish)。
2. drain 下 iso 合格格变成**同种子配对**:oracle_s1_ck9 0.2548 vs 盲臂_s1_ck9 0.2197,
   **盲臂低 13.8%**(踩在噪声底 10–13% 的上沿;单种子对,幅度保守读)。
3. **每一个盲臂 drain 格的碳都低于每一个 oracle drain 格**(0.220–0.253 vs 0.255–0.271),
   方向一致性 9/9 对。

**§2 的判决被推翻**:"预报买完成率不买碳"里那个完成率优势是 local 混杂的产物。
统一 local 之后,**v3 上预报既不买碳也不买完成率**;iso 下盲臂碳低 ~14%(单种子对)。
第七次 null 变得更干净也更严厉。连带确认:V3.1 无条件固定 drain 的决定被**实证追认**
——local 混杂不只是理论风险,它足以翻转判决。

**⚠️ 边界(第六轮复审,Codex,采纳)**:drainfix 是**带分布偏移的诊断**——它把与旧 local
联合训练的 checkpoint 临时换 local 评测,证明的是"旧判决的完成率优势=混杂"与"旧 oracle
global 在统一 local 下不如盲臂"(后者还有旁证:oracle 格换 drain 几乎不动,说明它本来就
不依赖它的 local),**但不预言"从第一步就固定 drain 重训的 V3.1 也必败"**。V3.1 的成败
仍由冒烟 P1 判,不由 drainfix 判。

---

## 2b. 训练收敛:全局学得动,局部一步没挪

⚠️ 先说取数:`monitor_worker*.csv` 里的 `last_global_*` / `last_local_*` 损失列
**全是 `nan`,`last_train_iteration` 恒为 0** —— 这些列在 iteration 0 冻结,
和 `progress.csv` 表头冻结是同一个坑。**损失/熵必须从 `result.json` 取。**

### 物理量轨迹(monitor CSV,按 episode 五等分取中位数)

| 臂 | 段1 reward | 段5 reward | 段1 碳 | 段5 碳 | 段1 green% | 段5 green% | 完成率 |
|---|---|---|---|---|---|---|---|
| oracle s1 | −45751 | −22576 | 3.656e-11 | 1.975e-11 | 38.03 | 56.53 | 100% 全程 |
| oracle s2 | −47145 | −23843 | 3.675e-11 | 2.138e-11 | 37.27 | 55.08 | 100% 全程 |
| 盲臂 s2 | −44432 | −28694 | 3.532e-11 | **2.015e-11** | 39.25 | **56.38** | 100% 全程 |
| oracle_sp s1 | −49755 | −38096 | 3.713e-11 | 2.348e-11 | 36.92 | 52.80 | 100% 全程 |

⚠️ **不要跨臂比 reward**(§5:两把尺子)。可比的是碳和 green%,
而这两列显示**盲臂末段与 oracle 持平甚至更好**。
oracle s1 的 monitor 数据丢了(被一次跑飞的重训覆盖),所以只有 3 条臂。

### 学习器曲线(result.json)

| 臂 | 迭代数 | VF解释度 首→末 | 全局熵 范围 | **局部熵 首→末** | **局部熵 全程min–max** |
|---|---|---|---|---|---|
| oracle s1 | 75 | −0.05 → 0.97 | — | 4.035 → 3.984 | 3.969 – 4.057 |
| oracle s2 | 75 | −0.02 → **0.99** | 0.015 – 0.997 | 3.981 → 3.984 | 3.926 – 4.033 |
| 盲臂 s1 | 75 | 0.01 → 0.77 | — | 4.024 → 4.026 | 3.955 – 4.061 |
| 盲臂 s2 | 87 | 0.19 → 0.98 | 0.027 – 0.874 | 4.002 → 4.008 | 3.890 – 4.050 |
| oracle_sp s1 | 75 | 0.23 → 0.94 | 0.008 – 1.208 | 3.997 → 3.982 | 3.953 – 4.031 |

**全局侧健康**:VF 解释度 ≈0 → 0.94–0.99,critic 学到了回报;全局熵在 0.85–1.2 的跨度内活动。
**不是欠拟合。**

**⚠️ 局部侧:熵平坦 ≠ 冻结(08-13 第三轮复审修正了本节的原措辞)。**
本文初版写"局部策略一步没挪",**错了**——那是从"熵全程平坦在最大值的 97%"
(3.89–4.06 / ln(65)=4.174)做的过度推断。复审 agent 做了我在边界③里承认没做的实验
(dump 实际动作 + 权重对比),三层结论:

1. **V3 两臂没改 local 的配置/观测/奖励** —— 是。两臂 local 完全相同
   (shared_local_policy / dispatch_rate / Discrete(65) / 无碳奖励 / 同超参)。
2. **"local 没训练" —— 不是。**`policies_to_train=list(policies)` 包含 local
   (`train_rlmodule_gtrxl.py:706`,已核);~4.8M local 步/臂(每 global 步 × 8 DC),
   461,730 参数,梯度持续非零,**ck8→ck10 权重 L2 变化 9–10%**(复审实测)。
   熵平坦的同时权重和众数都在动——熵不是策略的全部。
3. **"local 对两臂比较没影响" —— 不能断言,它是活的混杂变量。**
   复审解析 12 格 argmax 评测日志(每格 ~56–57 万 local 动作):
   action=0 出现 0%,action=64(严格 drain)也是 0%;实际形态是**每 DC 一个近似固定的
   release 率**,而且**两臂学得不一样**(nofc s1 ck10 主导动作 [37,53,46,63,34,46,63,34],
   oracle s1 ck10 [13,48,2,16,48,2,16,48]——后者有 2/16 这类小值,global 一次送大批量时
   可能成为节流阀,改变启动时刻)。

**两种危害要分开说**(原文混为一谈):
- **训练期(采样解码)**:熵 ≈97% max → 采样出的 release 数近似均匀 → 作业启动时刻
  被注入噪声,加重全局的信用分配难度——"下游打散"**只在训练期成立**。
- **评测期(argmax)**:每 DC 固定 release 率,不打散,但**臂各异**(local 与各自的
  global 路由分布共同演化)——所以 `--local rllib` 的 P3 比的不只是 global 预报策略,
  **混入了 local co-learning 差异**。
- 到达稀疏时正 release 率大多能清空队列(完成率≈100%、invalid=0),所以 local 大体
  "drain-ish" 没崩,但**不等价于 drain**;现有日志没有
  requested/placed/waiting-after-dispatch,节流比例算不出来。

去混杂手段(已排队):同 checkpoint、同 offsets、两臂统一 `--local drain` 重评
最终 ck——若预报臂仍输,local 混杂即被排除。V3.1 训练侧固定 drain(§6d 步6)
因此不只是简化,**是在清除一个已证实存在的混杂源**。

---

## 3. 已排除的假设(附反驳证据)

### 3.1 「argmax 解码把预报优势丢了」— 否

训练期本来就是采样解码。取各臂末 25% 局的中位数(全部 100% 完成):

| 臂 | carbon/MI | green% |
|---|---|---|
| oracle s1 | 2.183e-11 | 52.52 |
| oracle s2 | 2.243e-11 | 52.01 |
| **盲臂 s2** | **2.108e-11** | **55.29** |

采样解码下盲臂也不输。数据来源 `drl-manager/logs/v3_*/monitor_worker*.csv`
(列 `carbon_per_mi` / `green_ratio` / `finished_over_received_rate`,免费的训练期物理量)。

### 3.2 「奖励函数和真实目标没对齐」— 否(但仅指 episode 级;见本节末的关键限定)

臂内、只取末 25% 平台期、逐局:`corr(global_agent_reward, carbon_per_mi)` = −0.982 / −0.991 / −0.891。
再控制"本局可用绿电总量"(`green_used_wh + green_waste_wh`)做偏相关,仍为 **−0.935 / −0.977 / −0.690**。
不是共同因子撑起来的。

**⚠️ 关键限定(08-13 第五轮复审):这只是 episode 级的测量对齐,不是动作级的激励对齐。**
跨局相关证明"碳更低的整局得分更高",不证明"同一状态下绿电将至时 defer 的 advantage
高于立即 route"——前者由共享因子(本局绿电多寡)驱动,后者是局内边际比较。两者同时为真,
恰好解开"奖励对齐却学不出预报使用"的悖论:**测量层对齐,激励层反向**(route offset +
逐次 defer 税,§6c 已证实)。本节的"否"只排除了"尺子坏了",没有排除"工资表指错方向"。

### 3.3 「场景几何不对,作业太短撑不到预报窗口」— 否

COMPRESSED 模式下 **1 风电行 = 1 仿真秒**(`GreenEnergyProvider.java:76`)。
v3 作业 `length` 中位 14.9e6 MI,1 PE,`vm_pe_mips=40000` → **中位 373 秒 = 373 行**,范围 150–1000。

在真实风电序列上做样本外线性回归(预测作业执行窗口内的平均绿电):

| 作业窗口 | 只看当下 R² | 有预报 R² | 预报增量 |
|---|---|---|---|
| 60 行 | 0.885 | 0.947 | +0.06 |
| 240 行 | 0.443 | 1.000 | +0.56 |
| 600 行 | −0.254 | 0.832 | **+1.09** |

373 行正落在预报载重的带里。**几何是对的。**

### 3.4 「4 标量把预报平均没了(表征损失)」— 基本否

观测里的预报只有每 DC 4 个标量(`src/prediction/timecap_godeye_provider.py:9-12`):
`short_mean`(240 行均值)、`short_trend`、`long_mean`(1000 行均值)、`long_peak_timing`。
对比原始预报向量(20 bin),样本外 R² 是 **0.93–1.00**,只在窗口 ≠ 240 时掉 5–17 点。
信息损失存在但不足以解释 null。

### 3.5 「信息进了观测但到不了动作」— 否(这条最强)

新写的探针 `drl-manager/probe_forecast_sensitivity.py`(纯 CPU,不碰 Java 网关,分钟级)。
做法:直接加载 checkpoint 里的 `global_policy` RLModule,**只改一个通道**,看动作分布怎么变。
三个通道:`forecast`(4 个预报特征)、`control`(`dc_current_green_power_w`,已知策略在用)、
`null`(`dc_cumulative_wasted_green_wh`,无意义,给噪声底)。

```
             TV distance   argmax flips   mass follows   forecast/control
oracle s1    0.4419        69.0%          +0.3022        0.527
oracle s2    0.3849        67.0%          +0.2632        0.469
盲臂  s2     0.0172         8.6%          +0.0013        0.016   ← 阴性对照
（null 通道三臂均 ≈0.001,0% flip）
```

盲臂的预报通道被消融器屏蔽,探针读出 0.016 ≈ 噪声底 —— **探针本身通过了阴性对照**。
两个 oracle 种子则读出 0.47–0.53,而且概率质量确实搬向"预报绿"的 DC。

复现:`.venv/bin/python probe_forecast_sensitivity.py --checkpoint <ck> --trials 40`

---

## 4. 活的发现:用对了通道,用错了杠杆

同一个探针加的第二段,问**时间**问题:所有 DC 的预报保持一致(消除空间信号),
当前绿电固定,只切换"绿电将至"(现在低、窗口内升)与"绿电将去"(现在高、窗口内降)。

```
                    P(defer|绿电将至)   P(defer|绿电将去)      差值
oracle s1               0.0024              0.0092          −0.0067   ← 反号
oracle s2               0.0064              0.0168          −0.0105   ← 反号
盲臂   s2               0.0290              0.0276          +0.0013   ← ≈0,符合预期
```

三件事同时成立:

1. **符号是反的**。绿电将至时它延迟得更少,绿电将去时延迟得更多。
2. **总延迟率极低**。oracle 0.24%–1.68%,**比盲臂(2.8%)低 4–10 倍**。
3. **空间维度上它很敏感**(§3.5),但 v3 预检查**强制绿电 DC 同步**,
   同步之下"哪个 DC 更绿"几乎没有信息 —— 这个杠杆是设计上就空的。

**即:预报被花在了空杠杆上,而真杠杆上学反了。**

⚠️ 局限:探针的基础观测是合成的(见脚本 `base_observation`),
所以 **P(defer) 的绝对值不代表真实 rollout**。可信的是符号、以及臂间对比
(其余输入逐字节相同)。想要真实 rollout 的动作分布需要在评测里打点,尚未做。

---

## 5. 机制假设(待验证,但有强旁证)

`config_C.yml` 里 v3 两臂只差两个键:

```
forecast_mode         : full | none
window_carbon_source  : actual | persistence      ← 这个是问题所在
```

`MultiDatacenterSimulationCore.computeDcCostFeatures()`(约 L804)里:

```java
persistence: windowGreenW = getCurrentGreenPowerW(now)                    // 现货价
actual     : windowGreenW = getMeanFutureGreenPowerW(now, runSteps=373)   // 窗口均价
greenRatio = Math.min(1.0, windowGreenW / currentPowerW)
marginalKg = (MI/miPerKg) * (greenRatio*0.01 + (1-greenRatio)*0.55)
```

**假设:`actual` 定价取消了等待的动机。**当预报说"绿电将至",oracle **现在就派**这个动作
已经按未来 373 步的均值定价 —— 它无需等待就能拿到"绿电将至"的信用。而延迟要付
`defer_base_cost=0.5` 加 urgency ramp。于是"现在派"严格占优。
盲臂按现货价结算,派进波谷立刻挨罚,所以它延迟得更多。

与观测吻合的三点:oracle 延迟率低 4–10 倍;温度杠杆反号;
两臂**落位完全相同**(绿电 DC 占比 92.1 / 90.5 / **91.5%**,逐 DC 计数在噪声内)
而 per_action 分数差 37%(−1.60/−1.64 vs −2.52,已按 episode_length 归一)。

**这同时意味着 v3 是双变量消融**(观测里的预报 + 奖励里的定价),即使跑出正结果也无法归因。
预检查第 13 条"arms differ only as intended"是通过的 —— 差异确实有意,但两件事被绑在一条臂里了。

**未定量的部分**:per_action 的 37% 缺口里,Jensen/截断效应在真实风电序列上只能解释到
**0.88**,实测 **0.617**。残差取决于路由时刻 `currentPowerW` 的分布
(`idle_host_power_down=true` 让空闲 DC 的 P 变小才触发 `min(1.0,·)` 截断),
光靠 CSV 定不下来,**需要在仿真里打点记录 `(currentPowerW, green_now, green_window_mean)` 三元组**。
这是一个明确的、约半小时工作量的开放任务。

---

## 6. 建议的修复与预注册判据

**修复**:两臂用**同一个** `window_carbon_source`。推荐都用 `persistence`(现货价,
谁也不拿无条件折扣),预报只从观测进入策略。改完之后跨臂回报才重新可比,
任何回报差都是行为挣来的。

**重训后的预注册判据**(写在跑之前,不许事后改):

- **P1** 温度杠杆符号翻正:`P(defer|绿电将至) − P(defer|绿电将去) > 0`,两个种子都要。
- **P2** oracle 延迟率 ≥ 盲臂延迟率。
- **P3** 同完成率(≥97%)下 oracle 的 carbon/MI 低于盲臂最好的格子,**幅度 > 13%**
  (噪声底上界,见 `memory/project_eval_noise_floor.md`)。
- P1/P2 过而 P3 不过 → 机制修好了但物理收益不够,回到"这个考场的碳杠杆本身太小"。
- P1 不过 → 定价假设错,§5 作废,回到 §7 的开放问题。

**已经排好,不用再配**:`config_C.yml` 里新增了 `experiment_v3_oracle_sp`
(= `experiment_v3_oracle` 但 `window_carbon_source: persistence`)。核验过:

```
sp vs experiment_v3_noforecast  只差 forecast_mode         ← 干净的单变量消融
sp vs experiment_v3_oracle      只差 window_carbon_source
```

**盲臂不需要重训** —— 它本来就用 persistence,现有 checkpoint 就是配对对照。
所以成本是 15h/种子 × 2 种子,不是四条臂。

跑法:`local_eval_rt/run_v3_sp.sh` 已 armed(等当前评测退出后自动开始,进程门),
训完自动跑探针判 P1,结果追加进 `local_eval_rt/v3_sp.txt`。
P3 还需要一轮评测,尚未排。

---

## 6b. 统一定价臂的首个结果(seed 1,08-13 09:18 训完)

`experiment_v3_oracle_sp` seed 1 训完,探针结果:

| | 旧 oracle(actual) | **oracle_sp(persistence)** | 盲臂 |
|---|---|---|---|
| 预报敏感度 / 控制通道 | 0.47–0.53 | **0.078** | 0.016 |
| P(defer\|将至) − P(defer\|将去) | −0.0067 / −0.0105 | **+0.0024** | +0.0013 |
| 总延迟率 | 0.24% / 1.68% | **2.93%** | 2.76–2.90% |

- **seed 2 复刻(08-13 15:28 确认)**:敏感度 0.074(s1: 0.078),时间差 +0.0024,
  control 0.894——**"sp 忽略预报"不是单种子偶然,是系统性的**。
  机制解释(第五轮复审,Codex):actual 定价把未来窗口揉进当下 route 奖励,
  **预报到奖励只有一步**;persistence 一开,路径变成 defer→等待→未来route→节碳的长链,
  敏感度随之塌缩——同网络同观测只换定价的**干预证据**。
  另:探针只加载 global policy,**local co-learning 影响不到这个读数**——
  local 混杂的解释范围因此被限定在 P3 类结果比较(它能改"输多少",
  不能解释"global 为什么不读预报";训练期 local 采样噪声仍是背景信用噪声源之一)。
- **P1 名义通过**(符号翻正),**P2 通过**(2.93% ≥ 盲臂 2.8%)。
- **但要诚实读**:三个指标全都落在盲臂的数值附近。延迟率 2.93% ≈ 盲臂 2.8%,
  时间差 +0.0024 ≈ 盲臂 +0.0013,预报敏感度 0.078 离盲臂的 0.016 比离旧 oracle 的 0.53 近得多。
  **更准确的描述不是"反号被修正了",而是"反号行为消失,换成了近乎不理会预报"。**

**这意味着:`actual` 定价是唯一让策略去看预报的东西,而它教的方向是错的。
换成现货定价之后,奖励里就没有任何一项会奖励"正确使用预报",于是策略干脆不用了。**

延迟的收益(等到绿电来了再跑)只在**局末的碳**里体现,被摊到 128×T 个动作头上;
而 per-action 项只能看到当下的现货价,于是等待永远是纯成本。
**这正是信用分配问题本身**,也就是 EU-CRD 这篇论文的题目。

P3(碳)尚未判:`local_eval_rt/run_v3_sp_eval.sh` 已 armed,等 seed 2 训完(约 08-13 14:40)
后自动评测两个种子各 3 个 checkpoint,结果追加进 `local_eval_rt/v3_sp.txt`。

### 6b.1 训练期逐局配对(08-13 下午补,推翻了"收敛到盲臂"的预期)

闭卷机制让每局考不同的绿电窗口(offset 按局号确定,**各臂日程相同**),
所以按 (worker, 局号) 配对比较能控掉考题难度。末 40% 局:

| 配对 | Δ碳/MI 中位 | 更差的局数 | Δgreen% |
|---|---|---|---|
| oraclesp − nofc(**同账本**) | **+2.1e-12(≈+10%)** | **30/30** | −1.9pp |
| oraclesp − oracle(同预报,不同账本) | +3.1e-12 | **36/36** | −2.3pp |

三个新事实:

1. **同账本下,预报臂不是追平盲臂,而是每一局都更差**(30/30,符号检验 p≈1e-9)。
2. **旧 oracle(泄漏定价)物理上反而比 sp(干净定价)好**,36/36。
   窗口均值定价虽然不正当(用了盲臂看不到的真值),但它**方差小**——
   现货绿电是 feast/famine 的,现货定价给每个动作注入巨大的奖励噪声。
   泄漏定价顺带做了方差缩减,这可能才是它"帮助学习"的真实机制。
3. 我上午从十等分轨迹怀疑的"sp 训练震荡"是**误读**:逐局看,nofc 在同样的局号上
   有同样的起伏(考题难度波),sp 只是每局都稳定地差一点,不是不稳定。

**为什么同账本的预报臂会输给盲臂?**两个候选,现有数据分不开:
(a)预报特征成了干扰项——奖励噪声大时,策略把噪声归因到多出来的观测维度上;
(b)单种子噪声。**oraclesp_s2 今天训完即可判**(用同样的配对法对 nofc_s2)。

**对 P3 评测的预期要修正**:不再预期"sp ≈ 盲臂",按训练数据应预期 **sp 略差于盲臂**。
若评测证实,§5 的结论升级为:泄漏定价既是双变量消融的缺陷,**也是一次意外的方差缩减**;
干净的修复必须补上合法的方差缩减(见 §7.8)。

---

## 6c. 外部 agent 复审的核实结果(08-13 下午)

另一份独立分析 `docs/V3_FORECAST_RECOMMENDATIONS.md` 提出了三个结构性论断,
本文作者逐条对代码核实:

1. **路由补贴 +2,defer 拿不到 —— 证实。**`accumulatePerActionReward` 给路由
   `−w_c·margNorm + 2·probComplete`(可行路由 ≈ +1.8),defer 槽拿 `−0.5 − 2·urgency`
   (`MultiDatacenterSimulationCore.java:605-626`)。同一个 categorical 里 route 选项
   整体抬高 ~+2.3,softmax 被结构性推向"立即路由"。**这与所有臂 defer 率 ≤3% 完全吻合。**
2. **defer 逐次重复收费 —— 证实。**被 defer 的作业 `requeueCloudletToTail`(L627)
   回队尾,每次再进 batch 再收一次费。等待成本随重逢次数线性涨,和 SLA 风险无关。
3. **per_slot_reward Java 已产出、Python learner 零消费 —— 证实**(全仓 grep 零命中)。
   128 槽共享同一个 timestep advantage。

**⚠️ 拆除警告(复审文档没有的历史)**:这两个反 defer 项是 2026-06 治 **all-defer 塌缩**
(全 defer → 仿真慢 20 倍 → 完成率归零)时故意装的("Honest deferral cost (Route A)",
注释原文)。V3.1 拆除它们时必须保留 deadline backstop + 超时筛,
并把"defer 率轨迹"加进监控,防止旧病复发。

其余判断:§9 的观测维度担忧**已被现状满足**(`forecast_mode=none` 是把 4 个预报特征
置零,不是删键,两臂维度相同,`hierarchical_multidc_env.py:1351-1356`);
GAE 表格数学正确但忽略了 critic bootstrap。

**关于 backlog 观测的精确表述(08-13 第二轮复审修正,已逐条对代码核实):**
不是"critic 看不到 backlog"。`upcoming_cloudlets_count` 的 Java 来源是
`globalBroker.getGlobalWaitingCloudletsCount()`,**全局等待队列总量是可见的**,
且 `batch_cloudlet_pes/mi` 暴露队首 ≤128 个任务的规模。真正缺的是**消除状态混叠**的信息:
deferred 回队(`requeueCloudletToTail`)后与新到达**同队不可分**,批槽位只有 pes+mi,
没有等待年龄、defer 次数、deadline slack、backlog 剩余 MI。准确说法:

> critic 能看到聚合 waiting backlog,但无法区分 deferred/new arrivals,
> 也看不到等待年龄和 deadline slack;由此产生的状态混叠使 V(s) 难以稳定桥接
> defer 的延迟收益。

所以 V3.1 **不要只加一个 `global_deferred_backlog` 标量**,最低限度:
`global_deferred_count`、`global_deferred_mi`(或 deferred workload fraction)、
每个 batch 槽位的 `wait_age`、`deadline_slack`(或 defer_count)。

另:高 VF 解释度(0.94–0.98)**不能证明 critic 学会了 defer 信用**——defer 只占 ~3% 的
transition,整体指标由 route 样本和考题难度主导。V3.1 要加三个条件化诊断:
defer transition 的 TD residual、defer 条件下的 explained variance、
按 backlog/slack 分桶的 value calibration。**λ 只有在 defer 样本的 TD residual
仍明显异常时才动,且先 0.99 试点,不上 0.999。**

**训练侧确定性 local drain 目前不存在**(评测侧 `--local drain` 有,训练入口仍会创建并
训练 local RL policy)。最小风险实现:加 `fixed_local_scheduler: drain` 配置,在 PettingZoo
wrapper 调底层 `step()` 前把 local actions 覆盖为最大合法 dispatch;`policies_to_train`
只留 `global_policy`;local module 保留以满足 RLlib 接口但忽略其输出。

---

## 6d. 收敛后的 V3.1 执行序列(08-13 定稿,两轮复审合并)

1. 等 sp s2 + P3 跑完,不动现有进程。
2. **仿真内 heuristic 上界**(`oracle_hold_until_green.py` + drain 局部,同完成率)。
   若 heuristic 都省不了碳 → 先查 workload slack / 绿电窗口 / 碳差,**不要继续调 PPO**。
3. V3.1 基座:两臂统一 persistence + 共同冻结的分项 Welford
   (同一 reference policy 跑 2000 步产出唯一 normalization artifact,全 seed/worker/臂共用;
   defer 不以 carbon=0 进 z-score;记录 raw/normalized/clip rate)。
4. 奖励改造:`+2·p` 路由补贴 → `−2·(1−p)` 对称完成损失;defer 改 potential-based
   (`defer_base_cost=0`,`−w_s·[R(slack_{t+1})−R(slack_t)]`)。
5. 观测补齐(两臂同加):deferred count/MI 拆分、批槽位 wait_age、deadline_slack。
   仪表补齐:`local_dispatch_requested` / `local_dispatch_placed` /
   `local_waiting_after_dispatch`(量化 local 节流比例,现在算不出来)。
6. 训练侧固定 deterministic local drain,只训 global policy(实现见 §6c 末)。
   这不只是简化——§2b 已证实两臂 local 学出了不同的 release 率策略,drain 是在
   **清除已证实的混杂源**。
   6b. (已排队,先于 V3.1)存量判决的 drain 去混杂:v3 oracle/nofc 每 seed 最终 ck,
   统一 `--local drain` 重评。预报臂仍输 → local 混杂排除,§2 判决站稳。
7. **拆除的是反-defer 奖励项;deadline backstop、超时筛、defer-rate/forced-route 监控
   保留或新增**——护栏不跟着奖励项一起删,否则 all-defer 塌缩复发。
8. 100k 冒烟判 P1(探针现成),同时看 §6c 的条件 critic 指标。
9. 仅当 defer 样本 TD residual 仍异常 → 试 `gae_lambda=0.99`(不上 0.999)。
10. P1 仍失败 → factorized temporal gate(hold/balanced/release + spatial router),
    不再盲调奖励权重。

---

## 6e. 总战略:让预报被用起来(08-13 定稿)

**主线**:七次 null 的每个已证实原因都是同一件事——从"预报说绿电要来"到"等待省下碳"的
信用路径太长/太吵/太稀(373 步延迟 × 128 槽稀释 × 现货噪声 × 状态混叠 × 局部打散)。
手写 hold-until-green 在真实风电上早已赢过(Version B −15% 碳、rwdefer +28% 绿电),
因为它的信用路径长度为零。**工程原则:要么修通路径(赛道1),要么把假设空间缩到与
信用信号匹配(赛道2)。**

**战略安放(08-13 复审修正,原表述有选择性引用错误)**:rwtight 验证过**两次,结论相反**。
7 月老配方:godeye PASS −15~18% @ iso-comp。**8 月 1 日用最新 V3 对齐配方重认证:FAILED**
(盲臂 0.0577@100% 全场最优,godeye 0.063–0.081@96% 连完成率合同都不满足,
`local_rt_summary.txt` [rwcert] 行)——V3 配方把反应式 defer 教给了盲臂,和 v3 七连 null
同一个故事。老 PASS 另有三个未复审暴露面:无闭卷(offset 未设)、效应 15–18% 对噪声底
10–13% 只边际超出、配方已被迭代。**准确的现状:最新配方下没有任何考场显示预报的干净
正价值。**论文 claim B(rescue)作为老配方内部一致的故事仍站得住(其前提是"vanilla
过度信任预报",由 step-2 anti-forecast 双轴恶化证明,不需要"预报打赢盲臂");
但"预报有正价值"这个地基前提目前悬空 → **v3.1/赛道2 不是加强项,是论文前提的
真正补强**。仍按攻坚排法走,每步有决策门。

**三条赛道**:
- **赛道0(先行,一票定生死)**:仿真内 heuristic 上界(= §6d 步2)。≥15% → 进赛道1;
  <10% → 考场碳杠杆不足,改场景参数,不碰 PPO。这个数同时是"预报完美使用值多少"的标尺。
- **赛道1**:V3.1(§6d 步3–9)。**单项最大使能器 = 槽位级 slack 观测**:defer 决策的两个
  输入是「作业 slack × 绿电多久来」,前者当前完全不可见(批槽只有 pes+mi,urgency 只在
  奖励里不在观测里),后者只有 1000 行粗 peak_timing——策略连奖励最优规则都无法**表示**。
  门:100k 冒烟 P1。不过 → 不调权重,直接赛道2。
- **赛道2**:temporal gate 分解(hold/balanced/release × spatial router,= §6d 步10,
  G2 Tier 2/4);再难产 → **Tier 3 蒸馏保底**:用赛道0的启发式生成 (obs,action) 对,
  BC 预训练 gate 再 RL 微调——监督信号直接教"用预报",保证投稿前有正面展示。

**验收(预注册)**:行为 = P1 符号 + defer率 ≥ 盲臂 + P(defer) 随 slack/预报单调;
结果 = iso-comp 碳差 ≥13% 双种子;**机制 = anti-forecast 扰动使碳变差 ≥10%——
"被用起来"的终极证明是扰动它会痛**(前六考场多次"看似在用"最终是背诵/巧合,
只有扰动测试分得清;rwdefer 先例 anti +29%)。

**时间线**:本周赛道0 ‖ sp收尾 ‖ V3.1 代码;下周冒烟判门;8月底蒸馏保底兜住;
9-18 前 v3 能进则进,不能则论文靠 rwtight 站立,v3 转投稿后。

---

## 7. 给并行分析的开放问题

按我判断的价值排序。每条都独立,可以并行。

1. **per_action 缺口的残差归因**(§5 末)。在 `computeDcCostFeatures` 里打点,
   跑一局,记录三元组,看 0.88 → 0.617 的差从哪来。这是唯一一个我明确没做完的定量。
2. **真实 rollout 的动作分布**。§4 的探针用的是合成观测。
   在 `src/baselines/evaluate.py` 里把每步的全局动作分布 dump 出来,
   直接测真实 P(defer) 与预报的相关,验证探针结论。
3. **反号是怎么学出来的**。**部分回答(08-13,`docs/V31_FORECAST_LEARNING_TIMELINE.md`)**:
   可恢复的 ck8–ck10 × 两种子 = 6/6 全反号(Δ −0.0087…−0.0211),末段无翻转;
   预报敏感度 48–64% of control。⚠️ ck1–ck7 已被保留策略删除,**早期定型 vs ck8 前漂移
   分不出来**——V3.1 训练要保留全部 checkpoint(或每 ck 存探针读数)。
4. **v3 的碳杠杆到底有多大**。上界问题:在 v3 的真实动力学里,
   一个全知的规划器最多能省多少碳?`drl-manager/scan_voi_v3.py` 给的 78–84% 是抽象 cell 模型,
   不是仿真器。需要一个仿真器内的 oracle 上界。
   **08-14 更新:`oracle_hold_until_green.py` 作为上界仪器已被证伪**——它是 DC 级开关
   (绿电缺席+预报说要来 → 整 DC 全部 hold),在绿电同步的 v3 上行为二元:
   th≤0.40 全程死等(7199/7200,数字与 0.3 逐位相同),th≥0.6 从不等待,无中间点;
   且 0-hold 臂与基线数字不同(两路径差异不止 hold,单格 Δ 不可归因)。
   **必须换 slack-aware 逐任务 oracle**(第六轮复审规格,采纳):对每个任务,
   `可等时间 = deadline − now − 预计执行`;`预计省碳 = 预计能耗 × (当前碳强 −
   slack 窗内最低预测碳强)`;省碳 > 等待代价 且 slack 足够 且 backlog 未超限 → defer,
   否则 route。**只有这个仪器在同 offset、完成率 ≥99.5% 下仍省不到 10%,
   才有资格说"考场本身没有预报价值"。**在此之前,考场判决悬置。
5. **动作空间稀释**。全局动作是 factored `{0..8}^128`
   (`global_routing_batch_size=128`),一次 episode 的碳要摊到 128×T 个动作头上。
   对照 HPE SustainDC 的 LS 智能体是 `Discrete(3)`。
   低维控制头(受预报调制的"延迟激进度"标量,统一门控 128 个决策)是否能缩短梯度路径?
   这是 G2 阶梯里预注册的 Tier-4。
6. **预报剖面替代 4 标量**。`getFuturePowerPredictions(now, horizonSeconds[])` 现成,
   换成 12–20 bin 可以把 W=600 的 R² 从 0.85 拉回 1.00。便宜,可与 5 合并成一条臂。
6b. **⚠️ 诚实标注:V3.1 组合里其实没有真正的方差缩减器**。no_offset 是去偏置,
   incremental_urgency 是去重复计费,z-score 是修尺度(信噪比不变,§7.8 已注)。
   泄漏定价当年靠窗口均值实现的那种平滑,合法组合里没有等效物。若冒烟 P1 失败且
   条件 critic 诊断指向噪声,首选补救是**逐决策候选均值中心化**(对该 cloudlet 的
   8 个候选 DC 的 marginalKg 取均值作基线,奖励 = −w·(己选 − 候选均值)/σ)——
   这是控制变量式的真降方差,且不偷看未来;实现点在 computeDcCostFeatures 已为
   pickGreenestAvailableDc 算过全候选,复用即可。

7. **⭐ 局部策略为什么冻在最大熵**(见 §2b)。三步:
   ①dump 局部动作分布,确认它是不是真的平的(熵接近最大不等于无用);
   ②若确实平,查 `grad_clip=0.5` 是不是元凶(把它放到 5、局部 LR 提一档,
   跑 100k 步看熵动不动 —— `memory/project_gradclip_throttle.md` 有先例);
   ③若熵动起来了,重跑 oracle/盲臂,看时间杠杆是否复活。
   **这条我判断价值最高**:它可能是"时间杠杆整体不产生收益"的下游原因,
   而且和 1–6 完全正交,可以立刻并行开工。

8. **⭐ 合法的方差缩减**(§6b.1 的直接后果)。现货定价方差太大,泄漏定价靠偷看未来把方差
   压下去了。要一个不偷看未来的等效物,候选(按工程成本排序):
   (a)**分项 z 标准化,Welford 在线统计 + 2000 步后冻结** —— 照抄 SustainCluster
   (`rewards/predefined/composite_reward.py`),同时解决"固定归一把 1.6–17.5% 的
   DC 间价差压扁"的老问题;
   (b)per-action 奖励减去一个不看未来的基线(如各绿电 DC 现货价的均值,
   只保留"相对谁更好"的信号);
   (c)干脆去掉 per-action 碳项、只留局末碳 + 强化 GAE —— 风险大,信号更稀。
   另注:**SustainCluster 的碳定价 = 任务时长 × 当下 CI,正是我们的 persistence 方案**
   (`rewards/predefined/carbon_emissions_reward.py`),说明两臂同用 persistence 是该领域
   最接近的公开基准的标准做法,方向没错,缺的只是方差处理。

---

## 8. 已知的坑(踩过的,别重踩)

- **跨臂比 `episode_reward` 无效**(v3 与 `v2026_gamble_noforecast_symm` 两组)。
  只有这 3 个实验设了 `window_carbon_source`,其余继承默认、两边同源。跨臂只比物理量。
- **`progress.csv` 的表头在 iteration 0 冻结**,不能用它判断某个特征在不在。查 `result.json`。
- **不要编辑正在运行的 bash 脚本**。bash 增量读取,改长度会让执行流跳回去重跑早先的阶段。
- **筛选不能用完成率**。完成率是被比较的结果变量,拿它当门槛等于在被测量的量上做选择。
  只用 wall-clock 超时筛(全 defer 塌缩会让仿真慢 20 倍)。
- 其余见 `memory/project_eval_ops_pitfalls.md`(十条)。

---

## 9. 文件索引

| 用途 | 路径 |
|---|---|
| 预报敏感性探针(本轮新增) | `drl-manager/probe_forecast_sensitivity.py` |
| 探针结果 | `local_eval_rt/probe/{oracle_s1,oracle_s2,nofc_s2}.json`,`summary.txt` |
| v3 argmax 判决 | `local_eval_rt/v3_final.txt`,逐格日志 `local_eval_rt/final_v3_*.log` |
| 训练期物理量 | `drl-manager/logs/v3_{oracle,nofc}_s{1,2}/monitor_worker*.csv` |
| 场景预检查门 | `drl-manager/preflight_scenario.py`(13 项) |
| 奖励定价 | `cloudsimplus-gateway/.../multidc/MultiDatacenterSimulationCore.java` L~770-835 |
| 绿电取数 | `.../energy/GreenEnergyProvider.java`,`.../multidc/DatacenterInstance.java` L127-148 |
| 观测空间 | `drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py` L544-650 |
| 预报特征定义 | `drl-manager/src/prediction/timecap_godeye_provider.py` L9-18 |
| 实验配置 | `config_C.yml`,键 `experiment_v3_oracle` / `experiment_v3_noforecast` |
