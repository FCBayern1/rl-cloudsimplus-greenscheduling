# 交接文档 —— 2026-08-30

给新会话。**读完这份就能接手。** 结论、未决项、代码资产、计划、以及**哪些数字不可信**。

---

# 0. 一句话现状

论文的中心主张(EU-CRD 收敛预报污染的伤害)**已被自己的注册评测否定**;
而"预报在这类考场上到底有没有可兑现的价值"**至今未被有效测量**。
距 ICLR 摘要 **19 天**,全文 **26 天**。

---

# 1. 已确立的事实

## 1.1 G1 主战役(C-regime)—— 负判决

八臂训练 57h46m 全部 `rc=0`/600k;评测 72 格,协议先于数据冻结。

```
条件      100(exp(d̄)−1)     95% CI              方向
Clean      +0.51%      [−0.11, +1.14]      4/4 EU-CRD 略差
Blend      −5.82%      [−9.46, −2.03]      0/4  ← EU-CRD 赢
Shuffle    +4.91%      [−1.17, +11.37]     4/4  ← 主端点,方向反
```

- 留一法 +2.30% 到 +6.10%,**从不转负**
- 污染增量:Vanilla **+41.7%** vs EU-CRD **+48.6%**(`Δ_corr` 方向同样相反)
- SLA 共主端点:24/24 格 `ontime_mi_share = 1.0000`,平局,McNemar 无不一致配对
- **按冻结阶梯这是 STOP**。不扩十二对。

## 1.2 两个跨条件的稳健事实

- **碳的输赢由路由的加权棕电强度决定,5/5 完全同号**(none/blend/anti/shuffle/noise)
- **EU-CRD 在每个污染条件下对预报反应都更强**:blend 1.13× / anti 1.19× / noise 1.25× / shuffle 1.37×
  —— 与"降低对预报依赖"**方向相反**,四条件一致

## 1.3 T1:预报含信息,特征不冗余

- 盲态可见变量预测 forecast 特征的 blocked-OOF R² **≤ 0.27**(`long_mean` 为负)
- `ΔR²(blind+forecast → 未来绿电)`:H=1 时 0.0025,**H=12 时 0.091,H=24 时 0.215**,三 DC 一致

## 1.4 T2:兑现价值为负

No-Forecast(3 种子,101/102/103)vs matched Vanilla,clean 条件三窗池化:

```
几何均值 −5.74%    方向 1/3    原始 d_i [−0.1514, +0.1209, −0.1467]
扰动不变性哨兵:shuffle 与 clean 逐位相同(差 0.000%)
冻结预测 |差距|<5% —— 实测 5.74%,未中
```

## 1.5 TB12(3060)—— 唯一的正例,但不可用

- **物理层认证 −17.51%**(T114+115 六门,中位 −21.76%,方向 42/60,CI [−28.12%, −8.71%])
- **RL 四轮全败**(v1 奖励漏洞 / v2 defer 坍缩 / v3 全 route 停滞 / BC 探针 FAIL)
- 窗长特征线已判死(`cd62750`:绿窗比作业短 6 倍,"装不装得下"永远是否)
- **单 DC ⇒ EU-CRD 的 `Δr` 恒为零,机制退化,不适用**
- 构造:5 作业 × 4 小时,slack 15.67h(≈3.9×),1 DC,8 主机,
  `simulation_timestep=600`,episode 48h,REAL_TIME + STEP 插值

## 1.6 gwo1 的 4.13×(3060,单点,未独立复现)

离线理想 −12.24% vs 仿真实测 −2.04%,校准因子 **6.01×** = 1.31×(空转固定碳)× 4.13×(门开火碳不动)。
四个假设已排除(门不开火/容量/backlog cap/观测裁剪)。
**Codex 裁定:这是旧考场单点疑点,不是仿真器普遍属性(TB12 已证伪外推)。**

## 1.7 p* 恒等式(gwo1 §4.1)—— 解释前十考场

```
p* = (L_expire − C_brown)/(L_expire − C_green)
backstop 兜底 ⟹ L_expire = C_brown ⟹ p* = 0 ⟹ 永远等
```
**"等待免费"⟹ 最优盲策略在两端,不需要预报。** 实测 p̂ 83% 落在 [0, 0.05)。

---

# 2. 已被数据排除的解释

| 假设 | 推翻它的证据 |
|---|---|
| 预报没信息 | T1:ΔR²=0.215 |
| 特征是当前绿电的重述 | T1:冗余度 R² ≤0.27 |
| 作业太短、承诺窗口≈0 | 作业 37 步(per-PE 语义),slack 中位 6481 步 |
| deadline 太紧 | 完成率恒 100%,`deadline_forced_count=0` |
| 奖励通道腐坏(TB12 那种) | 五臂物理碳与归一化惩罚 1:1 同降 |
| 污染没生效 | Shuffle 碳比 Clean 高 42–49% |
| 构建污染 | 冻结 jar SHA 校验,`FIXED` 归一化生效,封顶零命中 |
| 臂标签颠倒 | eucrd 四臂 `crd.enabled=True`、`ρ_routing` 0.86–0.91 |
| 机制失活 | `crd_dr` 75/75 非零,`c_t` 0.30–0.65,ρ 有真实离散 |

---

# 3. ⚠️ 不可信的数字 —— 新会话必读

我在 8-27 到 8-30 之间**八次假设/断言被数据推翻**,其中多次是我自己"更正"错方向。以下数字**不得引用**:

| 数字 | 为什么不可信 |
|---|---|
| **"C-regime 有 21% 离线余量"** | 模型未对齐:只用 3 个 DC、无时区、容量口径错、静态是拟合常数、runtime 除了 pes(错) |
| **"批式约束保留 20.55%"** | 我的"批式"是"整批同一 DC",不是真实的 128 槽 MultiDiscrete。**不能据此排除动作空间瓶颈** |
| **"摘要特征吃掉一半余量"** | V3.2 观测里有 bins 和逐作业覆盖率;但 **C-regime 未启用**(`obs_v32_job_forecast` 缺省 False)。结论方向存疑 |
| **单作业 13.6pp / 40 并发 34.8pp** | 诊断值,未与仿真器对齐 |
| "6.01× 是通用折损系数" | **错**。TB12 在真实仿真器拿到 −17.51% |
| "十二个考场造不出预报价值" | **错**。TB12 是反例 |
| "作业时长 18.5 步" | **错**。CloudSim `length` 是 per-PE MI,应为 **37 步** |
| "拉伸风电时间"方向 | 错。TB12 是作业**大于**风窗才有效 |

**我三次把盲态基线写弱**(随机起跑 / 单 DC 特征集 / 不看容量),每次都让间隙虚高。这是最需要警惕的系统性错误。

---

# 4. 未决的核心问题

**C-regime 的物理 headroom 是多少?**

至今**没有运行过一个通过语义审计的、容量感知的、曲线级时空 oracle**。两次尝试都是臂自身实现失败:

- 旧 `green_forecast`:不做容量约束 → 堆积,完成率 67%,用 350 Wh 绿电浪费 7035
- 我的 `green_forecast_capacity`:按 level 统计量排序,**无风 DC 的中性默认值进了排序** → 93% 作业流向 DC3/DC4(棕 0.75/0.92)

准确定性(Codex):

```
预报是否含增量信息   C-regime 聚合层成立,逐作业决策层未证
物理调度是否能兑现   TB12 已认证 −17.51%;C-regime 【未被有效测量】
RL 是否能学会兑现    两个考场均未兑现,负结果成立
```

---

# 5. 正在跑的东西

**曲线级 oracle 单窗口 smoke**(low, k=19),`g1/run_oracle_smoke.sh`

```
盲态 green_queue_balanced   carbon 0.129579   完成 100%   ✓ 已出
oracle curve_oracle          运行中
```

**止损线(Codex 冻结)**:
- oracle 相对最强盲池化改善 **<5%** 或完成合同失败 → **停 C-regime 特征线**,转 GWO1 取消率审计
- **≥5% 且 ≥2/3 窗口同向** → 才进 TimeCAP clean/shuffle/anti 零训练特征门,**仍不烧 RL**

---

# 6. 代码资产(本轮新增)

| 文件 | 作用 |
|---|---|
| `src/baselines/global_schedulers.py` → `curve_oracle` | 曲线级时空 oracle,按 Codex 冻结语义(见 §7) |
| 同上 → `green_forecast_capacity` | 硬容量版,**已知有缺陷**(无风 DC 排序问题),保留作对照 |
| `src/baselines/evaluate.py` → `AUDIT_TRACE` | export-only 逐步逐 DC 仪表,**已过逐位恒等哨兵**(碳 0.129579 相同) |
| `drl-manager/tb13_certify.py` | 逐作业容量约束离线模型,**未对齐,只作诊断** |
| `drl-manager/tb13_search.py` | 涡轮组合 × divisor 搜索(流体上界) |
| `drl-manager/scenario_screen.py` | 考场普查(可调度占比/绿电倍数/粒度) |
| `drl-manager/t1_feature_dump.py` / `t1_analysis.py` | T1 特征冗余度与条件价值 |
| `drl-manager/check_submission_ready.py` | 投稿预检,6 道门 |
| `drl-manager/make_control_arms.py` + `tests/test_control_arms.py` | 对照臂程序化派生 + 33 个守卫 |
| `g1/run_g1_gate.sh` / `run_g1_eval.sh` / `run_t2.sh` / `run_audit.sh` / `run_oracle_smoke.sh` | 各战役 runner,均硬校验冻结 jar SHA |

**冻结 jar**:`/home/joshua/frozen/g1_gateway/lib`,SHA `aba6f0ed…`,只读。

---

# 7. Codex 冻结的 oracle 语义(必须逐条满足)

```
J(i,d,s) = Σ_τ [ c_g·min(P_i, G_res) + c_b,d·(P_i − G_res)⁺ ] Δt

runtime      CloudSim per-PE 语义:length / mips,【不除 pes】
容量         按作业 PES 扣减
G_res        扣静态功率、已运行负载、已承诺作业
无风 DC      G=0 + 真实棕碳因子参与,【不用中性 forecast sentinel】
未来         只知风电,不偷看未来到达
决策         最优时刻是现在则路由,否则 defer
对手         同一可行性模型下的【最强联合因果盲】,不是"立刻送当下最绿 DC"
```

---

# 8. 论文计划(B 路线)

## 8.1 主张

> 在一个预报确实携带信息、承诺窗口够长、信用传得到的考场上,一个训练良好的策略从预报里兑现不出正收益;
> 而一个专门用来分配预报信用的方法,在四种污染条件下**一致地放大**了策略对预报通道的响应,与设计意图相反。

## 8.2 措辞约束(Codex 硬性)

**不得写**"信用重加权**必然**放大依赖"——除非消融独立证明因果(消融未跑)。

**只能写**:

> In this registered evaluation, EU-CRD exhibited 1.13–1.37x larger forecast-induced
> policy shifts across all four corruptions, contrary to its intended robustness mechanism.

## 8.3 论文状态

**文字和结构基本定稿,缺的是数字。** 已完成的改动:

- Figure 1 XML 已修(`R_forecast` 加 `C_max` 归一 + 绝对值,与 Eq.2 一致)—— **等作者在 drawio 重新导出 PNG**
- 导师 6 组 §3.2 修订(control interval / B=128 / P̂_i 语义 / 末句只切全局通道)【已全部落地并核验】
- Limitations 已写好但**按作者要求撤出**,存 `reports/LIMITATIONS_HELD.tex`(B 路线下需重写)
- App F 已改成**可消失结构**(`\ifFragileNone` / `\ifFragilePattern`),判据冻结于 `reports/APPF_PREREG.md`
- `\ifRiskSampling` 默认 false,隐藏采样表的风险基线行
- 红色 `[UNRESOLVED]` 已关,防护改成 `check_submission_ready.py`
- 消融阶梯加第四级 `Decomposition off = Vanilla PPO`(复用主表 matchedvan)
- 腐坏模型措辞:**Shuffle = 目标失效,Blend = 对照**;对抗引用已划范围
- 全文**一段一行**(无句间硬换行)

## 8.4 投稿预检的 4 道 BLOCK

```
App F fragility claim is graded before it renders     等定级
headline macros come from the G1 campaign             等 G1 结果
sampling appendix regenerated                         整节是坏构建的数
Figure 1 agrees with Eq. 2                            等作者重新导出 PNG  ← 唯一等人的
```

## 8.5 作者对 B 路线的判断

**"写出来也是中不了的"** —— 主端点 CI 含零、n=4、单一考场、消融没跑。
备选是 **B'(基准诊断 + 筛选判据)**,把主张从"我们的方法失败"换成
"碳感知 RL 调度基准普遍测不到它们声称在测的东西,而这可量化、可预先筛"。
**未决,需作者拍板。**

---

# 9. 第 13 考场计划(Codex 2026-08-30 裁定)

## 9.1 诊断:两小时视界是**未经审计的设计惯性**

全部 27 个配置里,**只有 TB12** 用 `simulation_timestep=600` / episode 48h;
其余全是 `1 s/步 × 7200 步 = 2 小时`。要在 2 小时里塞进足够作业量,作业就必须短 ⟹ 落入流体区。

```
                    作业   DC  时长h   平均在服负载  占视界%  slack/run  集时h  s/步
tb12                   5   1  4.000       0.4      8.33       3.9   48.0   600
C-regime            8000   5  0.010      42.2      0.52     172.4    2.0     1
v2026(号称AI集群)   2000   8  0.015      48.3      0.76      21.8    2.0     1
gwo1                1200   8  0.135     105.6      6.73       1.6    2.0     1
```

**注意**:表里的"并发"应正名为**平均在服负载** `L = Σr_i / H`,它不是实测的同时可决策作业数。
这是**强共线性证据,不是单变量因果证明**。

## 9.2 起始规格(Codex 建议)

```
episode        48 小时
控制间隔       300 或 600 秒
作业数         30–50,不可分割、不可跨 DC
runtime        2–8 小时
slack          4–24 小时(不超 TimeCAP 的 24h 有效视野)
DC             3 个绿 DC + 可选棕电兜底 DC
平均在服负载   L ≈ 2–6(不必压到 TB12 的 0.4)
单作业占单站容量  10–30%
风电           相关但不完全同步的【真实 SDWPF】,按相关性选,不人工造获胜轨迹
```

## 9.3 放大 timestep 的四个工程风险(必须一起修)

1. **到达时间量化与提前可见** —— `GlobalBroker` 一次加载 [t, t+Δt];Δt=600 时作业可能提前 10 分钟进队列。
   新 trace 必须把到达对齐决策网格,或修成只在真实到达后可见
2. **按步超参失去物理语义** —— gamma、GAE λ、GTrXL `mem_len`、forecast refresh、defer cost、EMA、backstop margin
   **必须按物理时间重新注册**,不能照搬 1 秒配置
3. **决策样本稀疏** —— 50 作业分布在 288 步里,大量 transition 没有有效路由动作。
   需预先审计每 episode 的有效动作数
4. **600 秒动作量化** —— 必须证明 latest-start 至少保留一个量化余量,并报告最多 10 分钟的释放误差

## 9.4 归因干净 vs EU-CRD 可用:**不冲突**

只要比较双方**拥有完全相同的协调能力**,唯一差异是未来信息:

```
V_forecast = C(最强联合因果盲) − C(同动作空间的未来信息策略)
```

建议二维分解:

```
                     无未来信息   有未来信息
弱/固定空间路由         C00          C01
联合时空路由            C10          C11

信息价值 = C10 − C11      协调价值 = C00 − C10
交互     = (C00−C01) − (C10−C11)
```

**不要**把各 DC 风电做成逐位相同 —— 那会连预测的空间作用一起杀死。
目标结构:边际分布相近 + 较强共同天气分量(同时枯竭 ⟹ 等待有价值)+ 有限站点残差(⟹ 去哪有价值)。

## 9.5 不能叫"真实 AI 训练集群"

现有 Cloudlet 仍是单 VM、小 PES、几十瓦。要作该主张还需:多 GPU/gang scheduling、
作业级不可分割资源预留、更可信的功率模型、真实的 runtime/资源宽度/deadline 分布。
**现阶段只能叫 "AI-like 长时批作业"。**

## 9.6 执行顺序(不先训练)

1. 校正表中"并发"命名,补实测队列并发、单作业容量占比
2. 审计 600 秒步长的到达泄漏和全部按步超参
3. 冻结长时、多 DC、不可分割 workload
4. 用最强联合因果盲 + 正确 curve-oracle 做离线及单窗口仿真预检
5. **oracle 相对盲至少过 5%**,再跑多窗口认证
6. **物理门通过后**才接 Vanilla / EU-CRD

---

# 10. 写作要求(作者偏好,硬性)

- **LaTeX 一段一行** —— 句子之间不硬换行,只在分段处换行(硬换行让 git diff 整段标红)
- **不用 `\paragraph`**,`\emph` 少用,**caption 不写死内容**
- **不用冒号、分号** —— 拆成两句;`$^{\dagger}$:` 脚注标记和参数清单例外
- 英文 prose:**少用 "we"**(用 this work / 被动),去 AI 腔(em-dash、戏剧短句、公式化收尾、自指小尾巴)
- **commit 单行、无正文、无 Co-Authored-By trailer**
- **不用任务列表**(TaskCreate/TaskUpdate),多步工作记 memory 文件
- 不打包 zip,**直接更新 tex 文件**

---

# 11. 工作纪律(八荣八耻,已多次救场)

- **查档求证**:不引用未核实的数字;代码断言要读代码
- **对齐需求**:不擅自扩大范围
- **请示规则**:范围决定交作者/Codex,不自行推进
- **复用存量**:先查有没有现成的
- **完备测例**:新代码配测试;仪表要过**逐位恒等哨兵**
- **恪守规范**:预注册先于数据;**不改判据、不重跑、不换解释再宣称通过**
- **坦诚存疑**:上界不说成测量值,诊断不写成结果
- **分步迭代**:先便宜的判别实验,再烧机时

**本轮最痛的教训**:盲态基线写弱会让间隙虚高,我犯了三次。任何"全知赢很多"的结果,
**先怀疑盲态是不是写弱了**。

---

# 12. 关键文件索引

```
reports/G1_FREEZE_MANIFEST.md            G1 冻结清单(commit/jar SHA/种子表/评测常量)
reports/G1_GATE_VERDICT.md               G1 判决
reports/G1_RANKING_PROBE_VERDICT.md      排序假设被推翻
reports/T1_VERDICT.md                    T1 判决(信息在、特征不冗余)
reports/T1_T2_SPEC_FROZEN.md             T1/T2 口径 + B 路线措辞约束
reports/APPF_PREREG.md                   App F 定级判据(冻结)
reports/NUMBERS_MANIFEST.md              论文 25 项数字的污染源与处置
reports/DEFECT_FIX_PLAN_2026-08-23.md    缺陷修复清单
reports/OVERNIGHT_2026-08-28_...md       ⚠️ 21% 那份,标为【待对齐诊断】不可引用
reports/TB13_SCREEN_SPEC_FROZEN.md       TB13 筛选规格(Codex 修订版)
reports/WORKORDER_3060_DEMAND_AUDIT.md   需求—绿电—碳 审计工单(3060 down,5080 接手)
reports/tb12/*.md                        TB12 全线(认证/RL 四轮/表示审计/窗长探针)
reports/GWO1_VERDICT_AND_DIAGNOSIS.md    ⭐ 6.01× 分解 + p* 恒等式,解释前十考场
reports/PHASE_SCAN_STAGE1.md             相变扫描,首次有格过全部离线门
paper_latest/iclr2027_conference.tex     论文(已进版本管理)
paper_latest/figs/new_EUCRD_drawio_nature.xml   Figure 1 源(已修公式,待重新导出 PNG)
```

## 机器状态

```
5080   曲线 oracle smoke 运行中;冻结 jar 只读
3060   DOWN
```
