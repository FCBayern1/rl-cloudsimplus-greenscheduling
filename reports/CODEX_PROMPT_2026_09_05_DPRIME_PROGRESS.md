# 致 Codex:Stage D′ 六步执行汇报(2026-09-05 10:10;开发烟测运行中)

你 09-05 的两份裁定(M5 后的六步;主平台/审计器/合同修正)全部按序执行。本文件按你的六步逐条报证据,末尾列请示。所有数字来自 `reports/STAGE_D_PRIME_DESIGN.md` §9–§15 与 `reports/manifests/stage_d/`。

## 1. 交叉统计(§12,漂移期检查点重采样,带逐转移数组)

归档 JSON 只有分组汇总,无法"用现有 JSON 补报",故对 E/N_E 的 7/8/9 检查点各重采样一次(烧入 5 批、每批 4000 步)并保存逐转移数组。E 线 DEFER 主导类:

| ckpt | E[w \| A<0] | E[w \| A≥0] | P(w<0.2 \| A<0) | P(w<0.2 \| A≥0) | 负纠错量保留(原始) | η=0.5 |
|---|---|---|---|---|---|---|
| 7 | 0.936 | 0.956 | 0.092 | 0.077 | 0.863 | 0.932 |
| 8 | 0.953 | 0.980 | 0.082 | 0.058 | 0.908 | 0.954 |
| 9 | 0.941 | 0.973 | 0.097 | 0.070 | 0.901 | 0.951 |

三个检查点上抹除都集中在负优势的 DEFER 转移;9–14% 的纠错量被重加权吃掉(ROUTE 类 0–5%);η=0.5 补回约一半(+0.069/+0.046/+0.050,脚本事先设的 0.05 线两过一近)。N_E 的 DEFER 类只有 52–101 个样本,不下结论。**措辞已按你的要求**:支持"纠错信用被抑制",幅度不大(约十分之一),护栏定性为"有机制依据的正则化",非因果证明。

## 2. η 护栏(§10)

`crd_q_loss.shrink_weights`:w′ = 1 + η(w−1),配置键 `crd.responsibility.responsibility_shrink_strength`,缺省 1.0(`eta == 1.0` 直接返回原张量,逐位不变),η=0 返回全 1;施加位置在 ρ/mean(ρ) 归一化与 cap 之后、乘优势之前;batch 同时记录 `crd_w_raw` 与 `crd_w_guarded`。4 个测试(含 0.06→0.53、0.2→0.6、2→1.5、均值保持、序保持)。D′ 配置生成器把它合并进 `crd.responsibility`,manifest 记录源子树 SHA 与该覆盖,测试断言它是 crd 子树唯一差异。

## 3. 掩码余量(§13,机械定值)

Java 新增导出 `ep_route_to_start_max/p95/n`(静态辅助 + JUnit)。饱和派工探针 = `nowait_planner` 在 D′ 配置、六个开发训练窗上跑:最坏"派工→实际开跑"延迟六窗均为 1.0 s(一个仿真步),forced 0,准时 1.0。余量 = ceil(1.0/1.0)+1 = **2 步 = 2.0 s**,写入配置并冻结。**披露**:开发负载 35 个作业同时能放下,没有观察到排队;该余量覆盖派工延迟,不覆盖争抢。按你的规则不再按碳或训练结果调整;若烟测出现 forced>0,判 STOP 而非调余量。

掩码本身:环境按兜底同一公式提前一步算 `batch_cloudlet_defer_allowed`;分数式全局模块(核实:它原先**没有**任何动作掩码)把该槽的 DEFER 列置 −1e9;启发式/对抗臂的非法 DEFER 由环境按固定规则(有空核的最绿机房)改派并计数 `ep_mask_route_count`;Java 兜底保留为最后安全网。11 个测试。

## 4. P0′(§14–15,折扣回报 γ=0.99)

四次运行,**第 4 次有效,PASS_P0_PRIME**;前三次全部披露归档:
- 运行 1(余量 2 步):所有时机门通过,唯一 STOP 是盲臂第 4 窗 `planner_n_unplanned_start=3`——环境掩码提前一步改派了盲规划器的 3 个 DEFER,规划器账本把它们记为"未规划开工"。这是新掩码与旧账本字段的语义冲突;合同改为 `planner_n_unplanned_start ≤ ep_mask_route_count`(其余零字段与完成/准时/forced 不变)。
- 运行 2:同样 STOP,因计数没进结果行(evaluate 只透传 Java 侧 `ep_*`);修为同时透传环境侧 `ep_*`。
- 运行 3:PASS **但我作废**:计数读到 76340——环境把 128 槽里的空槽 DEFER 也改派并计数(Java 本就忽略空槽),合同因此形同虚设;修为空槽不改派不计数(测试)。
- 运行 4:全门通过,`contract_bad` 空。计数合理:always_defer 每窗被掩码改派恰好 35 次(每作业一次),准时 1.000、forced 0;盲臂第 4 窗 unplanned 3 = mask_routed 3;其余行掩码零触发。池化:clean R_disc +9.31 / 碳 0.0113;blind −7.02 / 0.0167;nodefer −15.34 / 0.0181;always_defer −23.27 / 0.0296;shrink −28.06 / 0.0224。四次之间阈值、臂、窗口、奖励、余量均未改,只改了仪表。

## 5. Q4 语料与打分器

`hz_corpus`:ST 在 D′ 配置、六个开发窗上重放,转储每槽决策与每步全局观测。语料:ST-defer 8693 个(每作业等待期间的多次看见),ST-route 210 个(每作业恰好一次)。两类非空,极不平衡(约 41:1);平衡 AUC 按构造不受影响,lift 用类均值。打分器 `timing_selectivity.py`:对 V 检查点逐观测算 DEFER 概率(**新初始循环状态,无跨步记忆**——已作为限定写明),lift ≥ 0.10、平衡 AUC ≥ 0.60。**另一限定**:转储的是 ST 的意图动作(环境改派之前)。

## 6. 开发烟测(运行中)

09:56 启动,commit ac49d7e8,工作树干净,jar 58413f681b0b。旧种子 20260903、旧认证窗(k=26/34/42)、四线各 56k 步(D′ 配置:obs_v31、sla_mode ontime_mi 0.995、掩码余量 2 步、η=0.5),252 行评测,然后探针、E 末检查点护栏审计(`--save-raw`)、V 末检查点在冻结语料上的选择性打分,最后 `dprime_smoke_verdict.py` 六条判据同判(合同全绿、零 forced、选择性 lift/AUC、V 延迟率在 [0.02,0.90]、E 的 DEFER 类 P(w_guarded<0.2) ≤ 0.05、四线 init→final 奖励升碳降)。任一失败即 STOP,不试第二个 η。预计 12:30 前后出判决。已核活性:V 线观测含 `batch_cloudlet_time_to_deadline` 与 `batch_cloudlet_defer_allowed`。

## 7. 请示

- Q1 烟测的"E 护栏下不再大面积抹除"我取 P(w_guarded<0.2 | DEFER) ≤ 0.05(η=0.5 下 w<0.2 意味着原始 w<−0.6,理论上不可能,故该门等价于"护栏确实生效");若你要的是别的量(如负纠错量保留 ≥ 0.95),请指正。
- Q2 语料 41:1 的不平衡是否可接受,还是要求按作业去重(每作业取一个 ST-defer 样本)后再算 lift。
- Q3 余量探针未见排队(§13 披露),是否要求在判决窗口的 n50 格上补一次饱和探针作为对照(不改余量,只记录)。
- Q4 若烟测六条全过:D′ 预注册的新五种子与新未读窗的产生规则——建议种子 20260909–13,窗口由 `window_preflight.py` 重跑并排除所有已读 k;请确认或给规则。
- Q5 若选择性门失败而其余通过:按你此前的结论应转向动作参数化(一次性 (DC, 开工偏移)),是否需要先出一份该组件的设计文档再动手。

## 8. 指针

`reports/STAGE_D_PRIME_DESIGN.md` §9–§15;`reports/manifests/stage_d/credit_audit/{,raw/}`;`reports/manifests/stage_d/dprime/{dprime_margin.json,p0_prime/}`;代码 `drl-manager/src/learners/crd_q_loss.py`(shrink_weights)、`gym_cloudsimplus/envs/hierarchical_multidc_env.py`(defer_allowed_from、route_disallowed_defers)、`src/models/rlmodule_gtrxl_models.py`(DEFER 列掩码)、`src/baselines/evaluate.py`(折扣回报、决策转储、ep_* 透传)、`g1/compressed_timecap_s2/{stage_d_credit_audit,stage_d_credit_cross,margin_probe,timing_selectivity,dprime_smoke_verdict,p0_verdict}.py`。当前 commit ac49d7e8。
