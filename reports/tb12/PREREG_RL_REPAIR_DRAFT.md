# TB12 RL 修复实验预注册草案(待 Codex 签发 50k)

日期:2026-08-25。前置:RL_PAIR_VERDICT(负判决,维持)、P0_REWARD_ALGEBRA_AUDIT、
P1_REWARD_CALIB(含 Codex 更正)。本文冻结后不得因结果回调。

## 实验定义

- 配置块:`experiment_tb12_rl_fc_v2` / `experiment_tb12_rl_nofc_v2`
  (config_C.yml 尾部 append-only;v1 块字节未动)。与 v1 的全部差异:
  1. `carbon_normalization_fixed_max: 6.637006600838674e-03`
     (冻结自 P1 标定,推导规则 fixed_max = C_max_cal/1.5,不得回调)
  2. `sla_mode: ontime_mi`,`sla_target: 0.995`
     (c_t = max(0, 0.995 − ontime_mi_share),MI 加权,与判决合同对齐)
  3. `defer_deadline_slack_sec: 720.0`(覆盖 600s 决策量化,保底 ≥120s)
  4. global_model `gamma: 1.0`、`gae_lambda: 1.0`(288 步有限 episode,
     碳信用无衰减)
  5. 命名/输出目录 _v2
- 两臂唯一实验变量:forecast_mode(fc=full / nofc 挖空)。
- 训练风:T100+101/**2021**(csv_year 如实注册),offset_range 52262。
- cap=3.0 保留(Codex 批准,减少实现变量),条件:校准四轨 cap count=0;
  训练中**任何 cap 命中立即停止**;`global_carbon_cap_count/max_ratio`
  已由 Java 导出并记录;训练后不得修改分母。

## 实现清单(全部已完成并测试)

| 变更 | 位置 | 测试 |
|---|---|---|
| ontime_mi SLA 分支 + 静态 `ontimeMiSlaCost` | MultiDatacenterSimulationCore | SlaOntimeMiCostTest(8 例) |
| cap 监控 `global_carbon_cap_count/max_ratio`(export-only) | 同上(TOTAL+PER_MI 两路) | 行为中性,套件全绿 |
| **backstop per-PE runtime 修复**:`deadlineForceLatestStart` 原用 length/(pes×mips),CloudSim length 为 per-PE → 2-PE 作业 runtime 低估 2x,backstop 晚开火 ~1.8h(v1 always_defer/RL 坍缩臂 ontime=0 的直接原因) | PerActionRewardMath | 真实 TB12 作业回归例 + 旧错误测试改正 |
| 720s 量化覆盖(≥120s 保底) | 配置 + 数学测试 | slack720CoversControlQuantization600 |
| 年份锁 `assert_year_consistency` | tb12_reward_calib.py | pytest 4/4 |

Java 全套件绿;`./gradlew -q compileJava` 通过。
**注**:backstop 修复改变 gradlew 路径未来所有 latest_start 行为;已封存的
T114+115 判决用冻结 jar(12c30342),不受影响。v1 RL 负判决维持不重跑。

## 判据(冻结)

- **主判据**:RL_fc vs RL_nofc 物理碳池化方向(fc<nofc)+ 逐格配对
  (同机同偏移)。
- **次级强度指标**:frozen-reference gap closure = (盲−RL_fc)/(盲−clair_ref),
  仅池化 盲>clair_ref 时有定义,分母≤0 报 undefined;不称"理论上限捕获"。
- 有效性:finished=5 且 ontime_mi_share ≥ 0.995 逐格。

## 50k 烟测四门(全过才准 300k/600k;50k 不过立即止损)

1. **奖励—物理门**:固定校准 offsets 上,ck50 pooled global reward 优于 ck0
   ⇒ pooled kg 必须下降,否则 STOP;所有 episode cap count == 0。
2. **SLA 门**:fc、nofc 固定 offsets 池化 ontime_mi_share ≥ 0.995。
3. **坍缩门**:每 episode deadline_forced_count < 5;至少一作业在 backstop
   前主动 route;有效槽 defer fraction < 0.95。
4. **信息活性门**:RL_fc vs RL_nofc 产生可测行为差异即可(释放时刻分布或
   kg 逐格差非全零);50k 不宣布胜利,物理优势留给正式 Gate。

## v2 真值表结果(修复后重跑,v2 块 + 重编译 Java)

| 轨 | 物理 kg | ΣĈ | env cap count | ontime |
|---|---|---|---|---|
| greenfollow | 0.51934 | 78.25 | 0(全格) | 1.0 |
| clair(在线启发式) | 0.56372 | 84.94 | 0 | 1.0 |
| nowait | 1.11946 | 168.67 | 0 | 1.0 |
| always_defer | 1.23340 | 185.84 | 0 | **1.0** |

**三条签发条件全过**:奖励-物理同序 PASS;cap rate=0(Python 侧计算 +
Java 侧 `global_carbon_cap_count` 读数双确认全零);always_defer 不夺冠
(仍居末位;backstop 修复后其碳从 1.421 降至 1.233、ontime 0.0→1.0——
它现在是合法的"latest-start 策略",判据只要求不夺冠)。
per-step 分布与 v1 标定逐位一致(p99=7.906e-3, max=9.956e-3),
fixed_max 推导不变。

哈希:tb12_reward_calib.json sha256[:16]=a8f7fb2f26cbd128;
tb12_reward_calib_v2.json sha256[:16]=4cc8fba40390a27b。
命令:`EVAL_CONFIG_PATH=$R/config_C.yml TB12_REQUIRE_FROZEN_JAR=0
.venv/bin/python tb12_reward_calib.py --experiment experiment_tb12_rl_fc_v2
--json-out ../local_eval_rt/audit/tb12_reward_calib_v2.json`(GATEWAY_LIBS unset)。

## 执行接线补齐(Codex 硬阻塞 8 项,2026-08-25 第二轮)

1. **cap 硬止损已接通**:`GreenEnergyLoggerCallback.check_carbon_cap` 消费
   Java 的 `global_carbon_cap_count/max_ratio`,写入训练 metrics;块开关
   `carbon_cap_hard_stop: true`(v2/v2s50k)下任一 cap 命中抛 RuntimeError
   立即停训。单测覆盖(启用抛/未启用只读/缺键容错)。
2. **50k smoke 块已建**:`experiment_tb12_rl_{fc,nofc}_v2s50k`(append-only,
   与 _v2 差分仅 total_timesteps 50000 / checkpoint_freq 50000 / 身份字段,
   由守卫测试锁死)。
3. **ck0 训练前固化**:`save_initial_checkpoint: true` 下
   `on_algorithm_init` 保存 `checkpoint_ck0`(save_to_path,失败即抛,
   ck0 为强制项)。
4. **四门机械执行器**:`tb12_smoke_gate.py` —— ck0/ck50 × fc/nofc × 6 校准
   offset,逐集采集 kg/reward/ontime/cap/forced/defer 比例/主动释放,
   四门纯函数判定 + 单测 8 例(含 v1 事故指纹回归:reward↑ 而 kg 不降 → STOP)。
5. **配置守卫**:`tests/test_tb12_v2_config_guard.py` 6 例 —— v1→v2 差分
   精确等于白名单(五项修复+两开关+身份),fc↔nofc 单变量,v2→smoke 仅步数,
   关键值 pin 死。
6. **修复 jar 冻结**:`calib/tb12_repair_jar_manifest.json`(source_commit /
   jar_sha256 / config_sha256),smoke gate 启动强制核验;训练由
   `scripts/tb12_smoke_run.sh` 设 GATEWAY_LIBS 并核验后启动。
7. SLA 测试计数已更正:8 例。
8. **实验阶梯冻结(读取任何 50k 数据前)**:
   - **50k**:只做健康四门,不宣布效果;T100+101/2021 六校准 offset。
   - **300k**:诊断门 —— fc 相对 nofc 碳方向有利、SLA 合同通过、cap=0、
     无全 defer 坍缩。
   - **600k**:固定最终 checkpoint(ck 末),**不挑 ck**。
   - **正式 held-out**:预注册 **T116+117/2021**(从未分析过;
     T110–T115 已反复使用,退役),只在 600k 完成后首次读取。
   - **正式主门**:RL_fc 相对 RL_nofc 聚合碳至少 **−5%**;ontime 全部合格;
     cap=0;严格方向 **≥42/60**。次级强度指标:frozen-reference gap
     closure ≥30%(仅池化 盲>clair_ref 时有定义)。
   - **s2**:s1 正式门通过才启动预注册复刻;失败则保留负结果并停止。

## v3 修订(Codex 四项,2026-08-26,读取任何 50k 数据前冻结)

① smoke runner/gate 全部指向 _v3s50k,防回归测试锁死(文件内容级断言,
   `test_smoke_chain_references_v3s50k_not_v2`)。
② **方向门重做**:greenfollow 标签仅 3 offset 含两类(≥4/6 不可达)且不证明
   预测载重,废弃;改为 greenfollow 固定 corpus、仅取 clair↔greenfollow
   **分歧作业**、以 clair 为目标,判定 fc 的 ck0→ck50 **signed p_hold 移动**
   (target=hold ⇒ +Δp / target=route ⇒ −Δp):池化 ≥ +0.05,有效 offset
   (含 ≥1 分歧)中 ≥4 正向,有效数 <4 ⇒ undefined FAIL。实测 corpus 分歧
   分布 {4000:1, 12000:1, 20000:0, 28000:3, 36000:1, 44000:5},**有效
   offset = 5**,样本 11。均匀 defer 漂移在 signed 度量下自动抵消(单测锁)。
③ 门位阶(冻结):**50k 放行 = G1+G2+G4 + 方向门**;旧 argmax 坍缩门(G3)
   在 50k **降为诊断**只记录不拦;**300k 恢复行为硬门**(G3 原判据)。
④ 不豁免首 defer −0.5:校准 runner 量化语义修正(计划释放落当前 600s 窗内
   即路由,nowait 首次 eligible 立即 route),真值表+计费审计重跑
   **PASS 全格**(同序 greenfollow 0.51425 < clair 0.56195 < nowait 1.12363
   < always_defer 1.23340;cap 0;哨兵全过)。

### 冻结哈希(v3 执行基线)
- jar_sha256: 940078777d788d68…(source 13348ae5673d,manifest 已入库)
- config_C.yml sha256: 2f31042f5e38adba…(与 manifest 一致,未变)
- direction corpus: calib/tb12_direction_corpus_v3.npz
  sha256=4c9577eb6064c6bcc…(完整值见下行哈希记录)
- 真值表 artifact: tb12_reward_calib_v3b.json
4c9577eb6064c6bcaa6a8b21d9ca2c51b0483b5f33498054db8bb22b53053950  drl-manager/calib/tb12_direction_corpus_v3.npz
c5809237b017efa73ba199c9478d80d9cd8fed63ee0db64d36674181a29a4437  local_eval_rt/audit/tb12_reward_calib_v3b.json
