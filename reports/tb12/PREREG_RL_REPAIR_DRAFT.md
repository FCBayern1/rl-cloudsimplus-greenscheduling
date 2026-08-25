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
| ontime_mi SLA 分支 + 静态 `ontimeMiSlaCost` | MultiDatacenterSimulationCore | SlaOntimeMiCostTest(7 例) |
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
