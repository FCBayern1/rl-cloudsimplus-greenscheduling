# V3.1 预注册(骨架 2026-08-13;开关决定栏明早判决后填,其余不再改)

预注册的意义:判据写在跑之前,结果出来后不许挪门柱。
背景与证据链:`docs/V3_FORECAST_DIAGNOSIS.md`;规格:`docs/V31_WORK_ORDERS.md`。

## 1. 实验对

`experiment_v3_1_oracle` vs `experiment_v3_1_noforecast`(config_C.yml,已建,
preflight 15/15 PASS,两臂 diff = {forecast_mode})。
**认证配置必须通过 `preflight_scenario.py <o> <n> --v31-cert`** 后才允许开训。

## 2. 协议(锁定)

- 训练:600k steps,6 workers,种子 {1, 2};两臂同机同种子(配对内不跨机)。
- 冒烟门(**只判 P1,分级省算力**):100k oracle s1 → 探针符号正才买 s2 的算力 →
  两种子都正 = 正式 P1 通过(run_v31_smoke.sh 自动串联)。任一种子非正 →
  temporal gate,不调权重。**P2/P3 不在冒烟门**:P3 的 iso-completion 合同对 100k
  欠训 checkpoint 无意义,P2 的盲臂 defer 率也要等盲臂在 600k 阶段重训后才可比
  (V3.1 配方对两臂都是新的,旧盲臂 checkpoint 不再是对照)。
- 600k 全量 = 2 臂 × 2 种子 ≈ 60h 本地串行;若 GPU 机 08-15 后空出,
  可迁移但**配对必须同机**(两臂同种子不许跨机)。
- 评测:argmax(DECODE_TOPK=0),10 局,`--local drain`(工单3 落地后强制;
  checkpoint 为 fixed-drain 时 evaluator 拒绝 --local rllib)。
- 筛选:只用 wall-clock 超时(全 defer 使仿真 ~20x 慢);完成率永不作筛选条件。
- 比较:iso-completion(≥99.5% 才比碳);跨臂只比物理量。

## 3. 判据(锁定)

| # | 判据 | 通过线 | 测法 |
|---|---|---|---|
| **P1** | 时间杠杆符号 | `P(defer\|绿电将至) − P(defer\|绿电将去) > 0`,**两个种子都要** | `probe_forecast_sensitivity.py`(探针含阴性对照;绝对值不采信,只采信符号与臂间差) |
| **P2** | 延迟率 | oracle defer 率 ≥ 盲臂 | 同探针 + 训练 monitor |
| **P3** | 碳 | 同完成率(≥99.5%)下 oracle carbon/MI 低于盲臂最好合格格,**幅度 > 13%**(实测噪声底上界,`memory/project_eval_noise_floor.md`) | argmax 评测 |
| **P4**(机制) | 扰动会痛 | anti-forecast 扰动使 oracle 碳变差 ≥ 10%(iso-comp) | FORECAST_PERTURB_MODE 评测;P3 过了才跑 |

判读树(锁定):
- P1✗ → **不调权重**,直接 temporal gate(§6d 步10)。
- P1✓ P3✗ → 机制修通但物理收益不足 → 看 track0 上限重估考场。
- P1✓ P3✓ P4✗ → 预报相关行为存在但可被反应式替代 → 诚实报告为部分成功。
- 全过 → 预报载重成立,进论文;再做减法消融归因(只对进声称的组件)。

## 4. 护栏(锁定)

- deadline backstop 保持开启(所有臂);
- 超时筛保持;
- **新增监控**:defer 率轨迹、forced-route 计数逐局记录;defer 率 > 50% 连续 3 局
  视为 all-defer 复发前兆,人工介入;
- 奖励真值表 JUnit(PerActionRewardSurgeryTest)对认证开关组合必须全绿,
  否则该组合禁止进训练。

## 5. 条件 critic 诊断(训练中记录,锁定)

1. defer transition 的 TD residual(与 route transition 分开统计);
2. defer 条件下的 explained variance;
3. 按 backlog/slack 分桶的 value calibration。

**λ 政策**:`gae_lambda` 保持 0.98;仅当 defer 样本 TD residual 相对 route 样本
明显异常(>3x)时才试 0.99;不上 0.999。

## 6. 开关决定(明早判决后填,其余节不动)

| 开关 | 候选 | 决定 | 依据 |
|---|---|---|---|
| per_action_completion_mode | no_offset | 待填 | 真值表 |
| defer_cost_mode | incremental_urgency | 待填 | 真值表 |
| per_action_carbon_norm | centered_zscore(μ/σ=标定件) | 待填 | 真值表(注意:scale_only+no_offset 结构性不过第①行,JUnit 已钉死) |
| fixed_local_scheduler | drain | **drain(无条件,已定)** | 结构清理;不由 drainfix 决定 |
| obs_v31_features | true | 待填(工单2 落地即开;两臂对称) | 表示能力(槽位无 slack 则策略无法表示 defer 规则) |

判决输入:track0(考场杠杆 ≥15%/<10%)、drainfix(旧判决 local 混杂复核)、
P3+sp_s2(sp 臂跨种子复现性)、真值表 JUnit(用标定 μ/σ 重跑)。

## 7. 已声明的局限(锁定)

- V3.1 是打包配方:若成功,无法归因单个修复(可接受——目标是存在性证明,
  臂间消融仍单变量);进论文的组件事后做减法消融。
- 标定 μ/σ 来自离线估计器(见 calibrate_reward_norm.py 的假设记录);
  首个 100k 冒烟后用 epCarbonRawKgSum/SampleCount 交叉核对,偏差 >2x 时重标定。
