# V3.2 预注册(2026-08-14 定稿;判据跑前锁死,门柱不挪)

规格:`V32_FORECAST_REVIVAL_PLAN.md` + §11 复核修订。继承 `V31_PREREG.md` 的全部
协议纪律(argmax、10 局、`--local drain`、超时筛、iso-completion ≥99.5%、
噪声底 13%、坑单⑫化石测试豁免)。V3.1 的修订 A1/A2(可判性=forecast ≥ 10×null)继承。

> **修订 A3(08-14 16:40,发生在任何 v3_2 训练/判决之前;第七轮复审,Codex)**:
> ①Gate 2 的温度差改由**作业对齐通道**测量(`job_temporal_delta`:forecast_gain 0.6/
> time_to_best 0.1 vs gain 0/time 1.0)——factorized gate 对 `dc_future_*` 零梯度是
> **设计目标**(解耦测试断言),旧通道会把健康模型判死。判据语义与阈值(≥+0.05)不变。
> ②预报 horizon 训前冻结:3600 步(=urgency 窗,覆盖 p95 slack 2918s;发现的默认 120 步
> 对 1029s 峰周期是瞎的),bins=16,两臂对称。③探针加 module.eval()(dropout 0.1 此前
> 在探针前向中处于激活态——历史读数第三位抖动的来源;旧 checkpoint 读数语义不变,已回归)。
> ④接线双白名单修复 + fail-fast + 集成测试(config 说 true 而模型没建 gate 的静默丢弃
> 已不可能复现)。

> **修订 A4(08-14 16:50,仍在任何 v3_2 训练样本之前;第八轮复审)**:
> ①**horizon 改由离线覆盖率证据冻结**(`drl-manager/scan_v32_horizon.py`,8 个闭卷 offset
> × 全 trace):120s 只覆盖教师可达收益的 **19.8%**、决策一致率 60.5%;3000/3600s 达
> 92.7–93.8% 覆盖、98.6–98.8% 一致(两者数值全同,slack p95=2918s 封顶)。
> **冻结 horizon=3600 / bins=20**(20 bins 比 16 多 1.1pp 覆盖)。
> ②**Gate 2 升级为多条件**(单一合成 Δ 不构成判决):`job_temporal.delta ≥ 0.05` ∧
> P(defer) 对 forecast_gain 单调(≥75% 相邻对) ∧ 对 time_to_deadline 单调(≥75%) ∧
> forecast TV ≥ 10×null。真实 rollout 同号在聚合器就绪后加入,当前显式标注
> NOT-AVAILABLE 而非静默跳过。阈值与语义均在训练前锁定。

## 实验对

`experiment_v3_2_oracle` vs `experiment_v3_2_noforecast`(待建:v3_1 模板 + V3.2 开关全开
+ μ=3.524/σ=2.512/σ_spatial=1.146 抄自标定件 v3)。开训前必须过
`preflight_scenario.py <o> <n> --v32-cert`,且 runner 断言模型配置里
`factorized_temporal_gate: true`(该键在 model config 块,preflight 管不到)。

## Gate 阶梯(唯一路径,不过即止损)

| Gate | 判据 | 状态 |
|---|---|---|
| **0 完整性** | seed 进 `result.json.config.seed`(≠null);两臂同机同 offset 日程;preflight --v32-cert 全过 | seed 接线已实现+测试 ✓,待冒烟实证 |
| **1 直连边** | 旧模型:raw defer 响应比 route 低 ≥2 个数量级(**已证:2260–3400×;冒烟 ck 30.6×——注意 100k 欠训模型比值缩小,判据以成熟 ck 为准**);V3.2 gate:`∂temporal/∂forecast_gain ≠ 0` 且 `∂logp_defer/∂dc_* ≡ 0`(**已证:JUnit+pytest**) | **双边闭合** ✓ |
| **2 冒烟(100k oracle s1)** | ①温度差 `P(defer\|将至)−P(defer\|将去) ≥ +0.05`(比 V3.1 的"仅正号"硬 6–25 倍,训前冻结);②合成探针与真实 rollout 同号(rollout 仪表已落地);③P(defer) 对 forecast_gain 单调增、对紧迫度单调减(分桶仪表);④判可性 forecast ≥ 10×null | 待跑 |
| **3 双种子 300k** | 两种子全部后段 ck 同号;defer 条件 TD residual ≤ 3× route;无 all-defer/backstop 主导/完成率塌缩 | 待跑 |
| **4 配对物理评测(600k)** | 完成率双方 ≥99.5%;oracle carbon/MI 低于盲臂最好合格格 **>13%**;defer 差与 forecast_gain/slack 相关(仪表),非无差别多 defer | 待跑 |
| **5 机制** | anti-forecast 使 oracle 碳恶化 ≥10% 且完成率合同仍满足 | P4 继承,Gate 4 过后跑 |

**判读树(锁定)**:Gate 2 不过 → **不调权重不延长训练**,查实现;实现无误仍不过 →
V3.2B 蒸馏(老师=slack-aware θ=0.5,−29% @ 100%;方法级对比与消融分表,信息集如实声明)。
Gate 3 的 TD-residual 异常且指向噪声 → 启用 §6b 备胎(逐决策候选均值中心化已在
spatial 项实现,可加大 w_s / 或将 level 项降权)。Gate 4 过 → Gate 5;全过 → 写进论文
premise + 减法消融(只对进声称的组件)。

## 与 V3.1 参照波的关系

参照波(间接通路,600k,4 臂)是 V3.2 的**文档化对照**:V3.2 的盲臂若与参照波盲臂
表现一致(同配方仅缺 v32 开关的部分),互为 sanity;oracle 侧的差 = 直连边+作业对齐
特征+双尺度的联合贡献(打包归因,存在性证明优先,减法消融事后)。

## 资源与时限

冒烟 ~25min;300k×2 ≈ 2.3h;600k×4 ≈ 9h + 评测(**final-ck-only,4 格 ≈ 3.7h**,
沿用主表先例 c0d52bf"配对分析不做选点");GPU 机 08-15 后可分流 s2 对(配对同机)。
到摘要(09-18)还有 34 天;V3.2 是三轮预算的第二轮。
