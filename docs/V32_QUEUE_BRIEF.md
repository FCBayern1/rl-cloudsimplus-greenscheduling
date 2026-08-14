# 实验队列简报(2026-08-14 傍晚;给 Codex 的同步件)

执行者:`local_eval_rt/run_v32_pipeline.sh`(单链无竞态,已接管机器)。
判决落点:`local_eval_rt/v32_pipeline.txt`(GATE 行)、`local_eval_rt/probe/*.json`。
判据权威:`docs/V32_PREREG.md`(门柱训前锁死)。

## 两套实验配置是什么

| 配置对 | 配方 | 身份 |
|---|---|---|
| `experiment_v3_1_*` | 公平性修复全集:no_offset + incremental_urgency + centered_zscore(μ=3.524, σ=2.512)+ fixed drain + obs_v31(slack/age/deferred)。**defer 头无预报直连边**(已实证:响应比 route 低 2260–3400×) | **参照系**(间接通路天花板;认证身份已注销) |
| `experiment_v3_2_*` | v3_1 全部 + **factorized temporal gate**(每槽 p_hold 直连预报,gtrxl 块 `factorized_temporal_gate: true`)+ **obs_v32_job_forecast**(forecast_gain 等作业对齐特征,盲臂 persistence 填充)+ **candidate_mean 空间项**(σ_spatial=1.146,双尺度) | **认证候选** |

两对内部:两臂只差 `forecast_mode`(preflight `--v31-cert`/`--v32-cert` 机械把门,均已 PASS)。

## 今天已完成(判决在档,勿重跑)

| 实验 | 判决 |
|---|---|
| drainfix(6 格统一 drain 重评) | 旧 v3 判决反转:盲臂完成率赤字=co-learned local 混杂;预报什么都没买到 |
| slack-aware oracle(θ=0.7/0.5) | **考场门 PASS:oracle-gap 21–29% @ 100% 完成**,考场无罪 |
| V3.1 300k 双种子探针 | 时间符号 7/7 翻正但幅度 +0.002~0.009(间接通路极限);P1 名义过 |
| raw-logit 直连边实证 | defer/route 响应差 3 个数量级 → V3.2 的存在理由 |

## 正在跑 / 今晚队列(按序)

| # | 实验 | 干什么 | 判据/产出 |
|---|---|---|---|
| 1 🔥 | `v31_nofc_s1` 600k(孤儿进程,~17:40 完) | 参照波盲臂 seed1,与已完成的 `v31_oracle_s1` 组成 s1 参照对 | — |
| 2 | installDist + 真值表真值门 | V3.2 Java(双尺度)进 jar。**对间换 jar、对内一致**(门控回归锁保证 v3_1 行为逐位不变) | 真值表 65 测须绿 |
| 3 | `v32_smoke_s1`:v3_2_oracle seed1 **100k** + 探针 | **GATE 2**:直连边配方能否快速学出强时间响应 | **Δ=P(defer\|将至)−P(defer\|将去) ≥ +0.05**(训前冻结,比 V3.1 硬 6 倍);FAIL→不调参不延长,跳过 Gate 3 |
| 4 | [仅 Gate2 过] `v32_g3_s1/s2`:**300k × 双种子** + 探针 | **GATE 3**(判决 ~21:00):跨种子符号一致 | 双种子 Δ>0;600k 审批留人工(TD-residual 条款过目) |
| 5 | `v31_oracle_s2` + `v31_nofc_s2` 600k(整夜) | 参照波 s2 对补齐(新 jar,对内一致) | — |
| 6 | 参照波符号曲线探针(全 ck,含 --raw-logits) | 间接通路符号形成曲线 + direct-edge 证据固化 | probe json |
| 7 | 参照波评测 **final-ck-only × 4 格**(--local drain) | **参照波 P3**(明早 ~05:40):间接通路天花板 = V3.2 边际贡献的消融锚点;V3.2 盲臂的 sanity 锚 | carbon/MI @ 完成率 |

## 明天(人工决策点)

| # | 条件 | 动作 |
|---|---|---|
| 8 | Gates 2+3 过 + TD 条款过目 | **v3_2 600k × 4** 点火(GPU 机 08-15 空出可分流 s2 对,配对必须同机)→ final-ck 评测 → **GATE 4:iso(≥99.5%)碳差 >13%** |
| 9 | Gate 4 过 | **GATE 5**:anti-forecast 扰动,oracle 碳恶化 ≥10% 且完成率合同仍满足 = "预报载重"终审 |
| — | Gate 2 或 3 败 | 按预注册:查实现(raw-logit/梯度测试复跑)→ 实现无误 → V3.2B 蒸馏(老师=slack-aware θ=0.5,方法级对比分表) |

## 给 Codex 的三个注意

1. **明早前主树只读**(参照波在跑;今天下午的教训:你的改动恰好全门控才没伤到 s1 配对);
2. 你的 `v32_gate_verdict.py` 与 rollout 仪表在 Gate 3/4 的 rollout 判读时首次实战——如有 CLI 用法差异,今晚看到 GATE 行后可以先干跑一遍你自己的判决脚本对照;
3. Gate 2 若败,第一嫌疑是**特征而非结构**(gate 结构的梯度连通已单测证明)——届时优先复核 forecast_gain 的生成数值(量纲/截断/盲臂填充),那是你的 B 单地盘。
