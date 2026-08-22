# EU-CRD 投稿处境:一页纸(2026-08-22)

给第二意见评审。截止:abstract 9-18(27天),full 9-25(34天)。

## 1. 今天发现的根因

**论文正文描述的算法,和产出论文数字的那个构建,不是同一个。**

| | 论文怎么写 | Table 1 那个 checkpoint 实际是什么 |
|---|---|---|
| Eq. 6 / §3.3 | "each signal is first divided by $s_k(t)$, a running estimate of its own typical magnitude" | `normalize_shares` **不在配置里**;代码 `cfg.get("normalize_shares", False)` → **关闭** |
| App B | "shares are scale-normalised by a running EMA (decay 0.99)" | 同上,`share_scale_decay` 也不在 |

产出 Table 1 的 `creg_eucrd_s2` 训练于 **2026-07-15**;v5 修复批次是 **07-17** 才实现的。该 checkpoint 的 params.json 里 `scale_fix / normalize_shares / infer_num_dc / anomaly_gate / mask_padding / carbon_norm / magnitude / stable_bootstrap / sigma2_norm` **全部为 None**。

按项目自己 07-17 的代码审计(4-agent audit,已记录),这意味着在该构建中:

- **BUG 3** — COMPRESSED 模式下 `predicted_wind_w` 比 `actual_wind_w` 大 **1500×**,`R_forecast`(方法名里的那个通道)是与预报误差无关的有偏常数
- **BUG 1** — `ρ_routing → 0.99`,advantage 乘子恒 ≈1,**quarantine 不运行**
- **BUG 2** — router 的 `Δr ≡ 0`,门的回退从未发生

审计当时的结论原文:*"pre-v5.1 EU-CRD results ran with R_forecast≈garbage + inert quarantine — their positives come from the gate-as-early-stabiliser + ensemble regularisation, NOT forecast attribution."*

**独立佐证**:昨天新跑的组件消融——去掉 advantage reweighting(ablW)没有可分辨变化,而这正是"quarantine 乘子恒为 1"所预测的;门开死(ablG)则 shuffle 碳 +23%,与"门只是早期稳定器"一致。

## 2. 三个候选构建的实测状态(全部同一考场:`probe_C_2xjob_dl6500.csv`, divisor 1500)

| 构建 | 日期 | v5 修复 | 种子 | 训练完成率 | argmax 部署评测 |
|---|---|---|---|---|---|
| `creg_eucrd_s2` **(论文在用)** | 07-15 | 全关 | 1 用于表 | 0.999 | 正常,clean 0.192 / shuffle 0.255 |
| `cregime_eucrdv5b_s1-3` | 07-18 | 开 | 3 | 0.999/0.996/0.999 | ❌ **崩溃:完成率 5–18%** |
| **`v3ht_knSb_s1-s4` (knSV3b)** | 07-28~08-01 | **开,且最完整**(多 `delta_r.mode=green` = BUG 2 完整修复) | **4** | 0.999 全部 | ✅ **正常,完成率 ~100%** |

v5b 的崩溃已用逐键配置比对排除"评测配错"(仅差 log 目录/worker 数/wandb),与既有记录 `project_v52_completion_collapse`(局部层 NoAssign 吸收态)签名一致。

## 3. knSV3b 已测的两个种子(argmax,1 episode/格,论文的 ckpt 选择规则)

| 臂 | Clean | Blend | Shuffle | 完成率 |
|---|---|---|---|---|
| Vanilla PPO(Table 1) | 0.184 | 0.184 | 0.269 | 99.7 |
| EU-CRD **v4**(Table 1) | 0.192 | 0.185 | 0.255 | 100 |
| **EU-CRD knSV3b(2 种子中位)** | **0.189** | **0.186** | **0.227** | ~100 |

分种子 shuffle:s1 = 0.205,s2 = 0.248(离散 21%)。

对 Vanilla 的相对量:**v4 = clean 学费 +4.3% / shuffle 优势 −5.2%;knSV3b = clean 学费 +2.7% / shuffle 优势 −15.6%。**
修复后学费减半、优势约三倍——方向与 bug 诊断预测一致。

**⚠️ 强度不足,不可直接下结论**:仅 2/4 种子、每格 1 episode、与 Vanilla 跨 campaign 比、种子间离散 21%。

## 4. 三条路线

**Route A — 把头条换成 knSV3b(推荐)**
- 需要:s3/s4 评测(~3.6h)→ 选定 ckpt 加密到 5–10 episodes(~12h)→ 同协议重评 Vanilla 对照(~4h)→ auditor 换健康 knSV3b ckpt(~2h)。合计 **~20h 机时**,加 2–3 天改写。
- 附带收益:**消融家族(ablG/ablW)本来就是 knSV3b 的单键差分**,换头条后消融第一次与主表同源,现存的"消融跨 campaign"问题自动消失。
- 风险:未评测的 2 个种子可能回归;1→5 episodes 后差距可能缩小(本考场噪声底 10–13%)。

**Route B — 保留 v4 数字,只改描述**
- 需要把 §3.3 改成描述 v4 实际跑的东西,即承认尺度归一化未启用、`R_forecast` 是有偏常数。
- **实际不可行**:等于书面描述一个已知损坏的算法。

**Route C — 保留 v4 数字,改成描述性/诊断性定位**
- 成本最低,但**方法—实测件不一致仍未解决**,是 Route B 的同一问题。

即:**A 是唯一能同时解决"描述不一致"和"证据偏弱"的路线。**

## 5. 需要第二意见回答的问题

1. Route A 的判据应该怎么预注册?(建议:s3/s4 先评,四种子中位 shuffle 相对 Vanilla 若 ≥10% 则换头条,否则回到描述性定位)
2. knSV3b 与论文 §3 描述是否完全一致?它比 `_eucrd_v5` 多了 `delta_r.mode=green`、`baseline.kind=policy_self`、`tau_0=1.5/tau_mode=linear`——**§3.4 需要相应更新到哪个程度**?
3. 与 Vanilla 的比较目前跨 campaign。是否必须同批重训 Vanilla,还是同协议重评现有 checkpoint 即可?
4. 如果四种子结果只有 ~8%(在 10–13% 噪声底边缘),应该报还是不报?

## 6. 论文里未被动摇的部分(供权衡)

- Vanilla 在 shuffle 下(0.269)比完全不用预报(0.196)还差 —— 论文核心动机,不依赖任何 EU-CRD 数字
- auditor:反相预报使完成率掉至 80%,repair 恢复到 99.9%(3-episode 网格与独立 10-episode 跑相差 ≤2.3%)
- 采样解码下 EU-CRD 0.379,低于 Vanilla 0.397 与 No-Forecast 0.464
- 方法自带的认知信号(critic 集成分歧)在任何常规指标之前翻倍,提前预警失效
- 四个 risk 基线全部丧失完成率控制(18–50%)
- 稿件当前 9 页合规,零悬空引用,数字自洽

## 7. 附:与本处境无关、已确认需修的清单

C_min 自相矛盾(§4 "over all seeds…100%" vs App F "91–96%");删除 risk 基线的 0.4% 独立性推断;"zero inference overhead" 与 auditor 滚动相关矛盾;"matches/pays no premium" 缺等价性检验;auditor 声称需收窄到 inversion;FI 分母病(改用腐蚀增量 + DiD);episode 口径三处矛盾(App B 10 / App H 1 / App I 3);`routing window` 术语残留。
详见 `reports/PAPER_REVIEW_2026-08-22.md`。
