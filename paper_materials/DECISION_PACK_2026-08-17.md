# 论文两大拍板项决策包(2026-08-17)

读完即可拍板。所有数字已从 `local_rt_summary.txt` / 论文 tex 逐格复核。
摘要截稿 9-18(剩 32 天),正文 9-25。

---

## 拍板项 ①:CCA-PG 回补

### 数据现状(单种子 ccaV3_s1,V3 配方与主表两臂同,协议一致)

| | Clean | Blend | Shuffle | FI | C_min |
|---|---|---|---|---|---|
| CCA-PG s1 | 0.1835 (100) | 0.1613 (99.9) | 0.1913 (100) | 1.04 | 99.9 |
| EU-CRD(4 种子中位) | 0.192 (100) | 0.185 (100) | 0.255 (100) | 1.33 | 100 |
| Vanilla | 0.184 (99.7) | 0.184 (100) | 0.269 (100) | 1.46 | 95.6 |

两个已核事实:
- 表面上 CCA-PG 全面占优,但它是 **1 个种子对 4 个种子中位**;EU-CRD 的 shuffle
  跨种子散布 0.207–0.255,CCA 的 0.191 比最好格还低 7.6%——差距在种子噪声量级内,
  **单种子数据没资格下"CCA 更好"的结论,也没资格下"CCA 不如"的结论**;
- 探针实测:CCA-PG 预报敏感度 0.399(EU-CRD 0.534)——**"它学会不用预报所以稳"
  这条辩护不成立**,不能写进论文。

### 选项

**A. 双种子补训后入表**(推荐,若 GPU 机可用)
代价:C-regime 600k × 2 种子 ≈ 20h 机器。本机被 V3.2 判决占用;GPU 机主表
s5/s6 无回传,状态待查——若空闲,这是唯一能把 CCA 行做成"和别的基线同权重
(1 seed)甚至更强(3 seeds)"的路。9 月初前完全来得及。

**B. 单种子如实入表 + 定位文字**(保底,零机器)
表行照实放(协议与其他 1-seed 基线一致),文字定位三点:
(i) CCA-PG 无部署审计器、无不确定性门——EU-CRD 的主张是**信任管理机制**,
不是单点碳数;(ii) C_min 99.9 vs 100:EU-CRD 仍是唯一全绿;(iii) FI 1.04 的
机制未知(探针显示它读预报,为何 shuffle 不伤它需要单独分析,论文如实说
"single-seed, mechanism unexplored")。
风险:审稿人盯着 0.191 vs 0.255 问"你们的方法为什么不如最近的 CA baseline"。
诚实的回答只有种子数,所以 B 必须配"1 seed vs 4-seed median,within EU-CRD's
seed spread"这句表注。

**C. 继续缺席**
不推荐:"没和最近 CA baseline 比"是硬伤,且我们数据在手却不报,一旦被要求
补测,比现在主动报难看得多。

**推荐:先查 GPU 机(一条命令的事),空闲→A,不空闲→B。**

### B 方案的表注草稿(直接可贴)

> CCA-PG is trained with the identical recipe and evaluated under the identical
> protocol (single seed, as for the risk baselines). Its shuffle cell (0.191)
> sits below EU-CRD's per-seed spread (0.207–0.255); with one seed against a
> four-seed median this difference is within seed noise and we do not read a
> ranking from it. Unlike EU-CRD, CCA-PG carries no reliability gate and no
> deployment-time auditor, and its low forecast-corruption sensitivity
> (FI 1.04) is a single-seed observation whose mechanism we did not isolate.

---

## 拍板项 ②:ablW(拆重加权)双消融叙事

### 数据现状(全部已复核)

| shuffle 碳(同种子配对) | s1 | s2 |
|---|---|---|
| EU-CRD 全量 | 0.218 | 0.255 |
| ablW(重加权脱离梯度) | 0.201(−7.6%) | 0.224(−12.2%) |
| ablG(拆门,c≡1) | 0.282(s1) | — |

- ablW 两种子都更低,但幅度都 ≤ 噪声底(10–13%)→ 严格结论:**拆掉重加权,
  shuffle 遏制没有可测的损失**;"ablW 更好"同样不成立;
- ablG 一拆门就崩(0.282–0.286,比 Vanilla 0.269 还差)→ 门 = 遏制来源,成立;
- 架构事实(代码已核):重加权是 CRD 通路作用于梯度的**唯一出口**,ablW 实为
  "整条通路脱离梯度"≈ vanilla+ensemble critic。**"只留门"的格子在架构上不存在**
  ——消融矩阵天然只有三格(全量/无门/无通路),这句写出来审稿人就无法要求第四格;
- ablW 的 anti 条件双种子掉出合同(93.1%/98.6%),但**全量臂的同条件对照格不存在**
  (本地 s3 checkpoint 是塌缩臂,主表臂 ckpt 在 Isambard/GPU)——"重加权买纪律"
  的辩护线**当前无证据支撑**,除非补对照。

### 选项

**A. 双消融如实入表 + 贡献重述**(推荐)
故事从"分解+重加权拯救"移到:**分解提供责任信号,门决定何时信任;遏制由门驱动,
重加权本身在噪声内中性**。这与标题 Epistemic-Uncertainty CRD 兼容(重心=认知门控)。
需要改:§4.2 消融段(草稿见下)、摘要一句、结论一句。
风险:方法叙事重心移动;收益:审稿要求全消融时我们已经先说了,且是数据支持的版本。

**B. 维持现状(只报 ablG)**
数据在手不报,两种子同号。被要求补测时非常被动。不推荐。

**推荐:A。**若想给重加权留辩护线,需要主表臂的 anti 对照格(依赖 Isambard/GPU
checkpoint,机器可用后 ~1h 评测)。

### A 方案的 §4.2 消融段替换草稿(直接可贴)

> Two ablations separate the components. Fixing the gate open ($c_t\equiv1$)
> leaves clean carbon unchanged (0.185 against 0.192) but erases the
> containment entirely, shuffle carbon rising to 0.282, past even Vanilla:
> without the reliability gate, acting on the decomposed credit is worse than
> not decomposing at all. Removing the reweighting instead (the shares are
> still estimated and logged, but no longer scale the advantage) leaves
> shuffle containment statistically unchanged on both seeds (0.201/0.224
> against paired 0.218/0.255, differences within the measured seed-noise
> floor). The mechanism the corruption robustness rests on is therefore the
> decision of *when to trust* the decomposed credit, not the redistribution
> itself; the decomposition supplies the signal that decision is made on.
> A gate-only variant does not exist as a fourth cell: the gate acts on
> learning solely through the reweighting path, so disabling the latter
> disables the former.

---

## 附:v3.x 战役一段话(讨论/局限节备用,等 H1 定稿)

> (H1 ≥10% 版)On a testbed where spatial routing absorbs most of the naive
> forecast value, we verified that an explicit temporal headroom of X% remains
> above the strongest no-forecast router, and that the failure of end-to-end
> training to capture it is a credit-assignment and distillation-interface
> problem rather than an information problem.
>
> (H1 ~5% 版)We additionally report a negative result: once the no-forecast
> policy is allowed to learn spatial routing, the residual value of even a
> perfect forecast on this testbed falls below our measurement noise floor —
> the deferral lever is largely substituted by spatial flexibility. This
> motivates testbeds with correlated (non-substitutable) green supply for any
> claim that forecasts are load-bearing.
