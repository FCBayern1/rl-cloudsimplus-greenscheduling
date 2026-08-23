# App F 预注册:脆弱种子的定义与分级

冻结于 2026-08-23,**在看到任何 G1 结果之前**。Codex 裁定,本文件为权威。
论文侧的开关在 `paper_latest/iclr2027_conference.tex` 前言(`\ifFragileNone` / `\ifFragilePattern`)。

## 为什么要预注册

现有 App F 讲的是"四分之一种子在贪心解码下过度延迟",这套叙事建立在坏构建的 4 个种子上。
新战役可能复现它、可能只出现一次、也可能完全消失。三种结局对论文的写法要求不同,
而**在看到结果之后再定义"存在"就是事后编故事**。所以先冻定义、再冻分级。

## 脆弱的定义(逐字)

一个 checkpoint 判为**脆弱**,当且仅当同时满足:

1. **训练期完成率达到合同**(训练日志末次完成率 ≥ 99.5%);
2. **同一 checkpoint 在 stochastic clean 解码下达到合同**(完成率 ≥ 99.5%);
3. **同一 checkpoint 在 deterministic clean 解码下低于 99.5%**。

三条缺一不可。这个定义把失败**锁定在 mode selection**:学到的分布是健康的
(条件 1 和 2),只有取 argmax 时塌掉(条件 3)。

计数对象是 **24 个策略 checkpoint**(12 种子 × 2 臂),不是 12 个种子。

### 99.5% 的口径(现在锁定)

条件 2 和 3 里的完成率,一律指**同一组三个注册窗口上 pooled 的 MI completion**:

```
completion = Σ_windows (completed_MI) / Σ_windows (submitted_MI)
```

三个窗口是 low(k=19)/ mid(k=56)/ high(k=34),每窗一局。
**不得**事后取最优窗口、最差窗口或任一单局。deterministic 与 stochastic 用同一组窗口。

## 定级流程(两步,不必等完整采样附录)

完整的 216 局采样附录排在最后,但 App F 定级不必等它。按定义分两步:

1. **筛候选。** 用训练日志(条件 1)和 deterministic clean 三窗 pooled(条件 3)找出候选。
2. **只对候选补 stochastic clean。** 候选为 0 则直接定级为 0,不跑任何 stochastic;
   候选存在则只对这些 checkpoint 跑 stochastic clean 三窗,完成定级。

这不是按结果筛选:stochastic 检查本来就是脆弱定义的必要条件,而条件 1 和 3 是它的前置。
对不满足前置的 checkpoint 跑 stochastic 不会改变任何一个判定。

完整采样附录仍按原排期最后运行,与本定级互不替代。

## 分级(结果 → 论文怎么写)

| 脆弱 checkpoint 数 | 处置 | 开关 |
|---|---|---|
| **0** | 删除 App F。正文 C$_{\min}$ 段改为"两臂所有种子均守住合同" | `\FragileNonetrue` |
| **1** | 最多作为孤立失败案例写进 Limitations,**不得声称是稳定模式**,不保留 App F 的分析 | `\FragileNonefalse` + `\FragilePatternfalse` |
| **≥ 2** | 可保留"重复出现的 mode-selection fragility"小节 | `\FragileNonefalse` + `\FragilePatterntrue` |

## 两条附加约束

**(1) 归属主张。** 只有当 **Vanilla 与 EU-CRD 各至少出现 1 个**脆弱 checkpoint 时,
才能写"这属于共享的训练配方,而不是 EU-CRD 特有"。若脆弱只出现在一个臂上,
那句话必须删掉,并且这一现象要按"该臂的性质"来报告 —— 这对我们不利,但那正是数据说的话。

**(2) 早期预警主张。** 现有的"critic-ensemble disagreement 是唯一的训练期先兆"
是 EU-CRD 的一个卖点。**新数据不再支持它时,该主张删除**,不得降格为"通常"或"倾向于"。
支持的判据:在脆弱种子上 $\sigma^2$ 的中段升高,且在同臂健康种子上不出现同等升高。

## 与采样附录的耦合

条件 2 需要 stochastic clean 评测。它属于 Codex 批准的采样附录那 216 局
(secondary / descriptive),排在确定性主判决之后。因此 **App F 的分级在采样附录跑完之前
无法定级**,在此之前论文保持 `\FragilePatterntrue` 的当前措辞,并视为未定。
