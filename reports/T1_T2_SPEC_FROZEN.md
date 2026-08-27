# T1 / T2 冻结口径(Codex 2026-08-27 裁定)

写于跑任何一格之前。**T1 与 T2 无论结果如何都必须报告;不得让 T1 的结果决定是否启动 T2。**

## T1 预报特征的冗余度与条件价值

原口径(只算 forecast 特征对 `dc_green_ratio` 的 R²)不够 —— 高相关不等于没有独立价值。
按裁定分两层:

### 第一层 冗余度
用**全部盲态可见变量**预测 forecast 特征,报 **时间分块 OOF R²**。
盲态可见 = 全局观测里除 `dc_future_short_mean / short_trend / long_mean / long_peak_timing` 之外的一切。
分两个子集报告:
- **外生子集**(与动作无关):`dc_current_green_power_w`、`dc_green_ratio`、`dc_current_power_w`、时间索引
- **全集**:外生子集 + 队列/利用率等受动作影响的量(采集时的脚本策略要写明)

### 第二层 条件价值
比较 `blind` 与 `blind + forecast` 在下列目标上的 OOF 增量:
- **T-a** 未来实现绿电(每 DC,视界 h)
- **T-b** oracle 排序标签(未来 h 步内实现绿电最高的 DC),策略无关

冻结报告量:

```
ΔR² = R²(blind + forecast) − R²(blind)
```

### 判据(先冻)
- **高冗余**:第一层 R² > 0.7
- **信息稀薄**:ΔR² < 0.05
- **只有两条同时成立**,才支持"被大量使用但几乎不携带独立信息"。任一不成立即该假设被削弱,如实报告。

### 方法约束
**一律 blocked OOF(按时间连续分块),禁止随机行切分** —— 风电自相关会把 R² 虚高。
折数 5,连续块,不打乱。

## T2 No-Forecast 诊断

定位为**诊断实验**,不再受已停止的扩种子阶梯约束。

- 种子:**101 / 102 / 103**,与 matched Vanilla 相同的前三个
- 窗口:相同三个分层窗口(low k=19 / mid k=56 / high k=34)
- 完成合同一致
- **每 seed 先池化三窗,再算配对 log-ratio**
- 报**几何均值、3/3 方向、原始值**
- **不做显著性声称**
- 配置从 matchedvan 程序化派生,只差 `forecast_mode` 与身份字段,加精确差分守卫
- 冻结 jar,SHA 开跑前校验

### 冻结预测
> No-Forecast 与 matched Vanilla 的 clean 碳差距**绝对值小于 5%**。

## 论文主线 B 的措辞约束(裁定原文)

不得写"信用重加权**必然**放大依赖",除非消融独立证明因果。现阶段措辞:

> In this registered evaluation, EU-CRD exhibited 1.13–1.37x larger forecast-induced
> policy shifts across all four corruptions, contrary to its intended robustness mechanism.

T1/T2 负责解释"为什么",但即使它们失败,**G1 负判决与四条件放大事实仍然成立**。
