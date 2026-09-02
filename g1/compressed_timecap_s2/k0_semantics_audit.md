# k=0 语义审计（工单 §6 开放项）

日期：2026-09-02。执行：GPU 侧（3060）。仓库起点 `7a40df7`。
探针：`g1/compressed_timecap_s2/k0_semantics_probe.py`，产物 `k0_probe.json`。
**未改动任何现有代码；未训练；未跑调度器碳评测。**

---

## 0. 机械结论

| 问题 | 结论 |
|---|---|
| 训练标签的行对齐 | `pred[0]` 指 history 末行的**下一行**（`label_len = 0` ⇒ `r_begin = s_end`），无歧义 |
| 部署消费的行对齐 | Java godeye 的未来窗**含当前行**；TimeCAP provider 的窗按训练语义从当前行的下一行起 |
| 是否存在偏斜 | **存在，偏斜在接线上，方向是 TimeCAP provider 的窗比 Java godeye 晚一行** |
| 网络本身是否学出了一行位移 | **无法认定**。lead-0 有干净的 offset=0 极小值，但逐 lead 的 argmin 漂移（0, −1, −2, −2, −2, +2, −2, +2, −2），按探针执行前写定的判别规则，这是持续性伪影而非系统性位移 |
| 对已有标定的影响 | **不作废**。`label-offset 0` 测的正是 Java 侧消费约定下的部署态质量 |

附带发现（超出 k=0 问题本身，但更重要）：
**该 checkpoint 在大部分 lead 上不优于"保持最后观测值"的持续性基线**（§4）。

---

## 1. 训练侧：标签窗从哪一行起

`drl-manager/Code/data_provider/data_loader.py`，两个取样类的索引代码相同：

    Dataset_Custom.__getitem__ (L387-401)        SingleDataset.__getitem__ (L24-35)
        s_begin = index                              s_begin = idx * stride
        s_end   = s_begin + seq_len                   s_end   = s_begin + seq_len
        r_begin = s_end - label_len                   r_begin = s_end - label_len
        r_end   = r_begin + label_len + pred_len      r_end   = r_begin + label_len + pred_len
        seq_x = data_x[s_begin:s_end]                 seq_x = data[s_begin:s_end, :]
        seq_y = data_y[r_begin:r_end]                 seq_y = data[r_begin:r_end, :]

`label_len = 0`（`train_timecap.py:106`、`predictor.py:84`，本轮 smoke 的
`model_args.json` 实测同样是 0）。代入：

    r_begin = s_end = s_begin + 96
    seq_x = 行 [s, s+96)          history，末行 h = s+95
    seq_y = 行 [s+96, s+240)      label

**训练标签的第 0 行是 h+1，即 history 末行的下一行。** 这一条没有歧义。

（顺带记录，与本审计无关但已在 `clean_dataset.py` 处理：
`border1s = [0, num_train - seq_len, len - num_test - seq_len]`（L343）就是 val/test
起点回拉 seq_len 的来源，也是审计报告里 train/val 共享 96 行的成因。）

## 2. 部署侧：特征窗从哪一行起

### 2.1 Java godeye（被替换的一方）

`GreenEnergyProvider.computeFutureTrendFeatures(double simTime)`：

    int currentIdx = simTimeToRowIndex(simTime);
    int shortEndIdx = Math.min(currentIdx + shortTermRows, series.length);
    for (int i = currentIdx; i < shortEndIdx; i++) shortSum += series[i];
    double startPower = series[currentIdx];
    double endPower   = series[shortEndIdx - 1];

循环从 `currentIdx` 开始，**含仿真当前所站的那一行**。窗是 `[t, t+short)`。
long 段同理，`peakIdx` 的下界也是 `currentIdx`（`peakTiming = 0` 表示"峰就在当下"）。

### 2.2 TimeCAP provider（替换方）

`TimeCAPGodEyeProvider.step_and_get(step)` 先 `update(step)` 再取特征；
`TimeCAP_GreenPredictor.update(simulation_step)` 推入的是**第 `simulation_step` 行**
（`get_feature_at_time(tid, float(simulation_step))`）。所以取特征时 buffer 末行 = 行 `t`。
特征用 `mean(forecast[:short])`，而按 §1 的训练语义 `forecast[0]` = 行 `t+1`。
窗是 `[t+1, t+1+short)`。

### 2.3 偏斜

    Java godeye            [t,   t+short)
    TimeCAP provider       [t+1, t+1+short)

**TimeCAP 侧的窗整体晚一行。** short_trend 同样错开一行：Java 是
`series[t+short-1] − series[t]`，Python 是 `pred[short-1] − pred[0]` = 行 `t+short` − 行 `t+1`
（跨度相同，起点差一行）。

模块 docstring 声称与 Java 的聚合"byte-for-byte"一致——**跨风机聚合确实一致，
但单风机的窗起点不一致**，这两件事不该被同一句话覆盖。

### 2.4 顺带查出的第二处不一致：peak_timing 的分母

    Java    peakTiming = (peakIdx - currentIdx) / longAvailable            // 分母 N
    Python  peak_timing_t = peak_idx / max(lt - 1, 1)                      // 分母 N-1
            (timecap_godeye_provider.py:552)

这是与行对齐无关的独立差异：同一个"峰在窗内的相对位置"，两边分母差 1。
`long_term_steps` 越小，差异越大。**建议一并纳入修复，不要只修行偏斜。**

## 3. 实证：网络是否真的学出了一行位移

在现有 checkpoint（`fa86c59d…`）、2020 五台风机、stride 480、335 个锚点上，
把标签窗整体平移 `offset ∈ {−2…+3}`，比较 `pred[i]` 与 `truth[anchor+offset+i]`。
锚点定义与标定脚本一致：`update(0..anchor)` 之后取预测，`anchor` 是最后一个喂进去的行。
单线程钉死（`torch.set_num_threads(1)`），与 `7a40df7` 的复现纪律一致。

    offset         RMSE          MAE       corr   RMSE@lead0  corr@lead0
        -2     425.1945     301.4363   0.372633     196.0764    0.880693
        -1     426.0146     302.1454   0.369661     132.5322    0.943274
         0     426.9312     302.9171   0.366390     101.4079    0.968572   <-- lead0 最小
         1     427.9413     303.8291   0.362698     137.2639    0.937754
         2     428.9097     304.6803   0.359262     215.4657    0.849784
         3     429.8126     305.4673   0.356119     209.5075    0.848911

**汇总 RMSE 不是对齐统计量，不要用它下结论。** 它随 offset 单调下降、在扫描区间内没有极小值：
这只是因为它被长 lead 主导，而长 lead 上模型回归到一个水平值，把标签窗整体往前移就等于
把它挪近已知的近期历史。真正的判别信号在短 lead。

lead-0 上 offset=0 有干净的 V 形极小（196.1 / 132.5 / **101.4** / 137.3 / 215.5），
相关系数同样在 0 处最高（0.9686）。孤立地看，这指向"`pred[0]` 对应 anchor 行"，
即部署侧的消费约定，而不是训练侧的 anchor+1。

**但这一条不足以定案**，因为一个在短 lead 上学成近似持续性的模型，无论训练标签怎么构造，
都会让 `pred[0]` 最贴近最后观测行。探针在执行前就写定了判别规则：
*同一个 offset 若在短、中、长 lead 上都胜出，才算真的位移；胜者随 lead 漂移则是持续性伪影。*

逐 lead 结果：

     lead         -2         -1          0          1          2          3  argmin    持续性
        0    196.076    132.532    101.408    137.264    215.466    209.507       0      0.000
        1    146.943    136.211    166.113    241.159    233.525    284.899      -1     88.456
        2    163.815    190.147    260.931    252.662    299.499    297.910      -2    181.561
        5    272.103    321.463    318.793    352.460    337.650    350.177      -2    240.336
       11    396.985    436.247    428.523    433.785    433.354    446.448      -2    358.088
       23    426.506    431.476    423.442    422.575    417.796    428.647       2    381.614
       47    442.088    442.516    446.151    460.721    446.317    442.345      -2    453.236
       95    418.253    407.940    394.779    388.404    387.151    398.926       2    408.638
      143    438.653    448.202    448.971    457.395    460.769    475.121      -2    440.703

argmin 序列是 `0, −1, −2, −2, −2, +2, −2, +2, −2` —— **漂移，不收敛到任何单一 offset。**
按预先写定的规则，**判定为：没有证据表明网络学出了系统性的一行位移。**

值得注意的是 lead 0/1/2 的 argmin 分别是 0/−1/−2，三者指向的**绝对行都是 anchor**：
早期几个 lead 都最贴近同一行（最后观测行）。这正是短 lead 近似持平输出的特征，
与"持续性伪影"的解释一致，而与"整体位移"的解释不一致。

**结论：偏斜在接线上（§2.3），不在网络里。**

## 4. 超出 k=0 问题的发现：该 checkpoint 不优于持续性

上表最后一列是"保持最后观测行不变"的持续性基线。逐 lead 对比（模型取该 lead 的最优 offset）：

    lead     模型最优 RMSE      持续性 RMSE      模型是否更好
       1          136.211           88.456      否（差 54%）
       2          163.815          181.561      是（好 10%）
       5          272.103          240.336      否（差 13%）
      11          396.985          358.088      否（差 11%）
      23          417.796          381.614      否（差  9%）
      47          442.088          453.236      是（好  2%）
      95          387.151          408.638      是（好  5%）
     143          438.653          440.703      持平

**在 1 到 23 步这段——也就是调度器最可能用得上的近场——模型多数比持续性还差。**
lead 0 的 101.4 也说明它连"照抄最后一行"都没做到（照抄的 RMSE 是 0）。

这与 `timecap_cal.json` 里 `sigma_rel = 1.19`（残差标准差约为平均绝对水平的 1.19 倍）
互相印证，是同一件事的两种量法。**这不是本审计要回答的问题，但它对
Stage A′ 的 `timecap_cal` 档意味着什么，应该由 5080 侧在判读阶梯时一并考虑：
"现实档"所站的质量水平，可能低于一个零成本的持续性基线。**

（措辞约束照旧：以上只描述现有 checkpoint 的已测残差水平，不构成对 TimeCAP 方法的判断。）

## 5. 对已有标定的含义

**`timecap_cal.json`（`label-offset 0`）不作废。** 理由：

1. 它测的是**部署态质量**——`label-offset 0` 把 `pred[0]` 对到 anchor 行，
   而 anchor 行正是 Java godeye 未来窗的第一行（§2.1）。也就是说它测的正是
   替换方在真实消费约定下会犯的误差，§2.3 的一行偏斜**已经被计入残差**。
2. §3 判定网络没有系统性位移，所以标定值不需要按"扣掉一行偏斜"重新解释。
3. 该产物在 `7a40df7` 的 Addendum A 中已按"原样冻结、降格声明"处理，本审计不改变该处置。

**唯一会受影响的是将来重训时的标签构造**，见 §6。

## 6. 给将来重训的要求（现在不执行）

如果 Stage A′ 之后决定重训预测器，必须在预注册里二选一并用测试钉死，不能两边各说各的：

- **甲案（改标签，贴合 Java）**：训练时令 `pred[0]` 对应**当前行**，即
  `r_begin = s_end - 1`（等价于 `label_offset = 0` 的消费约定）。改动落在新的
  dataset wrapper 里，不碰 `Code/`。好处是 Java godeye 与 TimeCAP provider 的窗
  自然对齐，`godeye` 档与 `timecap_cal` 档可以逐行比较。
- **乙案（改消费，贴合训练）**：保留 `label_len = 0` 的训练语义，把 Java 的
  `computeFutureTrendFeatures` 改成从 `currentIdx + 1` 起（或把 Python provider 的
  窗前移一行）。这动 Java，代价更大，且会改变既有 godeye 观测通道的数值。

**推荐甲案**：它只动新写的训练数据构造，不改任何已冻结的观测通道。

无论选哪案，`peak_timing` 分母的 N vs N−1（§2.4）要一并修，并加一个
Java/Python 逐位对拍的测试——否则"drop-in replacement"这个说法仍然不成立。

## 7. 复现

    drl-manager/.venv/bin/python g1/compressed_timecap_s2/k0_semantics_probe.py \
      --checkpoint drl-manager/timecap_prediction/TimeCAP/model/\
finetune_TimeCAP_custom_sl96_baseline_4358062/ckpt_best.pth \
      --turbine-id 12 --turbine-id 36 --turbine-id 91 --turbine-id 95 --turbine-id 96 \
      --year 2020 --stride 480 --device cpu --out g1/compressed_timecap_s2/k0_probe.json

    checkpoint sha256   fa86c59df99d4fa0228ba07e018bdd399017e5e1f673edc316032a5871a9fb59
    锚点数              335（与 timecap_cal.json 的 n_windows 一致）
    torch 线程          1（钉死）
