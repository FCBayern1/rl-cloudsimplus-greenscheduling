# TimeCAP 重训预注册（草案）

**状态：草案。生效以「今晚 ladder-v2 判决」+「Codex 批准」为共同前提。**
在两者齐备之前，本文件不授权任何训练，不授权更换任何档位，也不得被引用为已冻结判据。

起草：2026-09-02，GPU 侧（3060）。仓库起点 `d32f6f4`。
上游依据：`reports/WORKORDER_GPU_COMPRESSED_TIMECAP_SCHEME2.md`、
`reports/LADDER_V2_PREREG.md`、`g1/compressed_timecap_s2/k0_semantics_audit.md`、
`g1/compressed_timecap_s2/timecap_data_audit.json`。

措辞约束继承 ladder-v2：一切产物只能称 **synthetic forecast-quality ladder** 的输入，
新 checkpoint 的标定档只能称 `checkpoint_residual_surrogate_v3`，
**不得**称其为「TimeCAP 的表现」。

---

## 0. 为什么要重训（问题陈述，不是结论）

两件已测事实构成动机，二者都不预判重训会成功：

1. **旧管线判了 `STOP_DATA_PIPELINE`**（`timecap_data_audit.json`）：单文件拼接 +
   `Dataset_Custom` 按行 7:1:2，train 段有 478 个跨风机窗；val 起点回拉 seq_len，
   与 train 共享 96 行。现有 checkpoint 就是在这个管线上训出来的。
2. **现有 checkpoint 的近场残差劣于零成本持续性基线**
   （`g1/compressed_timecap_s2/persistence_baseline_cal.json`，见 §6）。

重训要回答的是：**换掉脏数据与错标签之后，预测质量能否越过持续性这条免费的地板线。**
若不能，结论不是「TimeCAP 不行」，而是「在本数据与本视界下，学习式预报未超过持续性」，
这本身是一个可发表的负结果，且它对 Scheme 2 的意义由 ladder 判决另行决定。

---

## 1. 数据（冻结）

    风机          12, 36, 91, 95, 96   —— 部署使用的五台，与 C-regime 拓扑一致
    年份          2020，且仅 2020
    文件          cloudsimplus-gateway/src/main/resources/windProduction/split/
                  Turbine_{id}_2020.csv                 （13 特征 + Patv 的 SDWPF 格式）
    装载器        g1/compressed_timecap_s2/clean_dataset.py
    切分          逐文件 7:1:2，三段行区间严格相邻、不重叠、无 seq_len 回拉
    窗口          必须整体落在**一个文件的一个 split** 内
    scaler        逐文件，只在该文件自己的 train 段拟合
    2021          训练与验证一律不得触碰
    2022          禁用（每台仅两行零值）

隔离依据：2020 与 2021 是**不同文件**，训练集与 2021 的调度评测窗在文件层面零交集；
`clean_dataset` 的 `forbid_years` 默认拒收 2022，由测试钉死。
这三条已由 `test_clean_dataset.py`（23 测试）与 `test_train_timecap_clean.py`（14 测试）覆盖。

冻结时必须记录五个源文件的 SHA256（`persistence_baseline_cal.json::val_csv_shas`
已含同一组，可直接复用核对）。

## 2. 标签：甲案（冻结）

k=0 审计（`k0_semantics_audit.md`）确认的事实：

    训练侧   label_len = 0 ⇒ r_begin = s_end，y[0] = history 末行的**下一行**
    部署侧   Java computeFutureTrendFeatures 从 currentIdx 起，**含当前行**
    ⇒        TimeCAP provider 的窗比 Java godeye 晚一行

**甲案：改标签贴合部署，令 `y[0] ≙ history 末行本身（当前行）。**

    y = 行 [s_end - 1, s_end - 1 + pred_len)
    窗口跨度 = seq_len + pred_len - 1 = 239 行（不再是 240）

### 2.1 需要的实现改动（本草案不执行）

`clean_dataset.CleanWindowDataset` 增加一个显式参数：

    label_start_offset : int = 0
        y = 行 [start + seq_len - label_start_offset,
                start + seq_len - label_start_offset + pred_len)
        0 = 现状（贴合旧训练语义）；1 = 甲案（贴合部署消费）
        窗口跨度随之为 seq_len + pred_len - label_start_offset

**命名警告（必须写进代码注释，否则一定会被混淆）**：这个 `label_start_offset` 与
`residual_calibration.py --label-offset` **不是同一个量**。后者是**消费侧**的对齐参数
（`pred[0]` 与哪一行真值比对），前者是**构造侧**的（标签窗从哪一行起）。
甲案生效后二者会指向同一行，但它们仍是两个独立的旋钮，不得互相推断。

### 2.2 必须钉死的测试（执行前提交）

    T1  label_start_offset=1 时，y[0] 逐位等于 x[-1] 所在行的真值
    T2  label_start_offset=0 时，行为与现状逐位一致（回归保护）
    T3  窗口跨度 = seq_len + pred_len - label_start_offset，窗口数等于边界算术
    T4  跨文件窗 = 0、跨 split 窗 = 0，在两种 offset 下都成立
    T5  用 label_start_offset=1 训出的模型，其 predict() 的 pred[0] 与
        Java godeye 未来窗的第一行指向同一 CSV 行（端到端对齐断言）

### 2.3 一并修复（甲案的附带条件）

`peak_timing` 分母不一致，必须同批修完并加 Java/Python 逐位对拍测试：

    Java     peakTiming    = (peakIdx - currentIdx) / longAvailable      // 分母 N
    Python   peak_timing_t = peak_idx / max(lt - 1, 1)                   // 分母 N-1
             (timecap_godeye_provider.py:552)

在这条修完之前，`TimeCAPGodEyeProvider` 的 docstring 不得继续声称与 Java 聚合
「byte-for-byte」一致。

## 3. 模型与超参（冻结：与现有 checkpoint 同构，只换数据与标签）

    seq_len            96
    pred_len           144
    enc_in             13
    features           MS（13 路输入，只预测 Patv）
    d_model / d_ff     736 / 992
    e_layers / n_heads 2 / 8
    depth              2
    patch_len          [96, 24]      stride_time [96, 24]      window_size [3, 3]
    use_ar_head / use_os_head   True / True
    alpha / beta       1.0 / 0.3326362081926146
    label_len          0（Code/ 的参数，保持 0；甲案的位移由 clean_dataset 承担，
                         **不通过改 label_len 实现**，那会改变 y 的长度）
    seed               20260901
    checkpoint 选择    **只按 validation loss**
    调度器碳           不得参与选 epoch、选学习率、选 checkpoint 或任何早停决定

模型结构不动，是为了让重训前后的差异可归因于**数据边界 + 标签对齐**这两个自变量，
而不是第三个。任何结构改动都要另立预注册。

## 4. 训练预算与单卡命令

实测基准（本机 RTX 3060，A′ 与 D3 smoke）：

    单 epoch，batch 32，AMP 自动启用    368 s（约 6.1 分钟，110139 个训练窗）
    显存峰值                            1477 MiB / 12288 MiB
    CPU 对照（i7-9700K 8 线程）          7101 s

甲案下五台风机 2020 的训练窗数 ≈ 5 × (22556 − 239 + 1) = 111590，与上面的基准同量级，
因此 **batch 32 下预计仍是约 6 分钟一个 epoch**。

    epochs             30
    patience           5
    batch_size         32（基线）
    lr                 5e-5
    预算上限           30 epoch ≈ 3.1 小时；patience 命中通常更早

显存余量很大（1477 / 12288 MiB），**允许**把 batch 提到 64 或 128 以缩短墙钟，
但那会改变优化轨迹，属于需要在本预注册里显式选定的参数——
**若要改 batch，必须在本文件冻结时就写死，不得开跑后再调。**
草案默认取 batch 32，理由是它与已测基准同参，便于与旧 checkpoint 对照。

命令（路径以冻结 manifest 为准）：

    cd /home/joshua/rl-cloudsimplus-greenscheduling
    drl-manager/.venv/bin/python g1/compressed_timecap_s2/train_timecap_clean.py \
      --turbine-id 12 --turbine-id 36 --turbine-id 91 --turbine-id 95 --turbine-id 96 \
      --year 2020 \
      --res-dir drl-manager/timecap_prediction/TimeCAP/model/retrain_clean_v3 \
      --label-start-offset 1 \
      --epochs 30 --batch-size 32 --lr 5e-5 --patience 5 --gpu 0 --seed 20260901

**不启用 `--multi-gpu`**：本机单卡，且分布式入口尚未做 1-epoch `torchrun` smoke 与
checkpoint 单写者检查（原工单 §9）。单卡能在预算内完成，就不引入新变量。

训练完成必须记录：checkpoint SHA256、`model_args.json` SHA256、五个源 CSV 的 SHA256、
code commit、CUDA / torch 版本、GPU 型号、best epoch、完整 validation 曲线、
test MSE / MAE（**只报告，不用来回调任何下游**）。
大 checkpoint 不入 Git，以路径 + 大小 + SHA256 登记。

## 5. 训后动作（顺序固定）

    1. 新 checkpoint 走 DC 级标定：g1/compressed_timecap_s2/dc_residual_calibration.py
       的同一口径（DC0=T12+36 / DC1=T95+91 / DC2=T96、2020、stride 480、
       label-offset 0、**单线程钉死**、线程数写入产物）
       —— 新产物 SHA 必须跨机可复算（ladder-v2 Addendum A 的新纪律对新产物生效）
    2. 产出 checkpoint_residual_surrogate_v3 的参数：
       sigma_rel_dc / ar1_rho / lead_alpha / 相关矩阵 / c，
       并检查单因子复现容差 max|r_ij − c| ≤ 0.10
    3. 与 E1 的持续性基线做 §6 的逐 lead 对比，机械出验收结论
    4. 以 **append-only addendum** 追加到 ladder-v2 预注册，交裁定
    5. **裁定通过之前不换档**：v3 不得替换 v2 出现在任何已跑或在跑的阶梯里

## 6. 验收（执行前冻结，看到结果后不得放宽）

靶子来自 `g1/compressed_timecap_s2/persistence_baseline_cal.json`
（持续性预报器走**完全相同**的 DC 级标定口径；该脚本的协议已与冻结的
`dc_residual_cal.json` 交叉校验，相对偏差 ~1e-7，即已记录的 BLAS 线程数签名）。

持续性基线（2020，67 锚点，stride 480，label-offset 0）：

    sigma_rel_dc      DC0 1.164368   DC1 1.080746   DC2 1.166804
    ar1_rho           0.983493
    lead_alpha        0.050000  ← 被 0.05 下限截断，原始值为 0（见下方说明）
    c                 0.823154        off-diagonals 0.797712 / 0.823154 / 0.902607
    single_factor_ok  True

现有 checkpoint 在同一口径下：

    sigma_rel_dc      DC0 1.122552   DC1 1.146334   DC2 1.205660
    ar1_rho           0.980798      lead_alpha 0.238700      c 0.860059

逐 lead RMSE（模型 / 持续性）：

     lead     DC0 模型   DC0 持续    DC1 模型   DC1 持续    DC2 模型   DC2 持续
        0      190.09       0.00      208.15       0.00       89.97       0.00
        1      275.56     123.11      363.06     194.32      146.74      79.51
        2      452.39     290.57      541.86     359.60      217.24     150.12
        5      617.73     526.79      640.28     393.84      253.95     195.91
       11      845.66     748.48      858.35     675.64      366.34     307.75
       23      776.22     791.62      890.31     746.50      375.70     304.86
       47      900.89     988.15      873.98     867.39      379.55     348.43
       95      860.41     900.34      727.53     735.02      308.85     337.48
      143      930.65     930.99      869.58     860.95      368.73     353.00

    现有 checkpoint 优于持续性的 (DC × lead) 格数
        近场 lead 1–23     DC0 4/23    DC1 0/23    DC2 0/23    合计 4/69
        全 144 lead        DC0 95/144  DC1 33/144  DC2 34/144  合计 162/432

### 6.1 验收门（全部满足才允许换档）

    G1  近场全面优于持续性
        对 DC ∈ {0,1,2} 与 lead ∈ [1, 23] 的全部 69 个格，
        新模型的残差 RMSE **严格小于**持续性基线的同格 RMSE
    G2  聚合不劣
        sigma_rel_dc 在三个 DC 上均 **不高于** 持续性基线的同 DC 值
    G3  单因子仍成立
        新标定的 max|r_ij − c| ≤ 0.10（与 ladder-v2 §2 同一容差）
    G4  管线合同
        clean_dataset 三个 split 的 cross_file_windows = 0、cross_split_windows = 0、
        split_row_overlaps = []、scaler_fit_is_train_only = True
    G5  标签对齐
        §2.2 的 T1–T5 全绿，其中 T5 的端到端对齐断言不得跳过
    G6  复现性
        新标定产物单线程生成，SHA 跨机可复算（同输入重跑逐字节相同）

**任一门不过 → 不换档。** 记 `STOP_RETRAIN_BELOW_PERSISTENCE`（G1/G2 失败）或
`STOP_RETRAIN_PIPELINE`（G3–G6 失败），负结果照常提交并推送。

### 6.2 关于 lead 0 的说明（避免误读）

持续性基线在 lead 0 的残差**恒为 0**，这是 `label-offset 0` 下的构造性结果
（真值窗与持续性外推都从锚点行起），不是它的预测能力。因此：

- 其 `lead_alpha` 原始值为 0，被共享的 0.05 下限截断，**该值不具可比性**，
  不得与模型的 0.2387 并列解读；
- G1 的判定带**从 lead 1 起**，已排除这个构造性零点；
- ladder-v2 的 R3 已规定**全档 lead 0 使用真值**，所以 lead 0 的差异在阶梯里
  本来就被中和，这与 G1 的取值带一致。

### 6.3 这条靶子的第二个用途

若阶梯的现实档失败，`persistence_baseline_cal.json` 提供「这不是天花板」论证的量化版：
一个零成本、零参数、零训练的预报器在同一口径下的残差水平是已测的，因此
「现实档没通过」与「任何预报都不可能通过」是两个可以被数据分开的命题。
**该用途不授权任何事后改判**，只授权在负结果报告里陈述这一量化对照。

## 7. 本草案不包含、也不授权的事

- 不授权改动 `drl-manager/Code/`（甲案的位移全部由 `clean_dataset` 承担）。
- 不授权改动 ladder-v2 的任何判据、窗口、档位或阈值。
- 不授权用新 checkpoint 重跑或改判任何已完成的阶梯。
- 不授权在看到 §6 结果之后调整 §3 的超参、§4 的预算或 §6 的门槛。
- 不授权把 batch 从 §4 冻结的值改掉。

## 8. 待 Codex 裁定的开放项

1. **batch_size**：草案取 32（与已测基准同参）。若为墙钟考虑要提到 64/128，
   请在批准时一并写死，之后不得再动。
2. **G1 的严格性**：69 格全胜是很硬的门。若认为应放宽（例如允许 ≤3 格不达标），
   必须在批准时写死具体数字，不得在看到结果后决定。
3. **失败后的下一步**：G1/G2 失败时，是就此收束为负结果，还是授权一次
   「换视界 / 换目标变量」的探索？后者需另立预注册，本草案不预设。

## 9. 裁定(2026-09-02,经用户授权由 5080 Claude 代行,供 Codex 事后复核)

§8 三项开放项与生效条件裁定如下,全部在任何重训结果产生之前写死:

**9.1 batch_size = 32,冻结。** 与已测基准同参,保持与旧 checkpoint 的可对照性;
3.1 小时墙钟可接受,不为省两小时引入优化轨迹这个新变量。此后不得再动。

**9.2 G1 修订为容错版,数字现在写死:**

    G1' 近场优于持续性
        69 个格(DC ∈ {0,1,2} × lead ∈ [1,23])中,新模型 RMSE 严格小于
        持续性同格 RMSE 的格数 ≥ 66;
        且任何败格的劣幅 (model − persistence) / persistence ≤ 5%

理由:69/69 全胜使单格 0.1% 的抖动就能废掉整次训练,脆而不严;66/69 + 败格 ≤5%
保住"全面优于"的实质,同时不给噪声一票否决权。G2–G6 原样冻结。

**9.3 失败路径 = 就此收束为负结果。** G1'/G2 失败记 STOP_RETRAIN_BELOW_PERSISTENCE,
照常提交推送;"换视界 / 换目标变量"的任何探索必须另立预注册,本文件不授权。

**9.4 生效条件精确化为自动触发,免去午夜人工往返:**

    情形 A  ladder-v2 四门全过(含 surrogate ≥50%)
            → 本预注册保持 PARKED,重训非必需;Stage D 预注册优先
    情形 B  单调、shuffle、anti、合同、排除率诸门全过,仅 surrogate 档 <50%
            → 本预注册自动 ACTIVE,3060 可即刻按本文件执行,无需再批
    情形 C  负控或单调或合同任一门失败
            → Scheme 2 按裁定 STOP;本预注册保持 PARKED,不为死考场重训

**9.5 附带条款照准:**§2.1 的 label_start_offset(构造侧)与 --label-offset(消费侧)
双旋钮命名分离;§2.3 的 peak_timing 分母修复 + Java/Python 逐位对拍测试为甲案必修项,
修完之前 provider 不得声称 drop-in / byte-for-byte。

本裁定为代行;Codex 复核若推翻任何一条,以 Codex 为准,已产生的产物按其裁定处置。

**9.6 §4 命令行更正(裁定,先于任何训练):**§4 的命令补上 `--label-start-offset 1`。
该参数在 E2 起草时尚不存在,漏掉它会静默训出 stock 标签约定,与 §2 冻结的甲案相悖,
且要到 G1′ 验收才暴露。这是让命令与正文自身的冻结条款一致的更正,不改任何判据;
情形 B 触发时以 §4 更正后的命令为准。
