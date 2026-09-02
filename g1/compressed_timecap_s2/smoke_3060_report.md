# GPU 侧预备任务报告：1-epoch smoke、跨机确定性复核、干净装载器

日期：2026-09-02。仓库起点：`origin/main` @ `90fd542`。
执行机：本地 GPU 机（Intel i7-9700K / 8 核 / 31 GB / NVIDIA RTX 3060）。

**本报告的 smoke 产物只证明管道通，数字无意义，不得被任何判据引用。**
本轮未跑任何调度器碳评测，未启动正式训练。

---

## 0. 结论速览

| 任务 | 结论 |
|---|---|
| A 1-epoch smoke | **部分通过**：checkpoint / `model_args.json` 落盘、predictor 回读并出 144 行预测均 OK；**`CUDA 可用` 这一条 FAIL**，实际跑在 CPU 上 |
| B① 17 个扰动测试 | **通过**，17/17 |
| B② 标定复核 | **未字节一致**：SHA 不同，但 provenance 字段全等、三个拟合数相对偏差 ≤ 1e-7。根因已定位为 BLAS 线程数，非语义差异 |
| C 干净装载器 | **通过**：`clean_dataset.py` + 23 个测试全绿，未用于任何训练 |

---

## 1. 环境

    Python        3.12.3
    torch         2.11.0+cu130   built cuda 13.0   cudnn 91900
    GPU 硬件      01:00.0 NVIDIA GA106 [GeForce RTX 3060 Lite Hash Rate]   (lspci)
    驱动包        nvidia-driver-580-open  580.126.09-0ubuntu0.24.04.2
    运行内核      7.0.0-30-generic
    CPU           Intel(R) Core(TM) i7-9700K @ 3.60GHz, 8 核 8 线程
    内存          31 GB

### 1.1 CUDA 不可用：根因与修复

`torch.cuda.is_available()` 返回 `False`，`nvidia-smi` 报
"couldn't communicate with the NVIDIA driver"。GPU 与驱动包都在位，问题是
**驱动内核模块只为另一个内核装了包**：

    /lib/modules/6.17.0-19-generic/kernel/nvidia-580-open/nvidia.ko    存在
    /lib/modules/7.0.0-30-generic/  下无任何 nvidia.ko                 缺失
    lsmod | grep nvidia                                                空
    /dev/nvidia*、/proc/driver/nvidia                                  不存在
    mokutil --sb-state                                                 SecureBoot disabled（不是它）

已装的模块包停在 6.17 系列，HWE 元包也停在 6.17.0-19：

    linux-modules-nvidia-580-open-6.14.0-35/36/37-generic
    linux-modules-nvidia-580-open-6.17.0-14-generic
    linux-modules-nvidia-580-open-6.17.0-19-generic
    linux-modules-nvidia-580-open-generic-hwe-24.04    6.17.0-19.19

即：内核升到 7.0.0-30 之后，对应的 nvidia 模块包没有被拉进来。
这**不是** DKMS 构建失败——`dkms status` 为空，本机走的是预编译 `linux-modules-nvidia-*`
包而不是 DKMS，所以 `dkms autoinstall` 不会有任何作用。

修复只要一条 apt（该包在源里存在，`linux-headers-7.0.0-30-generic` 已装），不需重编、不需重启：

    sudo apt install linux-modules-nvidia-580-open-7.0.0-30-generic
    sudo modprobe nvidia && nvidia-smi

本会话没有 sudo 权限（需要密码），因此未执行。
备选是在 GRUB 选 `6.17.0-19-generic` 重启（模块现成），但那会退回旧内核。

---

## 2. 任务 A：1-epoch smoke

按工单原样执行（**含 `--gpu 0`**，未改命令）：

    cd drl-manager && .venv/bin/python -m timecap_prediction.train_timecap \
      --data-csv timecap_prediction/data/turbines_merged.csv \
      --res-dir timecap_prediction/TimeCAP/model/smoke_3060 \
      --epochs 1 --batch-size 32 --lr 5e-5 --gpu 0

`train_timecap.py:150` 是 `use_gpu and torch.cuda.is_available()`，所以它**静默回落到 CPU**
并继续训练。日志首行即为 `使用 CPU`。这就是验收第一条判 FAIL 的直接证据。

### 2.1 日志尾部（去掉 tqdm 进度条）

    使用 CPU

    === TimeCAP Fine-tune 开始 ===
    数据文件  : timecap_prediction/data/turbines_merged.csv
    seq_len   : 96
    pred_len  : 144
    enc_in    : 13
    features  : MS  (MS: 13路输入，只预测 Patv)
    batch_size: 32
    epochs    : 1
    lr        : 5e-05
    setting   : finetune_TimeCAP_custom_sl96
    Use CPU
    number of model params 23829416

    >>> 开始训练 ...
    train 110139
    val 15626
    test 31393
    [finetune] AMP mixed precision: disabled

    Autoregressive loss: 0.1160
    One-shot loss: 0.8386
    Autoregressive loss: 0.0726
    One-shot loss: 0.5799
    Epoch: 1  Spend: 7101 s | Train Loss: 0.7710206  Vali Loss: 0.6198185  Test Loss: 0.4217501
    Validation loss decreased (inf --> 0.619819).  Saving model of epoch 1
    Updating learning rate to 5e-05

    >>> 开始推理评估 ...
    inference 31393
    Loading checkpoint for inference from .../smoke_3060/TimeCAP/model/finetune_TimeCAP_custom_sl96/ckpt_best.pth

    Test shape: (31393, 144, 1) (31393, 144, 1)
    96-pred-144, MSE: 0.6009, MAE: 0.5649

    评估结果 — MSE: 0.6009  MAE: 0.5649

    model_args.json 已保存到: .../finetune_TimeCAP_custom_sl96/model_args.json

    训练完成！
      Checkpoint : .../smoke_3060/TimeCAP/model/finetune_TimeCAP_custom_sl96/ckpt_best.pth
      Args JSON  : .../smoke_3060/TimeCAP/model/finetune_TimeCAP_custom_sl96/model_args.json

**单 epoch 耗时 7101 秒（约 2 小时）。** GPU 修好后这应当是几分钟量级；这本身就是把驱动补上的理由。

### 2.2 落盘产物

    ckpt_best.pth      sha256  ded8a255159fd56c0c3753c066ac425042a87f8bbb962bfee25861d916d57578
    model_args.json    sha256  70348b059062addcb0096de4c67a9c37b5a3adcd3e44050992155779d63d5515

两者均在 `drl-manager/timecap_prediction/TimeCAP/model/smoke_3060/TimeCAP/model/finetune_TimeCAP_custom_sl96/`。
（checkpoint 体积大，未入 Git；此处以路径 + SHA256 登记。）

### 2.3 predictor 回读

用部署路径 `TimeCAP_GreenPredictor(checkpoint_path=<smoke ckpt>, turbine_csv_paths={12: Turbine_12_2020.csv},
device="cpu")`，`reset()` 后喂满 `seq_len = 96` 步真实历史，再调 `predict()`：

    predict() -> ndarray, shape (144,), dtype float32
    finite: True    min 497.107178   mean 497.109    max 497.111084

形状、dtype、有限性全部符合契约，**回读通过**。
输出近乎常数是 1 epoch 模型的预期表现，**不构成任何关于预测质量的证据**。

### 2.4 验收判定

    CUDA 可用                                  FAIL   （见 §1.1，环境问题，非代码问题）
    checkpoint 与 model_args.json 落盘          PASS
    predictor 能加载并出一次 144 行预测          PASS

GPU 修好后我会按同一条命令重跑，并把 GPU 版日志尾部与 `nvidia-smi` 型号补进本节；
届时本次 CPU 结果只作为"管道通"的旁证保留。

---

## 3. 任务 B：跨机确定性复核

### 3.1 B① 扰动阶梯测试

    .venv/bin/python -m pytest drl-manager/tests/test_forecast_perturb.py -q
    17 passed in 15.29s

**17/17 通过。**

### 3.2 B② 标定复现

以 `timecap_cal.json` 自记的参数重跑（`--stride 480 --label-offset 0 --device cpu`，
五台风机 2020，顺序 12/36/91/95/96）。

先核对输入身份，全部一致：

    checkpoint  fa86c59df99d4fa0228ba07e018bdd399017e5e1f673edc316032a5871a9fb59   与 source_checkpoint_sha 相同
    五个 CSV     与 val_csv_shas 逐个相同（源目录为 windProduction/split/，非 simplified/）

**结果：不是字节一致。**

    committed   g1/compressed_timecap_s2/timecap_cal.json   37701d94eb39ef9f2dd04fbf52034867ea3c18b7d20a3188d1aa0d617ed94478
    repro       默认线程（8 核）                              fd6b6b105990ed9f95c666877700d4e128df11da6a1c01f53592fa86361ec1ea
    repro2      默认线程（8 核），同命令再跑一次               fd6b6b105990ed9f95c666877700d4e128df11da6a1c01f53592fa86361ec1ea
    thr1        OMP_NUM_THREADS=1 MKL_NUM_THREADS=1          0481ba2fd15a403f68431274a9046734efeeeab1bf056e5de641340c76cbb008

三次运行给出三个结果里的两个：**同线程数的两次运行逐字节相同（repro == repro2），
换线程数就变（thr1 不同），而库内文件与三者都不同。**

数值上差异极小：

                          sigma_rel              ar1_rho           lead_alpha        scale_ref
    committed    1.1902660851772053   0.9784656194273471   0.2257271998622998   358.2422345631
    repro        1.1902661796993528   0.9784656240203946   0.2257271847469866   358.2422345631
    thr1         1.1902662024364570   0.9784656245090950   0.2257271932929813   358.2422345631

    repro vs committed   三个标量最大相对偏差 7.94e-08，per_lead_std_rel 最大 2.79e-07
    thr1  vs committed   三个标量最大相对偏差 9.85e-08，per_lead_std_rel 最大 3.13e-07

**根因：CPU 前向的 BLAS 归约顺序随线程数变化，不是语义差异。** 支持这一判断的三条证据：

1. `scale_ref = 358.2422345631115` 在四个文件里**完全相同**。它只由 CSV 的真值算出、
   不经过模型，所以分歧全部来自模型前向。
2. `n_windows = 335`、`stride`、`label_offset`、`turbine_ids`、`source_checkpoint_sha`、
   五个 `val_csv_shas` 在四个文件里**完全相同**——锚点集合、输入、标签对齐都没有变。
3. 固定线程数即可复现（repro == repro2 逐字节相同）。

**对判据的影响。** 只影响"以 SHA 冻结标定产物"这条纪律，不影响标定值本身：
1e-7 的相对偏差远小于 `sigma_rel ≈ 1.19` 这个量级上任何有意义的判据分辨率。
但既然工单要求"产出 `timecap_cal.json`，提交并记 SHA"，就必须让这个 SHA 可复算，否则
它证明不了任何东西。建议二选一，由 5080 侧决定（**我没有单方面改
`residual_calibration.py`**，它是已被阶梯按 SHA 登记的冻结输入）：

- **A（推荐）**：在 `residual_calibration.py` 启动时钉死 `torch.set_num_threads(1)`
  并把线程数写进产物，然后重生成一次 `timecap_cal.json`。此后 SHA 跨机可复算。
- **B**：把冻结纪律从"SHA 逐字节一致"改成"provenance 字段逐字节一致 + 拟合值相对偏差 < 1e-6"，
  并在预注册里写明容差。

在二者之一落定之前，`timecap_cal` 档仍可运行（值是稳的），但**不应声称该产物跨机字节可复现**。

---

## 4. 任务 C：干净数据装载器

新增 `g1/compressed_timecap_s2/clean_dataset.py` 与 `test_clean_dataset.py`（23 测试全绿）。
**只写不训：本轮没有用它启动任何训练。**

`timecap_data_audit.json` 判 `STOP_DATA_PIPELINE` 的两条成因，逐条对应处理：

| 旧管线 | 本模块 |
|---|---|
| 全部风机/年份拼成一个 CSV，`Dataset_Custom` 按行 7:1:2 切，窗口在 split 内自由滑动 → train 段 478 个跨风机窗 | 每风机每年一个文件，**不拼接**；逐文件各自 7:1:2；窗口必须整体落在**一个文件的一个 split** 内 |
| val 起点回拉 seq_len（train 到 110378，val 从 110282）→ 两段共享 96 行 | **取消回拉**，三段行区间严格相邻不重叠 |
| scaler 在拼接文件的 train 段拟合 | scaler **逐文件**在该文件自己的 train 段拟合 |

代价是明说的、不藏着：取消回拉会在每个内部边界少 `seq_len + pred_len - 1` 个候选窗；
逐文件标定意味着每个站点对自己归一化。两条都写在模块开头。

特征列序**从 `predictor.TimeCAP_GreenPredictor.DEFAULT_FEATURE_COLUMNS` 直接 import**，
不手抄——一个和手抄列表一致、却和部署 predictor 不一致的装载器，正是这次审计要抓的东西。

测试覆盖：

    跨文件窗 = 0、跨 split 窗 = 0（三个 split 各自）
    没有窗口混入两台风机（用值域相隔 10000 的合成文件直接验）
    三个 split 的行集合两两不相交；边界相邻且无回拉
    列序与 predictor 逐位一致、共 13 列、Patv 在末列（index 12）
    把 val/test 行改成 1e6 后 scaler 均值不变（证明只看 train 段）
    每个文件各自标定，不共用 scaler；缩放可逆回原始行
    窗口数等于边界算术；split 短于一个窗口时产出 0 个样本
    2022 被拒收；未知 split、缺列、缺文件都会显式报错
    真实 `Turbine_12_2020` 在生产窗口尺寸（96+144）下三个 split 全绿

---

## 5. 未完成 / 待定

1. **CUDA**：等 `sudo apt install linux-modules-nvidia-580-open-7.0.0-30-generic`，
   之后重跑 GPU 版 smoke 并补 §2。
2. **B② 的冻结纪律**：等 5080 侧在 A / B 两案中选一，再决定是否重生成 `timecap_cal.json`。
3. `clean_dataset.py` **不得**用于启动训练——训练要等本机 Stage A′ 判决与独立预注册。
