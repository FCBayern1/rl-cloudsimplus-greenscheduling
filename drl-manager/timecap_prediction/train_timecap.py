"""
train_timecap.py
================
在风机数据上 fine-tune TimeCAP，用于替换 God's Eye 绿电预测。

任务设定：
    features = 'MS'  → 13路气象特征输入，只预测 Patv（最后一列）
    seq_len  = 96    → 16 小时历史（满足 96%96==0 和 96%24==0）
    pred_len = 144   → 24 小时预测

训练完成后自动保存 model_args.json 到 checkpoint 同目录，
供 TimeCAP_GreenPredictor 自动加载。

用法（从 drl-manager/ 目录运行）：
    # 先跑数据预处理
    python -m timecap_prediction.prepare_turbine_data

    # 再跑训练
    python -m timecap_prediction.train_timecap

    # 可覆盖关键参数
    python -m timecap_prediction.train_timecap \
        --data-csv timecap_prediction/data/turbines_merged.csv \
        --epochs 30 --batch-size 64 --lr 5e-5 --gpu 0
"""

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.distributed as dist

# ── 把 Code/ 加入 sys.path，使 exp/model/utils 可以正常 import ──────────────
_DRLMANAGER_DIR = Path(__file__).resolve().parent.parent
_CODE_DIR = _DRLMANAGER_DIR / "Code"
# Code/ 必须在最前面，避免 drl-manager/ 下同名包（utils/models）抢先被找到
for _p in [str(_DRLMANAGER_DIR), str(_CODE_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
# 确保 Code/ 是 index 0（最高优先级）
if sys.path[0] != str(_CODE_DIR):
    sys.path.remove(str(_CODE_DIR))
    sys.path.insert(0, str(_CODE_DIR))

# Code/ 内的模块（运行时才 import，确保 sys.path 已设置）
from exp.exp_TimeCAP import Exp_TimeCAP          # noqa: E402
from Arguments.load_setting import get_setting_str  # noqa: E402
from utils.tools import set_seed, make_dir, init_logger  # noqa: E402


# ── 默认路径 ────────────────────────────────────────────────────────────────
_DEFAULT_DATA_CSV = str(_DRLMANAGER_DIR / "timecap_prediction" / "data" / "turbines_merged.csv")
_DEFAULT_RES_DIR  = str(_DRLMANAGER_DIR / "timecap_prediction")


def build_args(
    data_csv: str,
    res_dir: str,
    epochs: int,
    batch_size: int,
    lr: float,
    patience: int,
    use_gpu: bool,
    gpu: int,
    use_multi_gpu: bool = False,
    devices: str = "0",
    num_workers: int = 4,
) -> SimpleNamespace:
    """
    构造 TimeCAP 所需的完整 args namespace。
    不使用 argparse，直接在代码里配置，方便修改。
    """
    data_csv_path = Path(data_csv)

    args = SimpleNamespace(
        # ── 任务类型 ──────────────────────────────────────────────
        model            = "TimeCAP",
        paradigm         = "pretrain_finetune",
        task_name        = "finetune",       # 直接 finetune，跳过 pretrain
        downstream_task  = "forecasting",
        is_training      = True,
        load_checkpoints = False,            # 无 pretrain checkpoint，从随机初始化开始
        best_pretrain_path = "",

        # ── 数据 ──────────────────────────────────────────────────
        data             = "custom",         # Dataset_Custom
        root_path        = str(data_csv_path.parent),
        data_path        = data_csv_path.name,
        features         = "MS",             # 13路输入，只预测 Patv
        target           = "Patv",
        freq             = "t",              # minutely（10分钟一行）
        embed            = "fixed",          # data_stamp 全 0，不用时间编码
        percent          = 100,
        augmentation_ratio = 0,
        seasonal_patterns  = "Monthly",      # Dataset_Custom 不用，占位
        drop_last          = False,
        num_workers        = num_workers,
        inverse            = False,

        # ── 序列长度 ──────────────────────────────────────────────
        seq_len          = 96,               # 16h，满足 96%96==0, 96%24==0
        label_len        = 0,
        pred_len         = 144,              # 24h 预测
        pretrain_pred_len = 16,              # AR head 每步预测步数
        mask_rate        = 0.125,

        # ── 模型结构（TimeCAP 默认，与 predictor 里 _DEFAULT_MODEL_CONFIG 一致）──
        enc_in           = 13,               # 13个特征（MS 模式下输入仍是13路）
        d_model          = 736,
        d_ff             = 992,
        e_layers         = 2,
        n_heads          = 8,
        depth            = 2,
        patch_len        = [96, 24],
        stride_time      = [96, 24],
        window_size      = [3, 3],
        stride_channel   = [1, 1],
        scope            = 0,
        dropout          = 0.1,
        activation       = "gelu",
        output_attention = False,
        flash_attention  = False,
        covariate        = False,

        # ── 训练目标权重 ──────────────────────────────────────────
        use_ar_head      = True,
        use_os_head      = True,
        lambda1          = 0.8361538000800285,   # AR loss 权重
        lambda2          = 0.6163727742056744,   # OS loss 权重
        lambda3          = 0.6885377212461313,   # 自蒸馏 loss 权重
        alpha            = 1.0,                  # 推理融合 sigmoid 参数
        beta             = 0.3326362081926146,

        # ── 优化器 & 训练 ─────────────────────────────────────────
        optimizer        = "adam",
        learning_rate    = lr,
        lr_decay         = 0.9,
        lradj            = "decay",
        train_epochs     = epochs,
        batch_size       = batch_size,
        pretrain_batch_size = batch_size,
        patience         = patience,
        use_amp          = False,

        # ── 设备 ──────────────────────────────────────────────────
        use_gpu          = use_gpu and torch.cuda.is_available(),
        gpu              = gpu,
        gpu_type         = "cuda",
        use_multi_gpu    = use_multi_gpu,
        devices          = devices,

        # ── 输出目录 ──────────────────────────────────────────────
        res_dir          = res_dir,
        efficiency       = False,
        seed             = 2024,
        use_dtw          = False,
    )
    return args


def save_model_args(args: SimpleNamespace, checkpoint_path: Path):
    """
    把训练用的 args 序列化为 model_args.json，
    存在 checkpoint 同目录，供 TimeCAP_GreenPredictor 自动加载。
    """
    # 只保存 predictor 需要的字段（SimpleNamespace → dict，过滤不可序列化项）
    keep = {
        "task_name", "downstream_task",
        "seq_len", "label_len", "pred_len", "pretrain_pred_len",
        "enc_in", "features", "target",
        "d_model", "d_ff", "e_layers", "n_heads",
        "depth", "patch_len", "stride_time", "window_size", "stride_channel",
        "scope", "dropout", "activation",
        "output_attention", "flash_attention", "covariate",
        "use_ar_head", "use_os_head",
        "alpha", "beta",
        "lambda1", "lambda2", "lambda3",
    }
    config = {k: v for k, v in vars(args).items() if k in keep}

    json_path = checkpoint_path.parent / "model_args.json"
    with open(json_path, "w") as f:
        json.dump(config, f, indent=2)
    print(f"\nmodel_args.json 已保存到: {json_path}")
    return json_path


def train(args: SimpleNamespace):
    # DDP initialisation — must happen before any CUDA calls
    if args.use_multi_gpu:
        dist.init_process_group(backend='nccl')
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        rank = int(os.environ.get('RANK', 0))
        args.gpu = local_rank  # point device at the correct GPU for this rank
    else:
        rank = 0
    is_main = rank == 0

    set_seed(args.seed)

    # 检查数据文件
    data_file = Path(args.root_path) / args.data_path
    if not data_file.exists():
        if is_main:
            print(f"[ERROR] 数据文件不存在: {data_file}")
            print("请先运行: python -m timecap_prediction.prepare_turbine_data")
        sys.exit(1)

    # 推断 enc_in：读 CSV 头确认实际列数
    import pandas as pd
    df_head = pd.read_csv(data_file, nrows=1)
    # Dataset_Custom 用 columns[1:] 作为特征（除 date 外全部），target 在最后
    actual_enc_in = len(df_head.columns) - 1  # 去掉 date
    if actual_enc_in != args.enc_in:
        print(f"[INFO] 自动修正 enc_in: {args.enc_in} → {actual_enc_in}")
        args.enc_in = actual_enc_in

    # 建立输出目录
    setting = get_setting_str(args)
    test_dir, model_dir, log_dir = make_dir(args)
    logger = init_logger(log_dir)

    # 设备
    if args.use_gpu:
        args.device = torch.device(f"cuda:{args.gpu}")
        if is_main:
            print(f"使用 GPU: cuda:{args.gpu}")
    else:
        args.device = torch.device("cpu")
        if is_main:
            print("使用 CPU")

    if is_main:
        print(f"\n=== TimeCAP Fine-tune 开始 ===")
        print(f"数据文件  : {data_file}")
        print(f"seq_len   : {args.seq_len}")
        print(f"pred_len  : {args.pred_len}")
        print(f"enc_in    : {args.enc_in}")
        print(f"features  : {args.features}  (MS: 13路输入，只预测 Patv)")
        print(f"batch_size: {args.batch_size}")
        print(f"epochs    : {args.train_epochs}")
        print(f"lr        : {args.learning_rate}")
        print(f"setting   : {setting}")

    exp = Exp_TimeCAP(args, logger, model_dir, test_dir, setting)

    if is_main:
        print(f"\n>>> 开始训练 ...")
    exp.finetune()

    # Inference and checkpoint saving only on rank 0
    if is_main:
        print(f"\n>>> 开始推理评估 ...")
        mse, mae = exp.Inference()
        print(f"\n评估结果 — MSE: {mse:.4f}  MAE: {mae:.4f}")

        checkpoint_path = Path(exp.best_checkpoints_path)
        if checkpoint_path.exists():
            json_path = save_model_args(args, checkpoint_path)
            print(f"\n训练完成！")
            print(f"  Checkpoint : {checkpoint_path}")
            print(f"  Args JSON  : {json_path}")
        else:
            print(f"[WARN] checkpoint 未找到: {checkpoint_path}")
            mse, mae = float('nan'), float('nan')
    else:
        mse, mae = float('nan'), float('nan')

    if args.use_multi_gpu and dist.is_initialized():
        dist.destroy_process_group()

    return mse, mae


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune TimeCAP on wind turbine data")
    p.add_argument("--data-csv", default=_DEFAULT_DATA_CSV,
                   help="合并后的训练 CSV 路径（prepare_turbine_data.py 的输出）")
    p.add_argument("--res-dir", default=_DEFAULT_RES_DIR,
                   help="结果输出根目录（checkpoint 会在此目录下）")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--no-gpu", action="store_true", help="强制使用 CPU")
    p.add_argument("--multi-gpu", action="store_true", help="启用 DataParallel 多卡训练")
    p.add_argument("--devices", type=str, default="0", help="多卡时使用的 GPU ID，如 '0,1,2,3'")
    p.add_argument("--num-workers", type=int, default=4, help="DataLoader 并行进程数")
    return p.parse_args()


if __name__ == "__main__":
    cli = parse_args()
    args = build_args(
        data_csv      = cli.data_csv,
        res_dir       = cli.res_dir,
        epochs        = cli.epochs,
        batch_size    = cli.batch_size,
        lr            = cli.lr,
        patience      = cli.patience,
        use_gpu       = not cli.no_gpu,
        gpu           = cli.gpu,
        use_multi_gpu = cli.multi_gpu,
        devices       = cli.devices,
        num_workers   = cli.num_workers,
    )
    train(args)
