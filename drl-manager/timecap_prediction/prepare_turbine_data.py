"""
prepare_turbine_data.py
=======================
把 windProduction/split/ 下的 turbine CSV 转换成 Dataset_Custom 能读的格式，
并把多个 turbine、多个年份合并成一个训练文件。

Dataset_Custom 要求：
    第一列必须叫 date（仅占位，实际 data_stamp 全为 0）
    最后一列是 target（Patv）
    其余中间列是气象特征

原始 split CSV 格式：
    TurbID, Tmstamp, Wspd, Wdir, Etmp, Itmp, Ndir, Pab1, Prtv,
    T2m, Sp, RelH, Wspd_w, Wdir_w, Patv

用法（从 drl-manager/ 目录运行）：
    python -m timecap_prediction.prepare_turbine_data
    python -m timecap_prediction.prepare_turbine_data --turbine-ids 1 15 30 --years 2021 2022
    python -m timecap_prediction.prepare_turbine_data --turbine-ids 1 --output data/turbine_1.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# 默认 split CSV 目录（相对于项目根目录）
_DEFAULT_SPLIT_DIR = (
    Path(__file__).resolve().parent.parent.parent
    / "cloudsimplus-gateway"
    / "src" / "main" / "resources"
    / "windProduction" / "split"
)

# 13 个特征列（顺序与 CSVFeatureLoader / predictor 一致，Patv 最后）
FEATURE_COLS = [
    "Wspd", "Wdir", "Etmp", "Itmp", "Ndir",
    "Pab1", "Prtv", "T2m",
    "Sp", "RelH", "Wspd_w", "Wdir_w",
    "Patv",
]

# RL 实验里用到的 turbine ID（config.yml 里各 DC 使用的）
DEFAULT_TURBINE_IDS = [1, 15, 30, 60, 90, 105, 118, 130]
DEFAULT_YEARS = [2020, 2021, 2022]


def find_csv(split_dir: Path, turbine_id: int, year: int) -> Path | None:
    """定位 Turbine_{id}_{year}.csv，不存在返回 None。"""
    p = split_dir / f"Turbine_{turbine_id}_{year}.csv"
    return p if p.exists() else None


def load_one_csv(path: Path, turbine_id: int, year: int) -> pd.DataFrame | None:
    """
    读取单个 turbine CSV，做列处理：
      - 重命名 Tmstamp → date
      - 删掉 TurbID
      - 确保 FEATURE_COLS 都存在（缺失列填 0）
      - 按 date + FEATURE_COLS 顺序输出
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"  [WARN] 读取失败 {path}: {e}")
        return None

    # 重命名时间列
    if "Tmstamp" in df.columns:
        df = df.rename(columns={"Tmstamp": "date"})
    elif "date" not in df.columns:
        print(f"  [WARN] {path.name} 没有 Tmstamp/date 列，跳过")
        return None

    # 删掉 TurbID
    df = df.drop(columns=["TurbID"], errors="ignore")

    # 补全缺失特征列
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"  [WARN] Turbine {turbine_id} {year}: 缺少列 {missing}，填 0")
        for c in missing:
            df[c] = 0.0

    # 最终列顺序：date + 12个气象特征 + Patv
    df = df[["date"] + FEATURE_COLS]

    # 处理缺失值：先前向填充，再用 0 填剩余
    df[FEATURE_COLS] = df[FEATURE_COLS].ffill().fillna(0.0)

    # Patv 不应该有 NaN，但可以有负值（静止时耗电），保留原始值
    # predictor.predict() 里已经 clip(0) 了

    print(f"  Turbine {turbine_id} {year}: {len(df)} 行，"
          f"Patv=[{df['Patv'].min():.1f}, {df['Patv'].max():.1f}] kW")
    return df


def prepare(
    turbine_ids: list[int],
    years: list[int],
    split_dir: Path,
    output_path: Path,
):
    print(f"\n=== 数据预处理 ===")
    print(f"Turbine IDs : {turbine_ids}")
    print(f"年份        : {years}")
    print(f"split 目录  : {split_dir}")
    print(f"输出文件    : {output_path}\n")

    if not split_dir.exists():
        print(f"[ERROR] split 目录不存在: {split_dir}")
        sys.exit(1)

    frames = []
    missing_files = []

    for tid in turbine_ids:
        for year in years:
            p = find_csv(split_dir, tid, year)
            if p is None:
                missing_files.append(f"Turbine_{tid}_{year}.csv")
                print(f"  [SKIP] Turbine_{tid}_{year}.csv 不存在")
                continue
            df = load_one_csv(p, tid, year)
            if df is not None:
                frames.append(df)

    if not frames:
        print("[ERROR] 没有成功加载任何 CSV 文件。")
        sys.exit(1)

    merged = pd.concat(frames, ignore_index=True)

    # 统计
    total_rows = len(merged)
    n_train = int(total_rows * 0.7)
    n_val   = int(total_rows * 0.1)
    n_test  = total_rows - n_train - n_val
    print(f"\n合并完成：共 {total_rows} 行")
    print(f"  训练集 (70%) : {n_train} 行")
    print(f"  验证集 (10%) : {n_val} 行")
    print(f"  测试集 (20%) : {n_test} 行")
    print(f"  Patv 范围    : [{merged['Patv'].min():.1f}, {merged['Patv'].max():.1f}] kW")
    if missing_files:
        print(f"\n[INFO] 跳过的文件（不存在）: {missing_files}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_path, index=False)
    print(f"\n已保存到: {output_path}")
    return output_path


def parse_args():
    p = argparse.ArgumentParser(description="准备 TimeCAP 风机训练数据")
    p.add_argument(
        "--turbine-ids", nargs="+", type=int, default=DEFAULT_TURBINE_IDS,
        help=f"Turbine ID 列表 (默认: {DEFAULT_TURBINE_IDS})"
    )
    p.add_argument(
        "--years", nargs="+", type=int, default=DEFAULT_YEARS,
        help=f"年份列表 (默认: {DEFAULT_YEARS})"
    )
    p.add_argument(
        "--split-dir", type=str, default=str(_DEFAULT_SPLIT_DIR),
        help="windProduction/split/ 目录路径"
    )
    p.add_argument(
        "--output", type=str,
        default=str(Path(__file__).parent / "data" / "turbines_merged.csv"),
        help="输出 CSV 路径"
    )
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    prepare(
        turbine_ids=args.turbine_ids,
        years=args.years,
        split_dir=Path(args.split_dir),
        output_path=Path(args.output),
    )
