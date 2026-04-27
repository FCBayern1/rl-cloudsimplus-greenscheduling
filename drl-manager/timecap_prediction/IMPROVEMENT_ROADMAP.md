# TimeCAP 风电预测模型改进路径

## 当前 baseline (Job 4358062)

| 指标 | 数值 |
|---|---|
| 训练 epoch | 30 (best @ 28) |
| Train Loss | 0.2824 |
| Vali Loss | **0.0771** |
| Test Loss | 0.0975 |
| **Test MSE** | **0.2130** |
| **Test RMSE** | **0.4616** |
| **Test MAE** | **0.3193** |
| **Test R²** | **0.7186** |
| 平均 epoch 时长 | 36.1 分钟 |
| 总训练时长 | ~18 小时 |

> 在标准化（z-score）数据上评估，模型解释了 71.9% 的方差。

---

## Phase 1 — 低成本快速提升（预计 R² → 0.75+）  ✅ 已实施

| # | 改动 | 实施方式 | 收益预期 |
|---|---|---|---|
| ② | 时间编码 → **改为输入列**（hour/doy/dow 各 sin+cos = 6 列） | `engineer_features.py` | R² +0.02-0.05 |
| ③ | 风向 sin/cos（3 个角度 → 6 列）+ Wspd³（1 列） | `engineer_features.py` | R² +0.01-0.03 |
| ④ | AMP 混合精度（`autocast` + `GradScaler`） | `exp_TimeCAP.py` finetune/vali | 速度 ~1.5-2× |

> ⚠️ 关键发现：TimeCAP 模型 `forward(batch_x, ...)` **不接收 `batch_x_mark`**，仅 `embed='timeF'` 改不动行为。  
> 因此把时间编码降级为"输入特征列"，与 ③ 合并为单次 CSV 改造。

**enc_in 变化：** 13 → **23**（13 原始 - 3 角度 + 6 sin/cos + 1 Wspd³ + 6 时间）  
**`enc_in` 自动从 CSV 列数推断**，无需手动改 `train_timecap.py`。

### 修改文件清单

| 文件 | 改动 |
|---|---|
| `drl-manager/timecap_prediction/engineer_features.py` | 新建 — CSV 转换脚本 |
| `drl-manager/timecap_prediction/data/turbines_all134_2021_v2.csv` | 新建 — 23 特征 CSV (1.6 GB) |
| `drl-manager/Code/exp/exp_TimeCAP.py` | 添加 AMP（`autocast` + `GradScaler`），由 `args.use_amp` 控制 |
| `drl-manager/timecap_prediction/train_timecap.py` | `use_amp = True` |
| `run_timecap_train.sh` | `--data-csv` 指向 `turbines_all134_2021_v2.csv`；增加 baseline 备份提示 |

### 重跑前操作（保留 baseline）

```bash
cd /lus/lfs1aip2/projects/u6fy/rl-cloudsimplus-greenscheduling/drl-manager
mv timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96 \
   timecap_prediction/TimeCAP/model/finetune_TimeCAP_custom_sl96_baseline_4358062
mv timecap_prediction/TimeCAP/test/finetune_TimeCAP_custom_sl96 \
   timecap_prediction/TimeCAP/test/finetune_TimeCAP_custom_sl96_baseline_4358062
```

### 提交训练

```bash
cd /lus/lfs1aip2/projects/u6fy/rl-cloudsimplus-greenscheduling
sbatch run_timecap_train.sh
```

**单次训练时间估计：** 18h → **9-12h**（AMP 加速）

### ⚠️ 下游兼容性

新模型的 `enc_in=23`，feature 顺序与旧 baseline 不同。  
**RL 调度的 `TimeCAP_GreenPredictor` 需要按新 schema 喂数据**：
1. 推理时输入 CSV 必须用 `engineer_features.py` 处理过
2. 或在 `predictor.py` 里做相同的特征工程
（先跑训练看 R²，确认有收益再处理下游）

---

## Phase 2 — 主力提升（预计 R² → 0.78-0.82）

| # | 改动 | 备注 |
|---|---|---|
| ① | 加入 pretrain 阶段（self-supervised mask 重构） | 设 `load_checkpoints=True` + 先跑 50-100 epoch pretrain |
| ⑤ | 增大全局 batch（4×128 = 512 起） | 配合 LR warmup |
| ⑥ | 学习率调度改 cosine + warmup | 替代当前 `lradj='decay'` |

**总训练时间：** pretrain ~15h + finetune ~10h ≈ 25h

---

## Phase 3 — 架构/超参调优（预计 R² → 0.80-0.85）

| # | 改动 | 备注 |
|---|---|---|
| ⑦ | Multi-horizon 多头输出 (1h/6h/12h/24h) | 短 horizon 的强信号辅助长 horizon |
| ⑧ | 增大模型容量（e_layers 2→4, depth 2→3, dropout 0.1→0.2） | 配合更长训练 |
| ⑨ | Optuna 重扫 lambda 权重 | 当前小数明显是历史调参遗留 |

---

## ⚠️ 数据划分前置确认

train/val/test 是按时间切还是按 turbine 切？
- 按时间切 → 评估结果代表**时间外推**能力
- 按 turbine 切 → 评估结果代表**新风机泛化**能力（更难，R²=0.72 实际很好）

确认命令：
```bash
cd drl-manager
python -c "import pandas as pd; df=pd.read_csv('timecap_prediction/data/turbines_all134_2021.csv'); print('rows:', len(df)); print('date range:', df['date'].min(), '→', df['date'].max())"
```

---

## 实施日志

- **2026-04-27**: baseline 完成 (Job 4358062)，R²=0.7186
- *(下次填)*: Phase 1 完成 → R²=?
