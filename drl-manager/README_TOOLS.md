# DRL Manager - Python 工具使用指南

清理日期: 2025-10-24
清理后保留: **4 个核心工具脚本** + 训练/测试代码

---

## 📁 文件结构（清理后）

```
drl-manager/
├── 📊 analyze_training_complete.py  [34KB]  ← 完整训练分析工具
├── 📈 monitor_success_rate.py       [5.1KB] ← 实时成功率监控
├── 🔧 generate_workload.py          [9.7KB] ← 工作负载生成器
├── ⚙️  setup.py                      [299B]  ← 包安装配置
│
├── 🎯 mnt/                                   ← 训练核心
│   ├── train.py                             ← 训练脚本
│   ├── test.py                              ← 测试脚本
│   ├── entrypoint.py                        ← 入口点
│   ├── callbacks/
│   │   └── save_on_best_training_reward_callback.py
│   └── utils/
│       └── config_loader.py
│
├── 🎮 gym_cloudsimplus/                     ← RL 环境
│   └── envs/
│       └── loadbalancing_env.py
│
└── 🧪 tests/                                ← 测试
    ├── cuda_test.py
    └── gateway_test.py
```

---

## 🚀 工具使用指南

### 1. 完整训练分析 (analyze_training_complete.py)

**功能**: 一站式训练结果分析，生成 7 个可视化文件

**生成内容**:
- ✅ `reward_comparison.png` - 奖励组件对比
- ✅ `energy_comparison.png` - 能耗指标对比
- ✅ `training_curves.png` - 训练曲线
- ✅ `ppo_losses.png` - **PPO Loss 曲线** (Actor, Critic, Entropy)
- ✅ `ppo_training_metrics.png` - **PPO 训练指标** (KL, Clip, Variance)
- ✅ `success_rate_analysis.png` - 成功率分析
- ✅ `training_report.txt` - 文本报告

**使用方法**:
```bash
# 基础分析
drl-manager/.venv/Scripts/python.exe drl-manager/analyze_training_complete.py \
    --log_dir logs/QuickTests/exp3_csv_quick

# 指定快速策略步数阈值
drl-manager/.venv/Scripts/python.exe drl-manager/analyze_training_complete.py \
    --log_dir logs/QuickTests/exp3_csv_quick \
    --fast_strategy_steps 170
```

**输出位置**: `<log_dir>/analysis/`

---

### 2. 实时成功率监控 (monitor_success_rate.py)

**功能**: 训练过程中实时监控成功率和关键指标

**使用方法**:
```bash
# 一次性查看当前状态
drl-manager/.venv/Scripts/python.exe drl-manager/monitor_success_rate.py \
    --log_dir logs/QuickTests/exp3_csv_quick \
    --once

# 持续监控（每 30 秒刷新）
drl-manager/.venv/Scripts/python.exe drl-manager/monitor_success_rate.py \
    --log_dir logs/QuickTests/exp3_csv_quick \
    --interval 30
```

**显示内容**:
- 总 episodes 数
- 快速策略成功率
- 奖励分布统计
- 最近 20 episodes 表现
- 趋势分析

---

### 3. 工作负载生成器 (generate_workload.py)

**功能**: 生成合成工作负载 CSV 文件

**支持模式**:
- `poisson` - 泊松分布到达（随机）
- `uniform` - 均匀分布到达
- `bursty` - 突发式到达

**使用方法**:
```bash
# 生成泊松分布工作负载（300秒，到达率 0.5/s）
drl-manager/.venv/Scripts/python.exe drl-manager/generate_workload.py \
    --mode poisson \
    --arrival_rate 0.5 \
    --duration 300 \
    --output traces/poisson_05_300.csv

# 生成均匀分布工作负载
drl-manager/.venv/Scripts/python.exe drl-manager/generate_workload.py \
    --mode uniform \
    --num_tasks 200 \
    --duration 300 \
    --output traces/uniform_200_300.csv

# 生成突发式工作负载
drl-manager/.venv/Scripts/python.exe drl-manager/generate_workload.py \
    --mode bursty \
    --burst_size 50 \
    --num_bursts 5 \
    --output traces/bursty_50x5.csv
```

---

## 🎯 核心训练代码

### 训练模型
```bash
# 运行指定实验
drl-manager/.venv/Scripts/python.exe drl-manager/mnt/train.py \
    --experiment experiment_3

# 使用自定义配置
drl-manager/.venv/Scripts/python.exe drl-manager/mnt/train.py \
    --experiment experiment_3_debug_v2
```

### 测试模型
```bash
drl-manager/.venv/Scripts/python.exe drl-manager/mnt/test.py \
    --log_dir logs/QuickTests/exp3_csv_quick
```

---

## 🧪 测试工具

### CUDA 测试
```bash
drl-manager/.venv/Scripts/python.exe drl-manager/tests/cuda_test.py
```

### Gateway 连接测试
```bash
drl-manager/.venv/Scripts/python.exe drl-manager/tests/gateway_test.py
```

---

## 📊 典型工作流程

### 1. 训练实验
```bash
# Step 1: 运行训练
drl-manager/.venv/Scripts/python.exe drl-manager/mnt/train.py \
    --experiment experiment_3_debug_v2
```

### 2. 实时监控（可选）
```bash
# Step 2: 在另一个终端监控训练进度
drl-manager/.venv/Scripts/python.exe drl-manager/monitor_success_rate.py \
    --log_dir logs/QuickTests/exp3_debug_reward \
    --interval 30
```

### 3. 训练后分析
```bash
# Step 3: 训练完成后进行完整分析
drl-manager/.venv/Scripts/python.exe drl-manager/analyze_training_complete.py \
    --log_dir logs/QuickTests/exp3_debug_reward \
    --fast_strategy_steps 170

# Step 4: 查看生成的分析结果
# - 打开 logs/QuickTests/exp3_debug_reward/analysis/ 查看图表
# - 阅读 training_report.txt 查看详细报告
```

---

## ⚠️ 常见问题

### Q: 如何更改快速策略的步数阈值？
A: 使用 `--fast_strategy_steps` 参数：
```bash
drl-manager/.venv/Scripts/python.exe drl-manager/analyze_training_complete.py \
    --log_dir logs/QuickTests/exp3_csv_quick \
    --fast_strategy_steps 200  # 改为 200 步
```

### Q: 如何对比多个实验？
A: 对每个实验分别运行分析脚本，然后手动对比 `analysis/` 目录下的图表和报告。

### Q: PPO Loss 曲线在哪里？
A: 运行 `analyze_training_complete.py` 后，查看：
- `<log_dir>/analysis/ppo_losses.png` - Actor/Critic/Entropy Loss
- `<log_dir>/analysis/ppo_training_metrics.png` - KL/Clip/Variance

### Q: 如何生成自定义工作负载？
A: 使用 `generate_workload.py`，详见上方"工作负载生成器"部分。

---

## 📚 相关文档

- **实验配置**: `config.yml` - 所有实验配置
- **调试方案**: `docs/EXP3_DEBUG_PLAN.md` - Experiment 3 详细调试计划
- **快速开始**: `EXPERIMENT3_QUICK_START.md` - Experiment 3 调试快速指南
- **清理记录**: `CLEANUP_PLAN.md` - 代码清理记录

---

## 🗑️ 已删除的冗余文件（2025-10-24）

以下文件已删除，功能已整合到 `analyze_training_complete.py`:
- ❌ `analyze_training.py` - 旧版分析脚本
- ❌ `tutorial_analyze.py` - 教程代码
- ❌ `quick_summary.py` - 硬编码对比
- ❌ `check_trace_order.py` - 一次性工具
- ❌ `fix_trace_order.py` - 一次性工具

**所有功能已保留**，无需担心丢失任何能力。

---

**更新日期**: 2025-10-24
**维护者**: Claude Code
**版本**: v2.0 (清理后)
