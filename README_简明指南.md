# RL-CloudSim 简明使用指南

## 🚀 快速开始（两步）

### 1. 启动 Java 服务器（新终端）

```powershell
cd cloudsimplus-gateway
.\gradlew.bat run
# 等待看到 "Starting server: 0.0.0.0 25333"
```

### 2. 运行实验（另一个终端）

```powershell
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1

# 运行实验（推荐 experiment_5）
$env:EXPERIMENT_ID="experiment_5"
python .\drl-manager\mnt\entrypoint.py
```

---

## ⚙️ 可用的实验

| 实验 | 任务数 | 训练步数 | 时长 | 说明 |
|------|--------|----------|------|------|
| **experiment_5** | 112 | 20,000 | ~15分钟 | ⭐ **推荐先跑** |
| **experiment_7** | 605 | 100,000 | ~2-3小时 | ⭐⭐ 大规模 |
| experiment_6 | 150 | 30,000 | ~30分钟 | 均匀分布 |
| experiment_4 | 165 | 50,000 | ~1小时 | CSV优化 |

---

## 📁 生成的工作负载位置

```
cloudsimplus-gateway/src/main/resources/traces/
├── synthetic_poisson_200s.csv   (112 任务, 泊松分布)  ← Exp 5 用这个
├── synthetic_uniform_300s.csv   (150 任务, 均匀分布)  ← Exp 6
├── synthetic_large_600s.csv     (605 任务, 泊松分布)  ← Exp 7
└── synthetic_bursty_400s.csv    (200 任务, 突发型)    ← Exp 8
```

---

## 📊 工作负载格式（CSV）

```csv
cloudlet_id,arrival_time,length,pes_required,file_size,output_size
0,0,164044,1,164,82              ← 任务0: 时间0秒到达，需要1个核心
1,6,542296,1,542,271             ← 任务1: 时间6秒到达，需要1个核心
2,9,209556,3,209,104             ← 任务2: 时间9秒到达，需要3个核心
...
```

**字段说明**:
- `cloudlet_id`: 任务ID
- `arrival_time`: 到达时间（秒）← 关键！
- `length`: 计算量（MI）
- `pes_required`: 需要的CPU核心数（1-8）
- `file_size`: 输入文件大小（KB）
- `output_size`: 输出文件大小（KB）

---

## 🔧 生成更多工作负载

```bash
cd data-analysis

# 快速生成
python generate_workload.py --type poisson --arrival-rate 1.0 --duration 300 --seed 42

# 完整参数
python generate_workload.py \
  --type poisson \
  --arrival-rate 1.0 \
  --duration 600 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/my_workload.csv \
  --seed 42
```

---

## ⚠️ 重要问题说明

### 原 README 的错误

```bash
# ❌ 错误（README中写的）
python train.py --timesteps 1000

# ✅ 正确
python entrypoint.py
```

### SWF 文件问题（experiment_2）

LLNL-Atlas SWF 文件任务到达时间是**38天后**，所以在200秒的 episode 里：
- ❌ 0 个任务执行
- ❌ 奖励完全相同

**解决**: 用 CSV 格式工作负载（experiment_5-8）

---

## 📖 文档清单（精简后）

| 文档 | 用途 |
|------|------|
| **README_简明指南.md** | 本文，快速开始 ⭐⭐⭐⭐⭐ |
| **QUICK_REFERENCE.md** | 命令速查 ⭐⭐⭐⭐⭐ |
| **USAGE_GUIDE_CORRECT.md** | 完整文档 ⭐⭐⭐⭐ |
| **WORKLOAD_GUIDE.md** | 工作负载详解 ⭐⭐⭐ |

---

## 🎯 现在就开始

**打开两个终端窗口**，按顺序执行：

```powershell
# 终端 1
cd cloudsimplus-gateway
.\gradlew.bat run

# 终端 2（等 Gateway 启动后）
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1
$env:EXPERIMENT_ID="experiment_5"
python .\drl-manager\mnt\entrypoint.py
```

**15分钟后查看结果**:
```powershell
Get-Content logs\Synthetic_Workloads\exp5_poisson_small\monitor.csv
```

---

**就这么简单！** 🚀

