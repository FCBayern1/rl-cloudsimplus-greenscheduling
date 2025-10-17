# 快速参考卡片 - RL-CloudSim Load Balancer

> 🎯 **核心要点**: 使用 `entrypoint.py` 而不是 `train.py`！

---

## ⚡ 最常用命令

### Windows PowerShell

```powershell
# === 启动系统（两个终端窗口）===

# 终端 1: Java Gateway
cd cloudsimplus-gateway
.\gradlew.bat run

# 终端 2: Python 训练
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1

# === 运行实验 ===

# 默认实验（experiment_1）
python .\drl-manager\mnt\entrypoint.py

# 指定实验
$env:EXPERIMENT_ID="experiment_2"
python .\drl-manager\mnt\entrypoint.py

# 自定义配置文件
$env:CONFIG_FILE="my_config.yml"
$env:EXPERIMENT_ID="exp_1"
python .\drl-manager\mnt\entrypoint.py
```

### Linux/Mac

```bash
# 终端 1: Java Gateway
cd cloudsimplus-gateway
./gradlew run

# 终端 2: Python
cd drl-manager
source venv/bin/activate
EXPERIMENT_ID="experiment_1" python mnt/entrypoint.py
```

---

## 📁 项目结构速查

```
rl-cloudsim-loadbalancer/
├── config.yml                    # ⚙️ 主配置文件
├── cloudsimplus-gateway/         # ☕ Java 模拟引擎
│   └── gradlew.bat run          # → 启动 Gateway
└── drl-manager/
    ├── venv/                     # 🐍 Python 虚拟环境
    ├── mnt/
    │   ├── entrypoint.py        # 🚀 主入口（运行这个！）
    │   ├── train.py             # 📦 训练模块（不直接运行）
    │   └── test.py              # 📦 测试模块（不直接运行）
    └── gym_cloudsimplus/         # 🏋️ RL 环境
```

---

## ⚙️ config.yml 快速配置

```yaml
# 基本结构
common:              # 全局默认值
  mode: "train"      # "train" 或 "test"
  algorithm: "PPO"   
  timesteps: 100000
  # ...

experiment_1:        # 实验配置（覆盖 common）
  simulation_name: "My Experiment"
  timesteps: 50000   # 覆盖 common.timesteps
  # ...

experiment_2:
  # ...
```

### 最常改的参数

| 参数 | 用途 | 常用值 |
|------|------|--------|
| `timesteps` | 训练步数 | `1000`（测试）, `50000`（实验）, `500000`（发表） |
| `algorithm` | RL 算法 | `"PPO"`, `"MaskablePPO"`, `"A2C"` |
| `mode` | 运行模式 | `"train"`, `"test"` |
| `initial_s_vm_count` | 初始小型 VM | `5-20` |
| `seed` | 随机种子 | `42`, `4567`, `"random"` |

---

## 🎯 常见任务

### 任务 1: 运行快速测试（5 分钟）

```yaml
# config.yml
experiment_quick:
  timesteps: 1000
  max_episode_length: 50
```

```powershell
$env:EXPERIMENT_ID="experiment_quick"
python .\drl-manager\mnt\entrypoint.py
```

### 任务 2: 训练生产模型（1-2 小时）

```yaml
experiment_production:
  timesteps: 100000
  algorithm: "MaskablePPO"
  save_experiment: true
```

### 任务 3: 测试已训练模型

```yaml
experiment_eval:
  mode: "test"  # 关键！
  train_model_dir: "CSV_Train/Exp1_CSVSimple_Ent_0_01"
  num_eval_episodes: 10
```

### 任务 4: 对比多个算法

```powershell
# 创建 3 个实验配置，然后：
foreach ($algo in @("PPO", "MaskablePPO", "A2C")) {
    $env:EXPERIMENT_ID = "experiment_$algo"
    python entrypoint.py
}
```

---

## 📊 查看结果

### 日志位置

```
logs/
└── {experiment_type_dir}/      # 例如: CSV_Train
    └── {experiment_name}/      # 例如: Exp1_CSVSimple_Ent_0_01
        ├── best_model.zip          # ← 加载这个进行测试
        ├── monitor.csv             # ← Episode 统计
        ├── progress.csv            # ← 训练曲线
        └── current_run.log         # ← 运行日志
```

### TensorBoard

```powershell
tensorboard --logdir=logs
# 浏览器: http://localhost:6006
```

---

## 🐛 问题排查

| 症状 | 原因 | 解决 |
|------|------|------|
| 运行 `train.py` 无输出 | ❌ 错误用法 | ✅ 改用 `entrypoint.py` |
| `config.yml not found` | 路径错误 | 从项目根目录运行 |
| Gateway 连接失败 | Java 未启动 | 先运行 `gradlew run` |
| `ModuleNotFoundError` | 未激活 venv | 运行 `.\venv\Scripts\Activate.ps1` |

---

## 💡 专业提示

1. **始终从项目根目录运行** `python .\drl-manager\mnt\entrypoint.py`
2. **使用环境变量切换实验**，无需编辑代码
3. **先快速测试**（1000 steps），再长时间训练
4. **每次实验前重启 Java Gateway**，避免状态污染
5. **固定随机种子**（`seed: 42`）保证可复现

---

## 🔗 进阶资源

- **详细文档**: `USAGE_GUIDE_CORRECT.md`
- **原始 README**: `README.md`（注意其中 train.py 用法是错误的）
- **配置文件**: `config.yml`
- **示例代码**: `drl-manager/mnt/entrypoint.py`

---

## ✅ 检查清单

运行实验前确保：

- [ ] Java Gateway 正在运行（端口 25333）
- [ ] Python 虚拟环境已激活
- [ ] `config.yml` 中的实验配置正确
- [ ] 环境变量 `EXPERIMENT_ID` 已设置（如果不用默认值）
- [ ] 工作负载文件（trace）存在且可访问

---

**需要帮助？** 查看 `USAGE_GUIDE_CORRECT.md` 或项目日志文件。

