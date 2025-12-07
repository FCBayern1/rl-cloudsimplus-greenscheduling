# RL-CloudSim Load Balancer - 正确使用指南

> ⚠️ **重要说明**: 本文档修正了 README.md 中的错误。实际项目使用 `entrypoint.py` 作为唯一入口，而不是直接运行 `train.py`。

---

## 📖 目录

1. [快速开始](#快速开始)
2. [系统架构](#系统架构)
3. [配置文件详解](#配置文件详解)
4. [运行实验](#运行实验)
5. [常见问题](#常见问题)
6. [最佳实践](#最佳实践)

---

## 🚀 快速开始

### 前置要求

- ✅ Java 11+ 已安装
- ✅ Python 3.12+ 已安装
- ✅ 已完成项目安装和依赖配置

### 5 分钟运行第一个实验

```powershell
# Windows PowerShell

# 1. 启动 Java Gateway（新终端窗口 1）
cd cloudsimplus-gateway
.\gradlew.bat run
# 等待看到: "Starting server: 0.0.0.0 25333"

# 2. 激活 Python 环境并运行训练（新终端窗口 2）
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1

# 3. 运行默认实验
python .\drl-manager\mnt\entrypoint.py

# 完成！查看结果：logs\CSV_Train\Exp1_CSVSimple_Ent_0_01\
```

```bash
# Linux/Mac

# 1. 启动 Java Gateway（终端 1）
cd cloudsimplus-gateway
./gradlew run

# 2. 运行训练（终端 2）
cd drl-manager
source venv/bin/activate
python mnt/entrypoint.py
```

---

## 🏗️ 系统架构

### 实际运行流程

```
用户
  ↓
entrypoint.py (主入口)
  ↓ 读取环境变量
  ↓ • CONFIG_FILE (默认: config.yml)
  ↓ • EXPERIMENT_ID (默认: experiment_1)
  ↓
config_loader.py
  ↓ 加载并合并配置
  ↓ common + experiment_specific
  ↓
根据 mode 字段动态导入
  ↓
  ├─→ train(params) ← train.py (mode="train")
  ├─→ test(params)  ← test.py  (mode="test")
  └─→ ...
```

### 关键组件

| 文件 | 作用 | 命令行调用？ |
|------|------|-------------|
| `entrypoint.py` | **主入口点** | ✅ YES |
| `train.py` | 训练逻辑模块 | ❌ NO（被 entrypoint 调用） |
| `test.py` | 测试逻辑模块 | ❌ NO（被 entrypoint 调用） |
| `config.yml` | 配置文件 | - |

---

## ⚙️ 配置文件详解

### config.yml 结构

```yaml
# 全局默认配置
common:
  mode: "train"              # 运行模式: "train" 或 "test"
  algorithm: "PPO"           # RL 算法
  timesteps: 100000          # 训练步数
  save_experiment: true      # 是否保存结果
  # ... 更多参数

# 实验特定配置（会覆盖 common）
experiment_1:
  simulation_name: "Exp1_CSVSimple_Ent_0_01"
  experiment_name: "Exp1_CSVSimple_Ent_0_01"
  experiment_type_dir: "CSV_Train"
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/three_60max_8maxcores.csv"

experiment_2:
  simulation_name: "Exp2_SWF_LongRun"
  timesteps: 500000
  algorithm: "MaskablePPO"
  # ... 其他配置
```

### 配置加载优先级

```
实验特定配置 > common 配置 > 代码默认值
```

### 重要参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `mode` | 运行模式 | `"train"` | `"train"`, `"test"` |
| `algorithm` | RL 算法 | `"PPO"` | `"PPO"`, `"MaskablePPO"`, `"A2C"` |
| `timesteps` | 训练总步数 | `100000` | `1000`, `50000`, `1000000` |
| `env_id` | Gym 环境 ID | `"LoadBalancingScaling-v0"` | - |
| `save_experiment` | 是否保存日志 | `true` | `true`, `false` |
| `base_log_dir` | 日志根目录 | `"logs"` | `"experiments"` |
| `experiment_type_dir` | 实验类型目录 | `"DefaultType"` | `"CSV_Train"`, `"SWF_Tests"` |
| `experiment_name` | 实验名称 | 同 experiment_id | 任意字符串 |
| `seed` | 随机种子 | `4567` | 任意整数或 `"random"` |

---

## 🎯 运行实验

### 方法 1: 使用环境变量（推荐）

#### Windows PowerShell

```powershell
# 运行 experiment_1
$env:EXPERIMENT_ID="experiment_1"
python .\drl-manager\mnt\entrypoint.py

# 运行 experiment_2
$env:EXPERIMENT_ID="experiment_2"
python .\drl-manager\mnt\entrypoint.py

# 使用自定义配置文件
$env:CONFIG_FILE="custom_config.yml"
$env:EXPERIMENT_ID="experiment_1"
python .\drl-manager\mnt\entrypoint.py
```

#### Linux/Mac Bash

```bash
# 运行 experiment_1
export EXPERIMENT_ID="experiment_1"
python drl-manager/mnt/entrypoint.py

# 或一行命令
EXPERIMENT_ID="experiment_2" python drl-manager/mnt/entrypoint.py
```

### 方法 2: 编辑 config.yml（简单快速）

1. 打开 `config.yml`
2. 编辑现有实验或添加新实验配置
3. 修改参数（如 `timesteps`, `algorithm`, `workload_mode` 等）
4. 直接运行 `python entrypoint.py`（默认使用 `experiment_1`）

#### 示例：创建新实验

```yaml
# config.yml

common:
  # ... 保持不变

experiment_my_test:
  simulation_name: "MyFirstExperiment"
  experiment_name: "my_first_exp"
  experiment_type_dir: "MyTests"
  mode: "train"
  algorithm: "MaskablePPO"
  timesteps: 50000
  initial_s_vm_count: 10  # 减少初始 VM 数量
  initial_m_vm_count: 5
  initial_l_vm_count: 2
```

运行：

```powershell
$env:EXPERIMENT_ID="experiment_my_test"
python .\drl-manager\mnt\entrypoint.py
```

---

## 🧪 完整实验流程示例

### 示例 1: 训练一个新模型

```powershell
# === 步骤 1: 启动 Java Gateway ===
# 终端 1
cd cloudsimplus-gateway
.\gradlew.bat run
# 保持运行，不要关闭

# === 步骤 2: 配置实验 ===
# 编辑 config.yml，添加：
# experiment_baseline:
#   simulation_name: "Baseline_PPO_10k"
#   experiment_name: "baseline_ppo"
#   experiment_type_dir: "Baselines"
#   mode: "train"
#   algorithm: "PPO"
#   timesteps: 10000
#   seed: 42

# === 步骤 3: 运行训练 ===
# 终端 2
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1
$env:EXPERIMENT_ID="experiment_baseline"
python .\drl-manager\mnt\entrypoint.py

# === 步骤 4: 查看结果 ===
# 日志位置: logs\Baselines\baseline_ppo\
# - best_model.zip       # 最佳模型
# - progress.csv         # 训练进度
# - monitor.csv          # Episode 记录
# - events.out.tfevents* # TensorBoard 日志
```

### 示例 2: 测试已训练模型

```powershell
# === 步骤 1: 配置测试实验 ===
# 在 config.yml 添加：
# experiment_test_baseline:
#   mode: "test"  # 关键：设置为 test
#   train_model_dir: "Baselines/baseline_ppo"  # 指向训练结果
#   experiment_name: "baseline_ppo_eval"
#   experiment_type_dir: "Evaluations"
#   num_eval_episodes: 10
#   save_experiment: true

# === 步骤 2: 运行测试 ===
$env:EXPERIMENT_ID="experiment_test_baseline"
python .\drl-manager\mnt\entrypoint.py

# === 步骤 3: 查看评估结果 ===
# logs\Evaluations\baseline_ppo_eval\
# - evaluation_summary.csv   # 每个 episode 的奖励和长度
# - evaluation_details.csv   # 详细的 step 信息
```

### 示例 3: 批量实验

```powershell
# 创建批量运行脚本: run_experiments.ps1

# 实验列表
$experiments = @(
    "experiment_ppo_10k",
    "experiment_ppo_50k",
    "experiment_maskable_10k",
    "experiment_a2c_10k"
)

foreach ($exp in $experiments) {
    Write-Host "=== Running $exp ==="
    
    # 重启 Java Gateway（每次实验前）
    # ... (需要手动或自动化)
    
    # 运行实验
    $env:EXPERIMENT_ID = $exp
    python .\drl-manager\mnt\entrypoint.py
    
    Write-Host "=== Completed $exp ===`n"
}
```

---

## 📊 理解输出结果

### 日志目录结构

```
logs/
└── {experiment_type_dir}/     # 例如: CSV_Train
    └── {experiment_name}/     # 例如: Exp1_CSVSimple_Ent_0_01
        ├── config_used.yml           # 本次运行使用的配置
        ├── seed_used.txt             # 使用的随机种子
        ├── current_run.log           # 当前运行日志
        ├── best_model.zip            # 最佳模型权重
        ├── final_model.zip           # 最终模型权重
        ├── monitor.csv               # Episode 级别统计
        ├── progress.csv              # 训练进度
        ├── best_episode_details_*.csv # 最佳 episode 详情
        ├── events.out.tfevents.*     # TensorBoard 日志
        └── 2025-10-16_14-52/
            └── run.log               # 带时间戳的运行日志
```

### 关键文件说明

#### `monitor.csv`
```csv
r,l,t,reward_wait_time,reward_unutilization,reward_queue_penalty,...
-1041.102,500,8.054,...
```
- `r`: Episode 总奖励
- `l`: Episode 长度（步数）
- `t`: 运行时长（秒）
- 其他列: 各奖励组件

#### `progress.csv`
```csv
rollout/ep_num_monitor,time/iterations,rollout/ep_rew_mean,...
1,,,,,
2,,,,,
,1,2048,71,500.0,-1027.54,28
```
- 每次 PPO 更新后记录一行
- `ep_rew_mean`: 平均 episode 奖励

#### `best_episode_details_*.csv`
- 最佳 episode 的每一步详细信息
- 包含：timestep, clock, action, reward 分解等

---

## 🔧 高级配置

### 调整训练超参数

```yaml
experiment_tuned:
  # === RL 算法参数 ===
  algorithm: "MaskablePPO"  # 推荐用于云调度
  learning_rate: 0.0001      # 降低学习率更稳定
  n_steps: 512               # 每次更新前收集的步数
  batch_size: 128            # 训练批量大小
  n_epochs: 15               # 每次更新的优化轮数
  gamma: 0.995               # 折扣因子
  gae_lambda: 0.98           # GAE λ
  ent_coef: 0.02             # 熵系数（探索）
  
  # === 训练控制 ===
  timesteps: 200000          # 总训练步数
  log_interval: 1            # 每 N 个 episode 记录一次
  save_experiment: true
  
  # === 环境参数 ===
  max_episode_length: 500    # 每个 episode 最大步数
  simulation_timestep: 1.0   # 模拟时间步长
```

### 调整奖励函数权重

```yaml
experiment_reward_tuning:
  # 奖励权重（根据实验调整）
  reward_wait_time_coef: 1.0          # 增加对等待时间的惩罚
  reward_throughput_coef: 0.85        
  reward_unutilization_coef: 0.5      # 增加对资源浪费的惩罚
  reward_cost_coef: 0.5               # 增加对成本的惩罚
  reward_queue_penalty_coef: 0.55     
  reward_invalid_action_coef: 2.0     # 严惩无效动作
```

### 调整基础设施配置

```yaml
experiment_scale_test:
  # === 主机配置 ===
  hosts_count: 64            # 增加物理主机数量
  host_pes: 32               # 每主机核心数
  host_ram: 131072           # 128 GB RAM
  
  # === 初始 VM 配置 ===
  initial_s_vm_count: 10     # 减少初始 VM（让 RL 学习扩展）
  initial_m_vm_count: 5
  initial_l_vm_count: 2
  
  # === VM 规格 ===
  small_vm_pes: 2
  small_vm_ram: 8192
  medium_vm_multiplier: 2    # M = 2 × S
  large_vm_multiplier: 4     # L = 4 × S
```

### 使用不同工作负载

```yaml
# SWF 格式（标准工作负载）
experiment_swf:
  workload_mode: "SWF"
  cloudlet_trace_file: "traces/LLNL-Atlas-2006-2.1-cln.swf"
  max_cloudlets_to_create_from_workload_file: 1000
  workload_reader_mips: 50000

# CSV 格式（自定义工作负载）
experiment_csv:
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/my_custom_workload.csv"
  # CSV 格式: cloudlet_id,arrival_time,length,pes_required,file_size,output_size
```

---

## ❓ 常见问题

### Q1: 为什么直接运行 `python train.py` 没有任何输出？

**A**: `train.py` 不是独立脚本，它只包含一个 `train(params)` 函数。必须通过 `entrypoint.py` 调用：
```powershell
# ❌ 错误
python train.py --timesteps 1000

# ✅ 正确
python entrypoint.py  # 使用 config.yml 配置
```

### Q2: 如何修改训练步数？

**A**: 在 `config.yml` 中修改 `timesteps` 参数：
```yaml
experiment_1:
  timesteps: 50000  # 改为 50000 步
```

### Q3: 如何指定使用哪个实验配置？

**A**: 使用环境变量 `EXPERIMENT_ID`：
```powershell
$env:EXPERIMENT_ID="experiment_2"
python entrypoint.py
```

### Q4: Gateway 连接失败怎么办？

**A**: 
1. 确保 Java Gateway 正在运行（端口 25333）
2. 检查 `cloudsimplus-gateway` 目录下是否有错误
3. 重启 Gateway：
```powershell
# Ctrl+C 停止，然后重新运行
.\gradlew.bat run
```

### Q5: 如何使用 GPU 训练？

**A**: 在配置中设置：
```yaml
experiment_gpu:
  device: "cuda"  # 或 "cuda:0" 指定 GPU
```

### Q6: 训练时间太长怎么办？

**A**: 
- 减少 `timesteps`（例如从 100000 到 10000）
- 减少 `max_episode_length`
- 使用更简单的工作负载
- 减少初始 VM 数量

### Q7: 如何查看训练进度？

**A**: 使用 TensorBoard：
```powershell
tensorboard --logdir=logs
# 浏览器打开 http://localhost:6006
```

### Q8: 默认使用哪个实验配置？

**A**: 如果不指定 `EXPERIMENT_ID`，默认使用 `experiment_1`（在 `entrypoint.py` 中定义）。

---

## 💡 最佳实践

### 1. 实验管理

**组织实验配置**
```yaml
# 按研究目标分组
experiment_baseline_ppo:
  experiment_type_dir: "Baselines"
  # ...

experiment_ablation_no_cost:
  experiment_type_dir: "Ablation"
  reward_cost_coef: 0.0  # 移除成本奖励
  # ...

experiment_production_test:
  experiment_type_dir: "Production"
  # ...
```

**命名规范**
- `experiment_id`: 简短、描述性（例如 `exp_ppo_10k`）
- `experiment_name`: 详细、包含日期/版本（例如 `PPO_10k_v1_20250116`）
- `simulation_name`: 人类可读（例如 `"Baseline PPO 10k steps"`）

### 2. 迭代开发流程

```
1. 快速原型（1000-5000 steps）
   ├─> 验证配置正确
   ├─> 检查奖励设计
   └─> 确认环境稳定

2. 中等规模测试（10000-50000 steps）
   ├─> 观察学习曲线
   ├─> 调整超参数
   └─> 对比不同算法

3. 完整训练（100000+ steps）
   ├─> 最终性能评估
   ├─> 多次运行取平均
   └─> 发表级别实验
```

### 3. 调试技巧

**启用详细日志**
```yaml
experiment_debug:
  verbose: 1  # Stable-Baselines3 详细输出
  log_interval: 1  # 每个 episode 都记录
```

**查看实时日志**
```powershell
# 监控 Python 日志
Get-Content logs\CSV_Train\your_exp\current_run.log -Wait -Tail 50

# 监控 Java 日志
Get-Content cloudsimplus-gateway\logs\cloudsimplus\cspg.current.log -Wait -Tail 50
```

**快速测试配置**
```yaml
experiment_quick_test:
  timesteps: 1000
  max_episode_length: 50
  save_experiment: false  # 不保存结果，快速测试
```

### 4. 性能优化

**减少 Java GC 压力**
```groovy
// cloudsimplus-gateway/build.gradle
applicationDefaultJvmArgs = [
    "-Xmx8g",   // 增加堆内存
    "-Xms2g"
]
```

**Python 加速**
```yaml
experiment_optimized:
  n_steps: 1024  # 减小更频繁更新
  batch_size: 256  # 增大批量提升 GPU 利用率
```

### 5. 结果可复现性

**固定种子**
```yaml
experiment_reproducible:
  seed: 42  # 固定种子
  # Python, NumPy, PyTorch 都会使用此种子
```

**保存完整配置**
- ✅ 系统会自动保存 `config_used.yml` 和 `seed_used.txt`
- ✅ 在论文/报告中引用这些文件

### 6. 多实验对比

**创建对比实验套件**
```yaml
# config_comparison.yml
common:
  timesteps: 50000
  save_experiment: true

experiment_ppo:
  algorithm: "PPO"
  experiment_name: "comparison_ppo"

experiment_maskable_ppo:
  algorithm: "MaskablePPO"
  experiment_name: "comparison_maskable"

experiment_a2c:
  algorithm: "A2C"
  experiment_name: "comparison_a2c"
```

**批量运行脚本**
```powershell
# run_comparison.ps1
$experiments = @("experiment_ppo", "experiment_maskable_ppo", "experiment_a2c")

foreach ($exp in $experiments) {
    Write-Host "Running $exp..."
    $env:CONFIG_FILE = "config_comparison.yml"
    $env:EXPERIMENT_ID = $exp
    python .\drl-manager\mnt\entrypoint.py
}
```

---

## 📞 获取帮助

### 检查日志

1. **Python 日志**: `logs/{experiment_type_dir}/{experiment_name}/current_run.log`
2. **Java 日志**: `cloudsimplus-gateway/logs/cloudsimplus/cspg.current.log`

### 常见错误模式

| 错误信息 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `Configuration file not found` | 在错误目录运行 | 从项目根目录运行 |
| `Gateway connection failed` | Java 未启动 | 启动 `gradlew run` |
| `ModuleNotFoundError` | 虚拟环境未激活 | 激活 venv |
| `Failed to load configuration` | YAML 语法错误 | 检查 config.yml 格式 |
| `PriorityQueue IllegalArgumentException` | 工作负载文件为空 | 检查 trace 文件路径 |

### 社区支持

- GitHub Issues: [项目 Issues 页面]
- 参考完整 README.md 了解更多细节

---

## 🎓 学习路径

### 初学者（第 1-2 天）
1. ✅ 运行默认 `experiment_1`（1000 steps）
2. ✅ 理解 `config.yml` 结构
3. ✅ 查看生成的日志文件
4. ✅ 使用 TensorBoard 可视化

### 进阶用户（第 3-7 天）
1. ✅ 创建自定义实验配置
2. ✅ 调整奖励权重
3. ✅ 对比不同算法（PPO vs MaskablePPO）
4. ✅ 测试已训练模型

### 高级用户（第 8+ 天）
1. ✅ 实现自定义奖励函数
2. ✅ 修改环境观察空间
3. ✅ 集成新的 RL 算法
4. ✅ 发布研究成果

---

## 📝 总结

### 核心要点

1. **唯一入口**: `entrypoint.py`（不是 `train.py`！）
2. **配置方式**: 
   - 方法 1: 环境变量 `EXPERIMENT_ID`
   - 方法 2: 直接编辑 `config.yml`
3. **两阶段运行**:
   - 阶段 1: 启动 Java Gateway
   - 阶段 2: 运行 Python 训练/测试
4. **结果位置**: `logs/{experiment_type_dir}/{experiment_name}/`

### 正确的命令

```powershell
# ✅ 正确
python entrypoint.py

# ✅ 正确（指定实验）
$env:EXPERIMENT_ID="experiment_2"
python entrypoint.py

# ❌ 错误（README 中的错误示例）
python train.py --timesteps 1000
python train.py --config ../config.yml --experiment experiment_drl
```

---

**祝您实验顺利！** 🚀

如有问题，请查看日志文件或提交 Issue。

