# DRL CloudSim 绿色调度 - 中文使用指南

> 基于深度强化学习的云计算绿色调度框架

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Java](https://img.shields.io/badge/Java-11+-orange.svg)](https://www.oracle.com/java/)
[![Python](https://img.shields.io/badge/Python-3.12+-green.svg)](https://www.python.org/)

---

## 📖 目录

1. [项目概述](#-项目概述)
2. [快速开始](#-快速开始)
   - [单数据中心快速运行](#5分钟快速运行)
   - [🆕 PettingZoo 多数据中心并行训练](#-5分钟快速运行---pettingzoo-多数据中心并行训练推荐)
3. [系统架构](#-系统架构)
4. [强化学习设计](#-强化学习设计)
5. [配置详解](#-配置详解)
6. [训练模型](#-训练模型)
   - [基本训练流程](#基本训练流程)
   - [多数据中心层次化MARL训练](#多数据中心层次化marl训练)
   - [🆕 PettingZoo 并行训练](#-pettingzoo-并行训练推荐---真正的同时执行)
   - [创建自定义实验](#创建自定义实验)
7. [评估模型](#-评估模型)
8. [工作负载管理](#-工作负载管理)
9. [结果分析](#-结果分析)
10. [常见问题](#-常见问题)
11. [最佳实践](#-最佳实践)
12. [快速命令参考](#-快速命令参考)

---

## 🎯 项目概述

本项目是一个基于深度强化学习（DRL）的云计算负载均衡和绿色调度研究框架，通过 CloudSim Plus 模拟器和 Stable-Baselines3 RL库实现。

### 核心特性

- ✅ **绿色调度优化** - 能源效率感知的任务调度
- ✅ **深度强化学习** - 支持 PPO、MaskablePPO、A2C 等算法
- ✅ **多数据中心层次化 MARL** - 全局路由 + 本地调度的两层智能体系统
- ✅ **联合训练框架** - Alternating 和 Simultaneous 两种训练策略
- ✅ **完整的监控和日志** - 实时训练进度、TensorBoard 可视化、自动保存最佳模型
- ✅ **可重复性保证** - 随机种子管理、配置自动保存
- ✅ **灵活的工作负载** - 支持 SWF 和 CSV 格式
- ✅ **完整的实验框架** - 从训练到评估的完整流程
- ✅ **详细的性能分析** - 能源消耗、成本、利用率等多维度指标

### 技术栈

| 组件 | 技术 | 版本 |
|------|------|------|
| 模拟引擎 | CloudSim Plus | 8.5.5 |
| RL框架 | Stable-Baselines3 | 2.7.0 |
| 环境接口 | Gymnasium | 1.2.1 |
| Java-Python桥接 | Py4J | - |
| 深度学习 | PyTorch | 2.6.0+cu124 |

---

## 🚀 快速开始

### 前置要求

- ✅ Java 11+ 已安装并配置 JAVA_HOME
- ✅ Python 3.12+ 已安装
- ✅ NVIDIA GPU（可选，推荐用于加速训练）

### 5分钟快速运行

#### 步骤 1: 启动 Java Gateway

```bash
# Git Bash（推荐）
cd cloudsimplus-gateway
./gradlew build  # 首次运行需要构建
./gradlew run

# 等待看到: "Starting server: 0.0.0.0 25333"
```

```powershell
# PowerShell
cd cloudsimplus-gateway
.\gradlew.bat run
```

#### 步骤 2: 激活 Python 虚拟环境

```bash
# Git Bash
cd drl-manager
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac
```

```powershell
# PowerShell
.\drl-manager\.venv\Scripts\Activate.ps1
```

#### 步骤 3: 运行第一个实验

```bash
# 运行默认实验（experiment_1）
python .\drl-manager\mnt\entrypoint.py

# 或指定实验
export EXPERIMENT_ID="experiment_1"
python .\drl-manager\mnt\entrypoint.py
```

```powershell
# PowerShell
$env:EXPERIMENT_ID="experiment_1"
python .\drl-manager\mnt\entrypoint.py
```

#### 步骤 4: 查看结果

```bash
# 训练日志
cat logs/CSV_Train/Exp1_CSVSimple_Ent_0_01/monitor.csv

# TensorBoard可视化
tensorboard --logdir=logs
# 浏览器打开 http://localhost:6006
```

```powershell
.venv/Scripts/python.exe analyze_training.py --log_dir D:\rl-cloudsimplus-greenscheduling\logs\QuickTests\exp3_csv_quick

.venv/Scripts/python.exe monitor_success_rate.py --log_dir D:\rl-cloudsimplus-greenscheduling\logs\QuickTests\exp3_csv_quick
```

---

### 🆕 5分钟快速运行 - PettingZoo 多数据中心并行训练（推荐）

**适用场景：** 多数据中心层次化 MARL、真正的并行训练、风力预测集成、碳排放优化

**可用实验：**
- `experiment_multi_dc_3` - 3个数据中心（快速测试）
- `experiment_multi_dc_5` - 5个数据中心（完整实验，推荐）⭐

#### 步骤 1: 启动 MultiDC Java Gateway

```bash
# Git Bash（推荐）
cd cloudsimplus-gateway
./gradlew build -x test  # 首次运行需要构建
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 等待看到: "Gateway Server Started on 0.0.0.0:25333"
```

```powershell
# PowerShell
cd cloudsimplus-gateway
.\gradlew.bat build -x test
.\gradlew.bat run "-PappMainClass=giu.edu.cspg.MainMultiDC"
```

#### 步骤 2: 激活 Python 虚拟环境并安装依赖

```bash
# Git Bash
cd drl-manager
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 安装 RLlib 依赖（首次运行）
pip install -r requirements_rllib.txt
```

```powershell
# PowerShell
cd drl-manager
.venv\Scripts\Activate.ps1

# 安装 RLlib 依赖（首次运行）
pip install -r requirements_rllib.txt
```

#### 步骤 3: 运行 PettingZoo 并行训练

##### 🌟 运行实验5（5个数据中心 - 推荐）

```bash
# Git Bash - 方式 A: 使用默认配置（最简单）
python entrypoint_pettingzoo.py

# 方式 B: 运行实验5（5个数据中心 + 碳排放优化）⭐
export EXPERIMENT_ID="experiment_multi_dc_5"
export NUM_WORKERS=0        # Windows建议设为0
export TOTAL_TIMESTEPS=32000
python entrypoint_pettingzoo.py

# 方式 C: 使用命令行参数（完整控制）
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 \
    --total-timesteps 32000 \
    --num-gpus 1

# 方式 D: 快速测试实验5环境
python entrypoint_pettingzoo.py --test --experiment experiment_multi_dc_5
```

```powershell
# PowerShell - 运行实验5（推荐）
$env:EXPERIMENT_ID = "experiment_multi_dc_5"
$env:NUM_WORKERS = 0
$env:TOTAL_TIMESTEPS = 32000
python entrypoint_pettingzoo.py

# 或使用命令行参数
python entrypoint_pettingzoo.py `
    --experiment experiment_multi_dc_5 `
    --num-workers 0 `
    --total-timesteps 32000 `
    --num-gpus 1
```

##### 运行实验3（3个数据中心 - 快速测试）

```bash
# Git Bash
export EXPERIMENT_ID="experiment_multi_dc_3"
export NUM_WORKERS=0
export TOTAL_TIMESTEPS=100000
python entrypoint_pettingzoo.py
```

```powershell
# PowerShell
$env:EXPERIMENT_ID = "experiment_multi_dc_3"
python entrypoint_pettingzoo.py
```

#### 步骤 4: 监控训练进度

```bash
# 查看实时训练日志
tail -f logs/experiment_multi_dc_3_*/current_run.log

# TensorBoard 可视化
tensorboard --logdir=logs/experiment_multi_dc_3_*

# 浏览器打开 http://localhost:6006
```

```powershell
# PowerShell - 查看实时日志
Get-Content logs\experiment_multi_dc_3_*\current_run.log -Wait -Tail 50

# TensorBoard
tensorboard --logdir=logs\experiment_multi_dc_3_*
```

#### PettingZoo 训练特点：

- ✅ **真正的并行执行** - Global 和 Local 智能体同时训练
- ✅ **自动参数共享** - Local 智能体共享神经网络
- ✅ **风力预测集成** - 提供未来 8 步功率预测
- ✅ **Action Masking** - 智能动作掩码
- ✅ **RLlib 框架** - 专业 MARL 训练框架
- ✅ **自动检查点** - 定期保存模型

#### 训练输出位置：

```
logs/experiment_multi_dc_3_<timestamp>/
├── checkpoints/              # RLlib 检查点
├── tensorboard/              # TensorBoard 日志
├── config_used.yml           # 使用的配置
├── seed_used.txt             # 随机种子
└── current_run.log           # 训练日志
```

---

## 🏗️ 系统架构

### 运行流程

```
用户
  ↓
entrypoint.py (主入口)
  ↓
config_loader.py (加载配置)
  ↓
  ├─→ train.py (mode="train") → 训练RL模型
  ├─→ test.py  (mode="test")  → 评估模型
  └─→ ...
  ↓
LoadBalancingEnv (Gymnasium环境)
  ↓
Py4J Gateway (Java-Python通信)
  ↓
CloudSim Plus Simulation (Java)
```

### 关键组件

| 组件 | 文件 | 作用 |
|------|------|------|
| **主入口** | `entrypoint.py` | 唯一的命令行入口点 |
| **训练逻辑** | `train.py` | RL模型训练（由entrypoint调用） |
| **评估逻辑** | `test.py` | 模型评估（由entrypoint调用） |
| **RL环境** | `loadbalancing_env.py` | Gymnasium接口实现 |
| **Java网关** | `LoadBalancerGateway.java` | Py4J服务器 |
| **模拟核心** | `SimulationCore.java` | CloudSim Plus管理 |

### 目录结构

```
rl-cloudsimplus-greenscheduling/
├── 📄 config.yml                          # 主配置文件
├── 📄 docker-compose.yml                  # Docker编排配置
├── 📄 Dockerfile                          # Docker镜像定义
├── 📄 README.md                           # 英文文档
├── 📄 README_CN.md                        # 中文文档（本文件）
├── 📄 WSL_SETUP_COMPLETED.md              # WSL安装指南
│
├── 📁 cloudsimplus-gateway/               # Java模拟引擎
│   ├── 📁 src/main/java/giu/edu/cspg/
│   │   ├── LoadBalancerGateway.java      # 单DC Py4J接口
│   │   ├── MainMultiDC.java              # 🆕 多DC主入口
│   │   ├── SimulationCore.java           # 单DC模拟核心
│   │   ├── MultiDCSimulation.java        # 🆕 多DC模拟核心
│   │   └── ...
│   ├── 📁 src/main/resources/
│   │   └── 📁 traces/                    # 工作负载文件（CSV/SWF）
│   ├── 📁 build/                         # 编译输出
│   ├── 📁 logs/cloudsimplus/             # Java端日志
│   ├── 📁 results/                       # Java端仿真结果
│   ├── build.gradle
│   └── gradlew / gradlew.bat
│
├── 📁 drl-manager/                        # Python RL训练管理
│   ├── 📁 .venv/                         # Python虚拟环境
│   │
│   ├── 📁 gym_cloudsimplus/              # Gymnasium环境
│   │   ├── 📁 envs/
│   │   │   ├── loadbalancing_env.py      # 单DC环境
│   │   │   ├── multidc_routing_env.py    # 🆕 多DC路由环境
│   │   │   ├── multidc_pettingzoo_env.py # 🆕 PettingZoo并行环境
│   │   │   └── ...
│   │   └── 📁 wrappers/                  # 环境包装器
│   │
│   ├── 📁 src/                           # 核心源代码
│   │   ├── 📁 callbacks/                 # 训练回调（日志、检查点）
│   │   ├── 📁 evaluation/                # 评估脚本
│   │   ├── 📁 models/                    # 模型定义
│   │   ├── 📁 networks/                  # 神经网络架构
│   │   ├── 📁 prediction/                # 🆕 风力预测模块
│   │   ├── 📁 training/                  # 训练脚本
│   │   │   ├── train_hierarchical_multidc_joint.py  # 层次化联合训练
│   │   │   ├── train_rllib_multidc.py               # 🆕 RLlib并行训练
│   │   │   └── ...
│   │   └── 📁 utils/                     # 工具函数
│   │
│   ├── 📁 scripts/                       # 辅助脚本
│   │   ├── analyze_training.py           # 训练分析
│   │   ├── generate_workload.py          # 工作负载生成
│   │   ├── monitor_success_rate.py       # 成功率监控
│   │   └── ...
│   │
│   ├── 📁 tests/                         # 单元测试
│   │   ├── test_pettingzoo_wind_prediction.py  # 🆕 PettingZoo测试
│   │   └── ...
│   │
│   ├── 📁 docs/                          # 详细文档
│   │   ├── MARL_REFACTORING_PLAN.md      # MARL重构计划
│   │   ├── MULTI_DC_ARCHITECTURE_GUIDE.md  # 多DC架构指南
│   │   ├── TENSORBOARD_METRICS_GUIDE.md    # TensorBoard指标说明
│   │   └── ...
│   │
│   ├── entrypoint.py                     # 🚀 单DC主入口
│   ├── entrypoint_multidc.py             # 🚀 多DC层次化训练入口
│   ├── entrypoint_pettingzoo.py          # 🚀 PettingZoo并行训练入口
│   ├── requirements_rllib.txt            # RLlib依赖
│   └── setup.py
│
├── 📁 data-analysis/                      # 数据分析
│   ├── analysis.ipynb                    # Jupyter分析笔记本
│   ├── generate_workload.py             # 工作负载生成器
│   ├── 📁 data/                          # 分析数据
│   └── 📁 figs/                          # 可视化图表
│
├── 📁 SWF_Prediction/                     # 🆕 风力预测模块
│   ├── 📁 Data/                          # SDWPF数据集
│   ├── 📁 models/                        # 预测模型（CViTRNN）
│   ├── 📁 scripts/                       # 数据处理脚本
│   └── PROJECT_GUIDE.md
│
├── 📁 logs/                              # 训练日志（自动生成）
│   ├── 📁 CSV_Train/                     # 单DC CSV训练
│   ├── 📁 multi_dc_training/             # 多DC层次化训练
│   ├── 📁 experiment_multi_dc_*/         # PettingZoo并行训练
│   └── ...
│
└── 📁 results/                           # 仿真结果（Java端生成）
```

---

## 🧠 强化学习设计

### State (观察空间)

本项目使用 **Dict** 类型的观察空间，包含以下维度：

```python
observation_space = spaces.Dict({
    "vm_loads": spaces.Box(low=0.0, high=1.0, shape=(num_vms,), dtype=np.float32),
    "vm_available_pes": spaces.Box(low=0, high=8, shape=(num_vms,), dtype=np.int32),
    "waiting_cloudlets": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
    "next_cloudlet_pes": spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32)
})
```

#### State维度说明

| 维度 | 类型 | 范围 | 说明 |
|------|------|------|------|
| `vm_loads` | Box(num_vms,) | [0.0, 1.0] | 每个VM的CPU负载百分比 |
| `vm_available_pes` | Box(num_vms,) | [0, 8] | 每个VM当前可用的PE（核心）数量 |
| `waiting_cloudlets` | Box(1,) | [0, ∞) | 等待队列中的cloudlet数量 |
| `next_cloudlet_pes` | Box(1,) | [0, ∞) | 队列中下一个cloudlet需要的PE数量 |

**示例观察值：**

```python
{
    "vm_loads": [0.75, 0.32, 0.0, 0.88, 0.45, ...],  # 21个VM的负载
    "vm_available_pes": [0, 1, 2, 0, 2, ...],       # 每个VM可用核心数
    "waiting_cloudlets": [5.0],                      # 5个cloudlet等待中
    "next_cloudlet_pes": [2.0]                       # 下一个cloudlet需要2核
}
```

### Action (动作空间)

使用 **Discrete** 类型的动作空间：

```python
action_space = spaces.Discrete(num_vms + 1)
```

#### 动作映射

| Agent输出 | Java端VM ID | 含义 |
|-----------|-------------|------|
| 0 | -1 | **NoOp** - 不分配cloudlet（等待） |
| 1 | 0 | 分配给 **VM 0** |
| 2 | 1 | 分配给 **VM 1** |
| ... | ... | ... |
| n | n-1 | 分配给 **VM n-1** |

**动作含义：**
- Agent在每个时间步选择一个动作（0到num_vms）
- 如果选择1-num_vms，将队列中下一个cloudlet分配给对应的VM
- 如果选择0，则本步不进行分配（等待）
- 如果队列为空但选择了分配动作，视为无效动作并受到惩罚

**示例：**
```python
num_vms = 21  # 系统有21个VM
action_space = Discrete(22)  # 0-21共22个可能的动作

# Agent选择动作
action = 5  # 分配给VM 4（Java端VM ID为4）
action = 0  # 不分配，等待
```

### Reward (奖励函数)

奖励函数采用 **多目标惩罚模型**，由4个负向组件组成。Agent通过**最小化总惩罚**来学习最优策略。

#### 奖励组件

```python
total_reward = reward_wait_time +
               reward_unutilization +
               reward_queue_penalty +
               reward_invalid_action
```

| 组件 | 公式 | 系数 | 目标 |
|------|------|------|------|
| **等待时间惩罚** | `-coef × log(1 + avg_wait_time)` | `reward_wait_time_coef` | 最小化cloudlet等待时间 |
| **利用率惩罚** | `-coef × (√variance + \|avg_util - 0.95\|)` | `reward_unutilization_coef` | 平衡VM负载，接近95%目标利用率 |
| **队列惩罚** | `-coef × (waiting / arrived)` | `reward_queue_penalty_coef` | 减少等待队列长度 |
| **无效动作惩罚** | `-coef × (1 if invalid else 0)` | `reward_invalid_action_coef` | 避免无效动作 |

#### 详细说明

**1. 等待时间惩罚 (Wait Time Penalty)**

```java
avg_wait_time = mean([cloudlet.start_time - cloudlet.arrival_time])
reward_wait_time = -reward_wait_time_coef * log(1 + avg_wait_time)
```

- 惩罚本步完成的cloudlet的平均等待时间
- 使用log函数平滑惩罚，避免极端值
- 默认系数：`0.75`

**2. 利用率惩罚 (Utilization Penalty)**

```java
avg_util = mean([vm.cpu_utilization for vm in vms])
variance = mean([(vm.cpu_util - avg_util)^2 for vm in vms])
deviation = |avg_util - 0.95|  // 目标利用率95%

reward_unutilization = -reward_unutilization_coef * (√variance + deviation)
```

- 同时惩罚两方面：
  - **负载不均衡**：VM间负载方差（鼓励负载均衡）
  - **偏离目标**：平均利用率偏离95%（避免过高或过低）
- 默认系数：`0.25`

**3. 队列惩罚 (Queue Penalty)**

```java
queue_ratio = waiting_cloudlets / arrived_cloudlets
reward_queue_penalty = -reward_queue_penalty_coef * queue_ratio
```

- 惩罚等待队列的相对大小
- 鼓励agent快速处理到达的cloudlet
- 默认系数：`0.55`

**4. 无效动作惩罚 (Invalid Action Penalty)**

```java
invalid = (action == assign && queue_is_empty) ||
          (action == assign && vm_is_full) ||
          (action == noop && queue_has_cloudlets)

reward_invalid_action = -reward_invalid_action_coef * (1 if invalid else 0)
```

- 无效动作包括：
  - 队列为空时尝试分配
  - 分配到已满的VM
  - 有cloudlet等待时选择NoOp
- 默认系数：`1.0`（固定惩罚）

#### 奖励示例

**场景1：良好的分配决策**

```
等待时间：2秒 → -0.75 * log(3) = -0.82
利用率：均值0.90，方差0.01 → -0.25 * (0.1 + 0.05) = -0.04
队列比例：2/10 = 0.2 → -0.55 * 0.2 = -0.11
无效动作：否 → 0
─────────────────────────────────────
总奖励 = -0.97
```

**场景2：差劲的分配决策**

```
等待时间：20秒 → -0.75 * log(21) = -2.28
利用率：均值0.50，方差0.25 → -0.25 * (0.5 + 0.45) = -0.24
队列比例：8/10 = 0.8 → -0.55 * 0.8 = -0.44
无效动作：是 → -1.0
─────────────────────────────────────
总奖励 = -3.96
```

#### 奖励权重调优

可以通过config.yml调整各组件的权重：

```yaml
experiment_custom_reward:
  # 重视性能（低等待时间）
  reward_wait_time_coef: 1.5      # 增加
  reward_queue_penalty_coef: 0.8

  # 重视资源效率
  reward_unutilization_coef: 0.5  # 增加
  reward_cost_coef: 0.8           # 增加

  # 严惩无效动作
  reward_invalid_action_coef: 2.0
```

**调优建议：**
- **高吞吐场景**：增加 `reward_queue_penalty_coef`
- **高效率场景**：增加 `reward_unutilization_coef`
- **低延迟场景**：增加 `reward_wait_time_coef`
- **调试阶段**：增加 `reward_invalid_action_coef` 快速纠正错误行为

### Episode终止条件

**Terminated (自然终止)：**
- 所有cloudlet执行完毕
- 模拟器时间推进结束

**Truncated (强制截断)：**
- 达到最大episode长度：`max_episode_length` (默认500步)
- 配置示例：
  ```yaml
  experiment_long:
    max_episode_length: 1000  # 允许更长的episode
  ```

---

## ⚙️ 配置详解

### config.yml 结构

配置文件采用层次化结构：**实验特定配置** > **common 配置** > **代码默认值**

```yaml
# 全局默认配置
common:
  mode: "train"              # 运行模式: "train" 或 "test"
  algorithm: "PPO"           # RL算法
  timesteps: 100000          # 训练步数
  # ... 更多参数

# 实验特定配置（会覆盖 common）
experiment_1:
  simulation_name: "Exp1_CSVSimple"
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/workload.csv"
  # ... 实验特定参数
```

### 重要参数说明

#### 运行模式

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `mode` | 运行模式 | `"train"` (训练), `"test"` (评估) |
| `algorithm` | RL算法 | `"PPO"`, `"MaskablePPO"`, `"A2C"` |
| `env_id` | 环境ID | `"LoadBalancingScaling-v0"` |

#### 训练参数

| 参数 | 说明 | 默认值 | 建议范围 |
|------|------|--------|----------|
| `timesteps` | 总训练步数 | `100000` | `10000-500000` |
| `learning_rate` | 学习率 | `0.0003` | `0.0001-0.001` |
| `n_steps` | 每次更新步数 | `2048` | `512-4096` |
| `batch_size` | 批量大小 | `64` | `32-256` |
| `gamma` | 折扣因子 | `0.99` | `0.95-0.999` |

#### 环境配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `max_episode_length` | 最大episode长度 | `500` |
| `simulation_timestep` | 模拟时间步长(秒) | `1.0` |
| `hosts_count` | 物理主机数量 | `32` |
| `initial_s_vm_count` | 初始小型VM数量 | `20` |
| `initial_m_vm_count` | 初始中型VM数量 | `10` |
| `initial_l_vm_count` | 初始大型VM数量 | `5` |

#### 工作负载配置

| 参数 | 说明 | 可选值 |
|------|------|--------|
| `workload_mode` | 工作负载格式 | `"SWF"`, `"CSV"` |
| `cloudlet_trace_file` | 工作负载文件路径 | 相对于resources的路径 |

#### 奖励权重

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `reward_wait_time_coef` | 等待时间惩罚系数 | `0.75` |
| `reward_throughput_coef` | 吞吐量奖励系数 | `0.85` |
| `reward_unutilization_coef` | 未利用率惩罚系数 | `0.25` |
| `reward_cost_coef` | 成本惩罚系数 | `0.35` |
| `reward_queue_penalty_coef` | 队列惩罚系数 | `0.55` |

#### 日志配置

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `save_experiment` | 是否保存实验 | `true` |
| `base_log_dir` | 日志根目录 | `"logs"` |
| `experiment_type_dir` | 实验类型子目录 | `"DefaultType"` |
| `log_interval` | 日志记录间隔 | `1` |

---

## 🎓 训练模型

### 基本训练流程

#### 方法 1: 使用环境变量（推荐）

```bash
# 启动 Java Gateway (终端1)
cd cloudsimplus-gateway
./gradlew run

# 运行训练 (终端2)
export EXPERIMENT_ID="experiment_1"
python ./drl-manager/mnt/entrypoint.py
```

```powershell
# PowerShell
$env:EXPERIMENT_ID="experiment_1"
python .\drl-manager\mnt\entrypoint.py
```

#### 方法 2: 修改配置文件

1. 编辑 `config.yml`
2. 修改或添加实验配置
3. 直接运行

```bash
python ./drl-manager/mnt/entrypoint.py  # 默认使用 experiment_1
```

### 多数据中心层次化MARL训练

训练一个层次化多智能体系统，包含全局路由智能体和多个数据中心的本地调度智能体。

#### 方法 1: 使用 entrypoint_multidc.py（推荐）✅

最简单的启动方式，自动管理配置、种子和日志：

```bash
# 步骤 1: 启动 Java Gateway (终端1)
cd cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC          # Bash/Git Bash
.\gradlew.bat run "-PappMainClass=giu.edu.cspg.MainMultiDC"     # PowerShell/Windows

# 步骤 2: 运行 Multi-DC 训练 (终端2)
cd drl-manager
source .venv/bin/activate   # Linux/Mac
.venv\Scripts\activate      # Windows

# 方式 A: 直接运行（使用默认配置）
#$env:EXPERIMENT_ID = "experiment_multi_dc_5"
export EXPERIMENT_ID="experiment_multi_dc_5"

python drl-manager/entrypoint_multidc.py

# 方式 B: 使用环境变量配置
export CONFIG_FILE="../config.yml"
export EXPERIMENT_ID="experiment_multi_dc_5"
export SEED="2025"
python entrypoint_pettingzoo.py
```

```powershell
# PowerShell
$env:CONFIG_FILE = "..\config.yml"
$env:EXPERIMENT_ID = "experiment_multi_dc_5"
$env:SEED = "2025"
python entrypoint_pettingzoo.py
```

#### 方法 2: 直接调用训练脚本

```bash
cd drl-manager
python -m src.training.train_hierarchical_multidc_joint \
    --config ../config.yml \
    --experiment experiment_multi_dc_3 \
    --strategy alternating \
    --seed 2025
```

#### 方法 3: 使用旧版 entrypoint（已过时）

```bash
# 不推荐：旧版 entrypoint.py 调用的是独立训练脚本
export EXPERIMENT_ID="experiment_multi_dc_3"
python entrypoint.py
```

**Multi-DC 特性：**

- **全局智能体（Global Agent）**：将到达的任务路由到最优数据中心，基于：
  - 所有数据中心的总能耗
  - 绿色能源可用性和使用比例
  - 数据中心间的负载均衡
  - 系统总队列长度
- **本地智能体（Local Agents）**：在各数据中心内将任务调度到VM，使用参数共享
- **3个异构数据中心**：
  - DC0: 高性能数据中心（24核/主机，60k MIPS）
  - DC1: 节能型数据中心（16核/主机，50k MIPS）
  - DC2: 边缘数据中心（12核/主机，40k MIPS）
- **独立绿色能源**：每个DC使用不同的风力涡轮机（ID: 57, 58, 59）

**训练配置示例：**

```yaml
experiment_multi_dc_3:
  # === 基础配置 ===
  multi_datacenter_enabled: true  # 启用多数据中心模式
  mode: "train"
  timesteps: 50000                # ✅ 总训练步数（会自动分配到 cycles）
  seed: 2025                      # ✅ 随机种子（或 "random"）
  max_episode_length: 1000        # Episode 最大长度
  save_experiment: true
  verbose: 1
  device: "auto"

  # === 联合训练配置（可选，提供更精细控制）===
  joint_training:
    enabled: true
    strategy: "alternating"       # "alternating" 或 "simultaneous"

    # Alternating 策略参数
    alternating:
      num_cycles: 10                     # 训练周期数
      global_steps_per_cycle: 2500       # 每周期 Global Agent 训练步数
      local_steps_per_cycle: 2500        # 每周期 Local Agent 训练步数
      # 总步数 = num_cycles × (global_steps + local_steps) = 50000

    checkpoint_freq: 5000          # 每 N 步保存检查点
    log_freq: 100                  # 每 N 步记录日志

  # === 数据中心配置 ===
  datacenters:
    - datacenter_id: 0
      name: "DC_HighPerformance"
      hosts_count: 30
      initial_s_vm_count: 20       # 小型 VM 数量
      initial_m_vm_count: 10       # 中型 VM 数量
      initial_l_vm_count: 6        # 大型 VM 数量
      # ... 其他配置

  # === 全局智能体奖励权重 ===
  global_agent:
    reward_total_energy_coef: 2.0      # 最小化总能耗
    reward_green_ratio_coef: 3.0       # 最大化绿色能源使用比例
    reward_load_balance_coef: 1.5      # DC间负载均衡

  # === 本地智能体奖励权重 ===
  local_agents:
    parameter_sharing: true            # 参数共享
    reward_wait_time_coef: 1.0         # 最小化本地等待时间
    reward_utilization_coef: 0.8       # 最大化本地利用率
```

**配置参数说明：**

| 参数 | 说明 | 默认值 | 优先级 |
|------|------|--------|--------|
| `timesteps` | 总训练步数 | 100000 | 如果没有 `joint_training.alternating`，会自动计算 cycles |
| `seed` | 随机种子 | 随机生成 | 命令行 > 配置文件 > 自动生成 |
| `joint_training.alternating.num_cycles` | 训练周期数 | 10 | 最高优先级（如果配置） |
| `joint_training.alternating.global_steps_per_cycle` | 每周期 Global 步数 | 10000 | - |
| `joint_training.alternating.local_steps_per_cycle` | 每周期 Local 步数 | 10000 | - |

**两种配置方式：**

1. **简单模式**：只指定 `timesteps`，系统自动分配
   ```yaml
   timesteps: 50000  # 自动分配为 10 cycles × 2500 步/agent
   ```

2. **精细模式**：完全控制 cycles 和每周期步数
   ```yaml
   joint_training:
     alternating:
       num_cycles: 5
       global_steps_per_cycle: 6000
       local_steps_per_cycle: 4000
   # 总计：5 × (6000 + 4000) = 50000 步
   ```

**监控 Multi-DC 训练：**

```bash
# 查看实时日志
tail -f logs/multi_dc_training/20250110_143022/current_run.log

# TensorBoard 可视化
tensorboard --logdir=logs/multi_dc_training/20250110_143022/tensorboard

# 查看训练进度
cat logs/multi_dc_training/20250110_143022/training_progress.csv

# 查看绿色能源指标（从 monitor 目录）
cat logs/multi_dc_training/20250110_143022/monitor/0.monitor.csv | grep "green_ratio"
```

### 🆕 PettingZoo 并行训练（推荐 - 真正的同时执行）

使用 PettingZoo ParallelEnv 实现所有智能体的**真正并行执行**，与RLlib等高级MARL框架无缝集成。

#### 什么是PettingZoo并行训练？

**对比传统训练方式：**

| 特性 | 顺序训练 (Sequential) | PettingZoo 并行训练 |
|------|---------------------|-------------------|
| **执行方式** | 两阶段轮流训练（先Local后Global） | 所有智能体同时执行 |
| **协同优化** | ❌ 无法协同（一个固定时另一个才能学） | ✅ 真正的协同优化 |
| **收敛速度** | 较慢（需要多次迭代） | 更快（并行学习） |
| **智能体数量** | 1 Global + N Local（分开训练） | 1 Global + N Local（同时训练） |
| **API标准** | Stable-Baselines3 | PettingZoo（兼容RLlib等） |
| **适用场景** | 简单实验、快速原型 | 生产级MARL、复杂协同 |

**并行训练的优势：**
- ✅ **真正的同时执行**：所有智能体在每个时间步同时选择动作
- ✅ **协同优化**：Global和Local智能体可以互相学习对方的策略
- ✅ **框架兼容**：支持RLlib、CleanRL等先进MARL框架
- ✅ **参数共享**：Local智能体自动共享神经网络参数
- ✅ **Action Masking**：完整支持动作掩码

#### 前置准备

1. **安装RLlib依赖（如果使用RLlib训练）**：

```bash
cd drl-manager
.venv/Scripts/python.exe -m pip install -r requirements_rllib.txt
```

2. **配置风力预测（可选但推荐）**：

在 `config.yml` 中启用风力预测：

```yaml
experiment_multi_dc_3:
  # ... 其他配置 ...

  # 风力预测配置
  wind_prediction:
    enabled: true  # 启用风力预测（提供未来8步功率预测）
    model_checkpoint: "D:/SWF_Prediction/CViTRNN/checkpoints/cvit_rnn_best_20241115.pth"
    scalers_path: "D:/SWF_Prediction/CViTRNN/scalers"
    data_path: "D:/SWF_Prediction/Data/sdwpf_baidukddcup2022_full.npz"
    turbine_ids: [1, 57, 124]  # 每个数据中心的风机ID
    turbine_csv_paths:
      1: "D:/SWF_Prediction/Data/by_turbid/turbine_001.csv"
      57: "D:/SWF_Prediction/Data/by_turbid/turbine_057.csv"
      124: "D:/SWF_Prediction/Data/by_turbid/turbine_124.csv"
    horizon: 8  # 预测未来8步（8秒 in COMPRESSED mode）
    device: 'cpu'  # 或 'cuda'
    csv_start_offset: 12  # 时间对齐偏移（不要修改）
```

#### 方法 1: 使用 entrypoint_pettingzoo.py（最简单 ⭐推荐）

最简单的一键启动方式，自动处理所有配置：

**步骤 1: 构建并启动 Java Gateway（终端1）**

```bash
# Git Bash（推荐）
cd cloudsimplus-gateway
./gradlew build -x test  # 首次运行或代码修改后需要构建
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC
```

```powershell
# PowerShell
cd cloudsimplus-gateway
.\gradlew.bat build -x test
.\gradlew.bat run "-PappMainClass=giu.edu.cspg.MainMultiDC"
```

**等待看到：**
```
INFO  [GatewayServer] Gateway Server Started on 0.0.0.0:25333
```

**步骤 2: 启动 PettingZoo 训练（终端2 - 新终端）**

```bash
# Git Bash - 使用默认配置
cd drl-manager
source .venv/Scripts/activate
python entrypoint_pettingzoo.py
```

```powershell
# PowerShell - 使用默认配置
cd drl-manager
.venv\Scripts\Activate.ps1
python entrypoint_pettingzoo.py
```

**进阶用法：**

```bash
# 方式 A: 使用环境变量
export EXPERIMENT_ID="experiment_multi_dc_3"
export NUM_WORKERS=8
export TOTAL_TIMESTEPS=200000
python entrypoint_pettingzoo.py

# 方式 B: 使用命令行参数
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_3 \
    --num-workers 8 \
    --total-timesteps 200000 \
    --num-gpus 1

# 方式 C: 仅测试环境（不训练）
python entrypoint_pettingzoo.py --test
```

```powershell
# PowerShell - 环境变量
$env:EXPERIMENT_ID = "experiment_multi_dc_3"
$env:NUM_WORKERS = 8
$env:TOTAL_TIMESTEPS = 200000
python entrypoint_pettingzoo.py

# PowerShell - 命令行参数
python entrypoint_pettingzoo.py `
    --experiment experiment_multi_dc_3 `
    --num-workers 8 `
    --total-timesteps 200000 `
    --num-gpus 1
```

**entrypoint_pettingzoo.py 特性：**
- ✅ **一键启动**：无需记忆复杂的命令路径
- ✅ **自动配置**：从 config.yml 读取所有设置
- ✅ **智能检查**：启动前验证 Java Gateway 连接
- ✅ **灵活配置**：支持环境变量、命令行参数、配置文件三种方式
- ✅ **测试模式**：`--test` 快速验证环境配置
- ✅ **详细日志**：清晰的进度和错误提示

---

#### 方法 2: 直接调用 train_rllib_multidc.py

如果您需要更多控制，可以直接调用训练脚本：

**步骤 1: 构建并启动 Java Gateway**

```bash
# Git Bash（推荐）
cd cloudsimplus-gateway
./gradlew build -x test  # 首次运行或代码修改后需要构建
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC
```

```powershell
# PowerShell
cd cloudsimplus-gateway
.\gradlew.bat build -x test
.\gradlew.bat run "-PappMainClass=giu.edu.cspg.MainMultiDC"
```

**等待看到：**
```
INFO  [GatewayServer] Gateway Server Started on 0.0.0.0:25333
```

**步骤 2: 启动 RLlib 并行训练（新终端）**

```bash
# Git Bash
cd drl-manager
source .venv/Scripts/activate  # Windows
# source .venv/bin/activate    # Linux/Mac

# 使用RLlib训练
python src/training/train_rllib_multidc.py \
    --config ../config.yml \
    --experiment experiment_multi_dc_3 \
    --num-workers 4 \
    --total-timesteps 100000 \
    --num-gpus 0
```

```powershell
# PowerShell
cd drl-manager
.venv\Scripts\Activate.ps1

# 使用RLlib训练
.venv\Scripts\python.exe src\training\train_rllib_multidc.py `
    --config ..\config.yml `
    --experiment experiment_multi_dc_3 `
    --num-workers 4 `
    --total-timesteps 100000 `
    --num-gpus 0
```

**命令行参数说明：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--config` | 配置文件路径 | `../../config.yml` |
| `--experiment` | 实验名称（config.yml中的key） | `experiment_multi_dc_3` |
| `--num-workers` | 并行工作进程数 | 4 |
| `--total-timesteps` | 总训练步数 | 100000 |
| `--num-gpus` | GPU数量 | 0 |
| `--output-dir` | 输出目录 | 自动生成（带时间戳） |

**步骤 3: 监控训练进度**

训练日志会实时输出，包括：
- ✓ 每个iteration的平均episode reward
- ✓ Global Agent和Local Agents的分别奖励
- ✓ Episode长度统计
- ✓ 自动checkpoint保存

```bash
# 示例输出：
======================================================================
Iteration: 10
Timesteps: 8000 / 100000
Episode reward mean: 156.32
Episode length mean: 245.7
  global_policy reward: 89.45
  local_policy reward: 66.87
======================================================================
```

#### 方法 3: 测试 PettingZoo 环境（不训练）

**方式 A: 使用 entrypoint 测试模式（推荐）**

```bash
cd drl-manager
python entrypoint_pettingzoo.py --test
```

**方式 B: 直接运行测试脚本**

```bash
cd drl-manager
.venv/Scripts/python.exe tests/test_pettingzoo_wind_prediction.py
```

**测试内容：**
1. ✓ 创建PettingZoo环境
2. ✓ 验证风力预测集成
3. ✓ 检查观察空间中的 `dc_predicted_green_power_w`
4. ✓ 运行几个步骤验证功能

**成功输出示例：**
```
✓ Wind prediction is ENABLED in config
✓ Environment created successfully
  Agents: ['global_agent', 'local_agent_0', 'local_agent_1', 'local_agent_2']
✓ Wind predictions found in observation!
  Prediction shape: (3, 8)
  Prediction range: [0.00, 850000.00] W
✓ ALL TESTS PASSED
```

#### PettingZoo 环境的观察空间

在PettingZoo并行环境中，每个智能体都有独立的观察：

**Global Agent 观察：**
```python
observations['global_agent'] = {
    'simulation_time': 120.0,  # 当前仿真时间
    'dc_queue_lengths': [5, 3, 7],  # 各DC等待队列长度
    'dc_active_cloudlets': [12, 8, 15],  # 各DC运行中任务数
    'dc_predicted_green_power_w': [  # 🆕 风力预测（如果启用）
        [120000, 125000, 118000, ...],  # DC 0未来8步预测
        [230000, 235000, 228000, ...],  # DC 1未来8步预测
        [180000, 182000, 179000, ...]   # DC 2未来8步预测
    ],
    # ... 更多全局观察
}
```

**Local Agent 观察（每个DC）：**
```python
observations['local_agent_0'] = {
    'vm_loads': [0.75, 0.32, 0.88, ...],  # VM负载
    'vm_available_pes': [0, 2, 0, ...],   # VM可用核心
    'waiting_cloudlets': 5,
    # ... 更多本地观察
}
```

#### 智能体行动结构

在每个时间步，所有智能体同时提供动作：

```python
actions = {
    'global_agent': np.array([0, 1, 2, 0, 1]),  # 路由5个cloudlet到DC
    'local_agent_0': 3,  # DC 0: 分配到VM 3
    'local_agent_1': 1,  # DC 1: 分配到VM 1
    'local_agent_2': 5   # DC 2: 分配到VM 5
}

observations, rewards, terminations, truncations, infos = env.step(actions)
```

#### 常见问题排查

**Q1: RLlib 导入错误**
```bash
ImportError: cannot import name 'PPOConfig' from 'ray.rllib.algorithms.ppo'
```
**解决方案：**
```bash
pip install -r requirements_rllib.txt  # 安装正确的RLlib版本
```

**Q2: Java Gateway 连接失败**
```
Py4JNetworkError: An error occurred while trying to connect to the Java server
```
**解决方案：**
- 确保 Java Gateway 已启动并显示 "Gateway Server Started"
- 检查端口 25333 是否被占用
- 确认 `config.yml` 中的 `py4j_port: 25333` 与Java端口一致

**Q3: 风力预测数据不出现**
```
✗ Wind predictions NOT found in observation (but enabled in config)
```
**解决方案：**
- 检查 `config.yml` 中 `wind_prediction.enabled: true`
- 确认 `turbine_csv_paths` 路径正确
- 检查 CSV 文件是否存在且格式正确

**Q4: 训练速度慢**
**优化方案：**
- 增加 `--num-workers`（CPU核心数允许的情况下）
- 使用 GPU：`--num-gpus 1`
- 减少 episode 长度：`max_episode_length: 500`

#### 🌟 实验5配置详解（experiment_multi_dc_5）

实验5是一个**大规模5数据中心绿色调度实验**，包含碳排放优化。

##### 数据中心配置

| DC ID | 名称 | 风机ID | 主机数 | 核心/主机 | MIPS | 碳排放因子 (kg CO₂/kWh) |
|-------|------|--------|--------|----------|------|-------------------------|
| 0 | DC_HighPerformance | 1 | 20 | 24 | 60000 | Brown:0.8 / Green:0.01 (煤炭重型) |
| 1 | DC_EnergyEfficient | 57 | 24 | 16 | 50000 | Brown:0.5 / Green:0.01 (天然气) |
| 2 | DC_Edge | 124 | 12 | 12 | 40000 | Brown:0.6 / Green:0.01 (混合电网) |
| 3 | DC_MidRange | 80 | 18 | 20 | 55000 | Brown:0.3 / Green:0.01 (清洁电网) |
| 4 | DC_Regional | 100 | 16 | 18 | 52000 | Brown:0.45 / Green:0.01 (天然气混合) |

**总容量：**
- 总主机数：90台
- 总核心数：1,656核
- 总VM数：154个（初始）

##### 风力数据配置

每个数据中心使用独立的风力涡轮机数据：

```yaml
datacenters:
  - turbine_id: 1
    wind_data_file: "windProduction/split/Turbine_1_2021.csv"
  - turbine_id: 57
    wind_data_file: "windProduction/split/Turbine_57_2021.csv"
  - turbine_id: 124
    wind_data_file: "windProduction/split/Turbine_124_2021.csv"
  - turbine_id: 80
    wind_data_file: "windProduction/split/Turbine_80_2021.csv"
  - turbine_id: 100
    wind_data_file: "windProduction/split/Turbine_100_2021.csv"
```

##### 碳排放优化配置

**碳排放惩罚系数：1000.0** （重要参数）

```yaml
carbon_emission_penalty_coef: 1000.0
```

**作用机制：**
- 总惩罚 = `carbon_emission_penalty_coef × total_carbon_kg`
- 观察值参考：
  - 本地奖励量级：~7,800（负值）
  - 每episode碳排放：~1.0 kg CO₂
  - 系数1000 → 约13%的惩罚影响

**调优建议：**
- 如果碳排放不下降：增加到 1500-2000
- 如果奖励下降太多：减少到 500-800
- 推荐范围：100-2000

##### 工作负载

```yaml
workload_mode: "CSV"
cloudlet_trace_file: "traces/my_uniform.csv"
```

**生成工作负载：**
```bash
cd data-analysis
python generate_workload.py \
  --arrival-rate 1\
  --type uniform \
  --num-jobs 3000 \
  --duration 1000 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/dc10_3000.csv
```

##### 训练配置

```yaml
training:
  total_timesteps: 32000              # 总训练步数（适合5DC规模）
  num_workers: 0                      # Windows建议0（避免DLL问题）
  num_gpus: 1                         # 使用GPU加速（RTX 5080）
  train_batch_size: 4000              # 训练批次大小
  sgd_minibatch_size: 512             # SGD mini-batch（利用RTX 5080）
  num_sgd_iter: 10
  checkpoint_freq_timesteps: 15000    # 每15000步保存检查点
```

##### 运行实验5（完整命令）

```bash
# === 终端1: 启动 Java Gateway ===
cd cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# === 终端2: 运行训练 ===
cd drl-manager
source .venv/Scripts/activate  # Windows

# 运行实验5
export EXPERIMENT_ID="experiment_multi_dc_5"
export NUM_WORKERS=0
export TOTAL_TIMESTEPS=32000
python entrypoint_pettingzoo.py

# 或使用命令行参数
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 \
    --total-timesteps 32000 \
    --num-gpus 1
```

```powershell
# PowerShell
cd drl-manager
.venv\Scripts\Activate.ps1

$env:EXPERIMENT_ID = "experiment_multi_dc_5"
$env:NUM_WORKERS = 0
$env:TOTAL_TIMESTEPS = 32000
python entrypoint_pettingzoo.py
```

##### 监控实验5

```bash
# 实时日志
tail -f logs/experiment_multi_dc_5_*/current_run.log

# TensorBoard（查看碳排放趋势）
tensorboard --logdir="logs/experiment_multi_dc_10_*"
# 浏览器打开 http://localhost:6006
# 查看指标：
#   - global_policy/carbon_emission
#   - global_policy/green_ratio
#   - local_policy/utilization
```

##### 实验5的关键指标

训练过程中重点关注：

1. **碳排放（Carbon Emission）**
   - 目标：逐步下降
   - 典型值：0.5-2.0 kg CO₂/episode

2. **绿色能源比例（Green Ratio）**
   - 目标：逐步上升
   - 理想值：> 60%

3. **负载均衡（Load Balance）**
   - 各DC的队列长度方差
   - 目标：保持较低方差

4. **DC利用率分布**
   - 观察哪些DC被优先使用
   - 是否学会选择低碳排放的DC

##### 预期训练时间

- **单次episode**：约5-15分钟（取决于工作负载）
- **32000步**：约2-4小时（RTX 5080 + num_workers=0）
- **建议**：
  - 首次测试：TOTAL_TIMESTEPS=5000（约30分钟）
  - 完整训练：TOTAL_TIMESTEPS=32000-50000

#### 下一步

训练完成后，可以：
1. **查看训练日志**：`logs/experiment_multi_dc_5_<timestamp>/`
2. **加载最佳模型**：使用RLlib的checkpoint恢复
3. **分析碳排放**：对比训练前后的碳排放变化
4. **评估性能**：编写评估脚本测试训练好的策略
5. **调整超参数**：修改 `carbon_emission_penalty_coef` 和奖励权重

**训练结果目录结构：**

```
logs/multi_dc_training/20250110_143022/
├── config_used.yml              # 使用的配置文件副本
├── seed_used.txt                # 使用的随机种子
├── environment_info.txt         # 环境信息（Python/PyTorch 版本）
├── current_run.log              # 当前运行日志
├── 2025-01-10_14-30/           # 时间戳目录
│   └── run.log                  # 详细运行日志
│
├── monitor/                     # Episode 统计（Stable-Baselines3 Monitor）
│   └── 0.monitor.csv           # Episode 级别指标
│
├── tensorboard/                 # TensorBoard 日志
│   ├── global/                 # Global Agent 训练曲线
│   └── local/                  # Local Agent 训练曲线
│
├── training_progress.csv        # ✅ 训练进度（每个 episode）
├── best_episode_details.csv    # ✅ 最佳 episode 详细数据
│
├── best_global_model.zip        # ⭐ 最佳 Global 模型
├── best_local_model.zip         # ⭐ 最佳 Local 模型
│
├── checkpoints/                 # 定期检查点
│   ├── model_5000_steps.zip
│   ├── model_10000_steps.zip
│   └── ...
│
├── global_cycle_1.zip          # 每个 cycle 的 Global 模型检查点
├── local_cycle_1.zip           # 每个 cycle 的 Local 模型检查点
├── global_cycle_2.zip
├── local_cycle_2.zip
├── ...
│
└── final_global_model.zip      # 最终 Global 模型
    final_local_model.zip        # 最终 Local 模型
```

**训练日志示例：**

```
======================================================================
  Cycle 1/10
======================================================================
Training Global Agent...

Episode 1 completed (Timestep: 512)
  Episode Reward: 142.350
  Global Agent Reward: 85.120
  Local Agent Reward: 57.230
  Mean Reward (last 1 eps): 142.350
  Best Mean Reward: -inf
============================================================
🎉 New best mean reward! Saving models...
  ✅ Saved best global model to best_global_model.zip
  ✅ Saved best local model to best_local_model.zip
  ✅ Saved best episode details to best_episode_details.csv
============================================================

Training Local Agents...
...

======================================================================
  Cycle 2/10
======================================================================
...
```

#### 环境变量配置参数

使用 `entrypoint_multidc.py` 时可配置的环境变量：

| 环境变量 | 说明 | 默认值 | 示例 |
|---------|------|--------|------|
| `CONFIG_FILE` | 配置文件路径 | `config.yml` | `../config.yml` |
| `EXPERIMENT_ID` | 实验 ID | `experiment_multi_dc_3` | `experiment_test` |
| `STRATEGY` | 训练策略 | `alternating` | `alternating` 或 `simultaneous` |
| `SEED` | 随机种子 | 自动生成 | `2025` 或 `random` |
| `OUTPUT_DIR` | 输出目录 | `logs/multi_dc_training` | `../logs/my_exp` |
| `TOTAL_TIMESTEPS` | 总训练步数（覆盖配置） | 从配置文件 | `50000` |

#### 命令行参数（直接调用训练脚本时）

```bash
python -m src.training.train_hierarchical_multidc_joint \
    --config ../config.yml          # 配置文件路径
    --experiment experiment_multi_dc_3  # 实验 ID
    --strategy alternating          # 训练策略
    --seed 2025                     # 随机种子
    --total_timesteps 50000         # 总训练步数
    --output_dir ../logs/my_exp     # 输出目录
```

#### 训练策略说明

**Alternating（交替训练）** - 推荐 ✅
- 先训练 Global Agent，再训练 Local Agent
- 交替进行多个 cycles
- 更稳定，收敛更快
- 适合大多数场景

**Simultaneous（同时训练）** - 实验性 ⚠️
- 在同一 mini-batch 中同时更新两个智能体
- 理论上更快收敛
- 训练不稳定，可能震荡
- 仅用于研究和实验

#### 常见使用场景

**场景 1: 快速测试**
```bash
export TOTAL_TIMESTEPS=5000
export SEED=2025
python entrypoint_multidc.py
```

**场景 2: 批量实验（不同种子）**
```bash
for seed in 2025 2026 2027; do
    export SEED=$seed
    export OUTPUT_DIR="logs/seed_$seed"
    python entrypoint_multidc.py
done
```

**场景 3: 对比不同策略**
```bash
# Alternating
export STRATEGY=alternating
export OUTPUT_DIR=logs/alternating
python entrypoint_multidc.py

# Simultaneous
export STRATEGY=simultaneous
export OUTPUT_DIR=logs/simultaneous
python entrypoint_multidc.py
```

> **提示：** 联合训练会自动读取 `experiment_multi_dc_3` 中的数据中心配置。若更改数据中心规模，只需修改 `config.yml`，环境会自动调整观测空间维度。

### 创建自定义实验

在 `config.yml` 中添加新的实验配置：

```yaml
experiment_my_test:
  # 实验标识
  simulation_name: "MyFirstExperiment"
  experiment_name: "my_first_exp"
  experiment_type_dir: "MyTests"

  # 运行模式
  mode: "train"
  algorithm: "MaskablePPO"

  # 训练参数
  timesteps: 50000
  learning_rate: 0.0001
  n_steps: 1024
  batch_size: 128

  # 环境配置
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/my_workload.csv"
  max_episode_length: 500

  # 初始VM配置
  initial_s_vm_count: 10
  initial_m_vm_count: 5
  initial_l_vm_count: 2

  # 奖励权重调整
  reward_cost_coef: 0.5        # 更重视成本
  reward_wait_time_coef: 1.0   # 更重视等待时间
```

运行：

```bash
export EXPERIMENT_ID="experiment_my_test"
python ./drl-manager/mnt/entrypoint.py
```

### 训练输出

训练过程会生成以下文件：

```
logs/{experiment_type_dir}/{experiment_name}/
├── best_model.zip                    # 最佳模型（验证性能最好）
├── final_model.zip                   # 最终模型（训练结束时）
├── monitor.csv                       # Episode级别统计
├── progress.csv                      # 训练进度
├── events.out.tfevents.*            # TensorBoard日志
├── config_used.yml                   # 本次运行的配置
├── seed_used.txt                     # 使用的随机种子
├── current_run.log                   # 当前运行日志
└── {timestamp}/run.log              # 带时间戳的日志
```

### 监控训练进度

#### 使用 TensorBoard

```bash
tensorboard --logdir=logs

# 浏览器打开 http://localhost:6006
```

可查看：
- Episode奖励曲线
- 策略损失
- 价值函数损失
- 熵值（探索程度）

#### 查看实时日志

```bash
# Git Bash
tail -f logs/MyTests/my_first_exp/current_run.log

# PowerShell
Get-Content logs\MyTests\my_first_exp\current_run.log -Wait -Tail 50
```

### 训练技巧

#### 1. 快速原型验证（5-10分钟）

```yaml
experiment_quick_test:
  timesteps: 1000              # 少量步数
  max_episode_length: 50       # 短episode
  save_experiment: false       # 不保存结果
```

#### 2. 标准实验（1-2小时）

```yaml
experiment_standard:
  timesteps: 50000
  max_episode_length: 500
  save_experiment: true
```

#### 3. 完整训练（3-5小时）

```yaml
experiment_full:
  timesteps: 200000
  max_episode_length: 1000
  save_experiment: true
  device: "cuda"               # 使用GPU加速
```

---

## 🧪 评估模型

### 评估流程

训练完成后，使用以下步骤评估模型性能：

#### 步骤 1: 配置评估实验

在 `config.yml` 中添加评估配置：

```yaml
experiment_eval_my_model:
  # 核心配置
  mode: "test"                                    # 设置为test模式
  train_model_dir: "MyTests/my_first_exp"        # 指向训练结果目录

  # 评估设置
  experiment_name: "eval_my_first_exp"
  experiment_type_dir: "Evaluations"
  num_eval_episodes: 10                          # 评估10个episode

  # 算法必须与训练时一致
  algorithm: "MaskablePPO"

  # 环境配置必须与训练时一致
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/my_workload.csv"
  max_episode_length: 500
  initial_s_vm_count: 10
  initial_m_vm_count: 5
  initial_l_vm_count: 2

  # 其他参数
  save_experiment: true
  device: "auto"
```

#### 步骤 2: 运行评估

```bash
# 确保 Java Gateway 正在运行
cd cloudsimplus-gateway
./gradlew run

# 新终端运行评估
export EXPERIMENT_ID="experiment_eval_my_model"
python ./drl-manager/mnt/entrypoint.py
```

#### 步骤 3: 查看评估结果

```bash
cd logs/Evaluations/eval_my_first_exp/

# 查看评估摘要
cat evaluation_summary.csv
```

### 评估输出文件

```
logs/Evaluations/eval_my_first_exp/
├── evaluation_summary.csv       # 每个episode的总奖励和长度
├── evaluation_details.csv       # 每一步的详细信息
├── config_used.yml              # 评估配置
└── current_run.log              # 评估日志
```

#### evaluation_summary.csv 格式

```csv
episode,reward,length,energy_kwh,energy_wh
1,-150.23,500,2.5432,2543.2
2,-142.56,480,2.4123,2412.3
3,-138.91,490,2.4856,2485.6
...
```

**字段说明：**
- `episode`: Episode编号
- `reward`: Episode总奖励
- `length`: Episode长度（步数）
- `energy_kwh`: 能源消耗（千瓦时）
- `energy_wh`: 能源消耗（瓦时）

### 多模型对比评估

创建多个评估配置对比不同模型：

```yaml
# 评估模型1
experiment_eval_ppo:
  mode: "test"
  train_model_dir: "Experiments/ppo_model"
  experiment_name: "eval_ppo"
  num_eval_episodes: 10
  algorithm: "PPO"

# 评估模型2
experiment_eval_maskable:
  mode: "test"
  train_model_dir: "Experiments/maskable_ppo_model"
  experiment_name: "eval_maskable"
  num_eval_episodes: 10
  algorithm: "MaskablePPO"
```

批量运行：

```bash
for exp in experiment_eval_ppo experiment_eval_maskable; do
    export EXPERIMENT_ID=$exp
    python ./drl-manager/mnt/entrypoint.py
done
```

### 评估最佳实践

1. **环境一致性** - 评估环境必须与训练环境完全一致
2. **足够的样本** - 至少运行10个episode以获得可靠估计
3. **确定性评估** - test.py默认使用 `deterministic=True`（无探索）
4. **不同工作负载** - 在不同工作负载上测试泛化能力
5. **记录完整信息** - 保存 evaluation_details.csv 用于深入分析

---

## 📦 工作负载管理

### 支持的格式

#### 1. CSV 格式（推荐）

简单、灵活、易于生成自定义工作负载。

**文件格式：**

```csv
cloudlet_id,arrival_time,length,pes_required,file_size,output_size
0,0,164044,1,164,82
1,6,542296,1,542,271
2,9,209556,3,209,104
...
```

**字段说明：**

| 字段 | 说明 | 单位 |
|------|------|------|
| `cloudlet_id` | 任务唯一ID | - |
| `arrival_time` | 任务到达时间 | 秒 |
| `length` | 计算量 | MI (Million Instructions) |
| `pes_required` | 需要的CPU核心数 | 1-8 |
| `file_size` | 输入文件大小 | KB |
| `output_size` | 输出文件大小 | KB |

**配置：**

```yaml
experiment_csv:
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/my_workload.csv"
```

#### 2. SWF 格式（标准工作负载）

标准工作负载格式（Standard Workload Format），适合使用真实集群traces。

**配置：**

```yaml
experiment_swf:
  workload_mode: "SWF"
  cloudlet_trace_file: "traces/LLNL-Atlas-2006-2.1-cln.swf"
  max_cloudlets_to_create_from_workload_file: 1000
  workload_reader_mips: 50000
```

**注意：** 某些SWF文件任务到达时间可能很晚（如38天后），导致短episode内无任务执行。推荐使用CSV格式。

### 生成自定义工作负载

#### 使用工作负载生成器

```bash
cd data-analysis

# 泊松分布工作负载（推荐，符合真实场景）
python generate_workload.py \
  --type poisson \
  --arrival-rate 1.0 \
  --duration 600 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/my_poisson.csv \
  --seed 42

# 均匀分布工作负载（用于调试）
python generate_workload.py --type uniform --num-jobs 10000 --duration 2000 --output ../cloudsimplus-gateway/src/main/resources/traces/my_uniform.csv

# 突发型工作负载（压力测试）
python generate_workload.py \
  --type bursty \
  --num-jobs 200 \
  --duration 400 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/my_bursty.csv
```

#### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--type` | 到达模式 | `poisson` |
| `--arrival-rate` | 泊松到达率（任务/秒） | `0.5` |
| `--num-jobs` | 任务总数（uniform/bursty） | `100` |
| `--duration` | 总时长（秒） | `3600` |
| `--length-dist` | 任务长度分布 | `uniform` |
| `--pes-dist` | CPU需求分布 | `weighted` |
| `--output` | 输出文件路径 | - |
| `--seed` | 随机种子 | `None` |

### 工作负载特性

#### Poisson（泊松分布）

- ✅ 符合真实云环境
- ✅ 随机到达，符合统计规律
- ✅ 适合学术研究

**使用场景：** 通用负载均衡研究

#### Uniform（均匀分布）

- ✅ 到达时间均匀
- ✅ 可预测性强
- ✅ 便于调试

**使用场景：** 验证算法正确性、调试

#### Bursty（突发型）

- ✅ 模拟流量突发（如促销活动）
- ⚠️ 挑战性高
- ⚠️ 需要动态扩展能力

**使用场景：** 压力测试、弹性伸缩研究

---

## 📊 结果分析

### Java端结果（详细）

**位置：** `cloudsimplus-gateway/results/{simulation_name}/`

```
cloudsimplus-gateway/results/MyExperiment/
├── cloudlets.csv              # 任务执行详情
├── vms.csv                    # VM统计
├── energy_consumption.csv     # 能源消耗
└── host*.csv                  # 主机历史
```

#### cloudlets.csv（任务执行记录）

```csv
ID,Status,VM ID,ArrivalTime,StartTime,FinishTime,ExecTime,WaitTime,Cost
0,SUCCESS,7,0.0,1.0,4.0,3.0,1.0,$25.53
1,SUCCESS,9,6.0,6.0,17.0,11.0,0.0,$103.91
...
```

**关键字段：**
- `Status`: SUCCESS（完成）/ FAILED（失败）
- `WaitTime`: 等待分配的时间
- `ExecTime`: 实际执行时间
- `Cost`: 该任务的执行成本

#### vms.csv（VM统计）

```csv
ID,Type,Status,PEs,CPU Util %,Cost $
0,S,IDLE,2,0.0,$202.84
9,M,IDLE,4,39.3,$396.68
12,L,IDLE,8,62.5,$784.36
```

**关键指标：**
- `CPU Util %`: 平均利用率（整个episode）
  - 0% = 从未使用（资源浪费）
  - 低于30% = 利用率不足
  - 30-70% = 合理利用
  - 高于70% = 高利用率
- `Cost $`: 总成本（不管是否使用都要付费）

**成本计算公式：**
```
VM成本 = PEs数量 × 单核小时成本 × 运行时间(秒) / 3600
```

#### energy_consumption.csv（能源消耗）

```csv
Summary Statistics
Total Energy Consumption (Wh),2543.2
Total Energy Consumption (kWh),2.5432
Average Power (W),41.5
Peak Power (W),128.3
...

Detailed Host Energy Data
Host ID,Total Energy (Wh),Avg Power (W),Peak Power (W)
0,156.7,12.3,45.6
1,189.4,15.2,52.1
...
```

### Python端结果（训练数据）

**位置：** `logs/{experiment_type_dir}/{experiment_name}/`

#### monitor.csv（Episode总结）

```csv
r,l,t,reward_wait_time,reward_throughput,reward_unutilization,...
-172.32,220,3.52,-0.0,-0.446,-0.125,...
-180.95,220,5.59,-2.77,-0.510,-0.098,...
```

**字段说明：**
- `r`: Episode总奖励
- `l`: Episode长度（步数）
- `t`: 运行时长（秒）
- 其他：各奖励组件

#### progress.csv（训练进度）

PPO每次更新后记录一行：

```csv
rollout/ep_rew_mean,time/iterations,rollout/ep_len_mean,...
-1027.54,1,500.0,...
-985.23,2,498.2,...
```

**关键指标：**
- `ep_rew_mean`: 平均episode奖励（越高越好）
- `ep_len_mean`: 平均episode长度
- `policy_loss`: 策略网络损失
- `value_loss`: 价值网络损失

### 使用Jupyter进行分析

```bash
cd data-analysis
jupyter notebook analysis.ipynb
```

**可进行的分析：**
- 任务完成率随训练的变化
- 能源消耗趋势
- VM利用率分布
- 成本优化效果
- 不同算法对比

### 性能评估指标

#### 任务性能

- **完成率** = 完成任务数 / 总任务数
- **平均等待时间** = Σ(WaitTime) / 完成任务数
- **平均完成时间** = Σ(FinishTime - ArrivalTime) / 完成任务数

#### 资源效率

- **平均CPU利用率** = Σ(VM CPU Util%) / VM数量
- **资源浪费率** = 0%利用率的VM数 / 总VM数

#### 经济性

- **总成本** = ΣVM Cost
- **单任务成本** = 总成本 / 完成任务数

#### 绿色指标

- **总能耗（kWh）** - 从 energy_consumption.csv 读取
- **单任务能耗** = 总能耗 / 完成任务数
- **平均功率（W）** - 系统平均功率消耗

---

## ❓ 常见问题

### 安装与配置

**Q: 如何确定下载的PyTorch版本是CUDA的？**

```bash
python -c "import torch; print(torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

输出应显示：
```
2.6.0+cu124
CUDA: True
```

如果显示 `+cpu` 或 `CUDA: False`，需要重新安装：

```bash
pip uninstall torch torchvision torchaudio -y
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**Q: Java在Git Bash中不可用？**

添加到 `~/.bashrc`:

```bash
export JAVA_HOME="/d/jdk21"  # 根据实际路径调整
export PATH="$JAVA_HOME/bin:$PATH"
```

然后：
```bash
source ~/.bashrc
java -version  # 验证
```

**Q: Gradle wrapper JAR 缺失？**

```bash
cd cloudsimplus-gateway
powershell -Command "Invoke-WebRequest -Uri 'https://raw.githubusercontent.com/gradle/gradle/master/gradle/wrapper/gradle-wrapper.jar' -OutFile 'gradle/wrapper/gradle-wrapper.jar'"
```

### 运行问题

**Q: 为什么直接运行 `python train.py` 没有输出？**

`train.py` 不是独立脚本，必须通过 `entrypoint.py` 调用：

```bash
# ❌ 错误
python train.py --timesteps 1000

# ✅ 正确
python entrypoint.py  # 使用 config.yml 配置
```

**Q: Gateway连接失败？**

1. 确保Java Gateway正在运行（端口25333）
2. 检查防火墙设置
3. 重启Gateway:

```bash
# Ctrl+C 停止
./gradlew run
```

**Q: ModuleNotFoundError: No module named 'yaml'**

```bash
source drl-manager/.venv/Scripts/activate
pip install pyyaml stable-baselines3 tensorboard
```

**Q: 如何指定使用哪个实验配置？**

```bash
export EXPERIMENT_ID="experiment_2"
python entrypoint.py
```

### 性能问题

**Q: 训练时间太长？**

- 减少 `timesteps`（从100000到10000）
- 减少 `max_episode_length`
- 使用更简单的工作负载
- 启用GPU: `device: "cuda"`

**Q: 如何使用GPU训练？**

```yaml
experiment_gpu:
  device: "cuda"  # 或 "cuda:0" 指定GPU
```

验证GPU可用：
```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### 结果问题

**Q: Episode太短，任务没执行完？**

增加 `max_episode_length`:

```yaml
experiment_fix:
  max_episode_length: 1000  # 从500增加到1000
```

**Q: 奖励值完全相同，没有变化？**

可能原因：
1. SWF文件任务到达时间太晚 → 改用CSV格式
2. Episode太短 → 增加 `max_episode_length`
3. 环境配置问题 → 检查Java日志

**Q: 如何查看训练进度？**

```bash
# TensorBoard
tensorboard --logdir=logs

# 实时日志
tail -f logs/MyExp/current_run.log
```

### 实验5（Multi-DC）问题

**Q: 实验5训练时间过长怎么办？**

减少训练步数用于快速测试：
```bash
# 快速测试（约30分钟）
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --total-timesteps 5000 \
    --num-gpus 1
```

**Q: Windows上num_workers应该设置为多少？**

Windows建议设置为0以避免DLL问题：
```bash
export NUM_WORKERS=0  # Linux/Mac可以设置为4-8
```

**Q: 如何调整碳排放惩罚强度？**

修改 `config.yml` 中的系数：
```yaml
experiment_multi_dc_5:
  carbon_emission_penalty_coef: 1000.0  # 默认值
  
  # 如果碳排放不下降，增加到：
  # carbon_emission_penalty_coef: 1500.0 - 2000.0
  
  # 如果奖励下降太多，减少到：
  # carbon_emission_penalty_coef: 500.0 - 800.0
```

**Q: 如何查看碳排放指标？**

```bash
# 在日志中搜索碳排放
grep "carbon" logs/experiment_multi_dc_5_*/current_run.log

# 在TensorBoard中查看
tensorboard --logdir=logs/experiment_multi_dc_5_*
# 打开 http://localhost:6006
# 查看：global_policy/carbon_emission
```

**Q: 实验5需要哪些风力数据文件？**

需要5个风机的CSV文件（已预处理）：
```
cloudsimplus-gateway/src/main/resources/windProduction/split/
├── Turbine_1_2021.csv    # DC 0
├── Turbine_57_2021.csv   # DC 1
├── Turbine_124_2021.csv  # DC 2
├── Turbine_80_2021.csv   # DC 3
└── Turbine_100_2021.csv  # DC 4
```

如果文件缺失，检查：
```bash
ls -lh cloudsimplus-gateway/src/main/resources/windProduction/split/
```

**Q: RLlib安装失败？**

确保安装正确的依赖：
```bash
cd drl-manager
pip install -r requirements_rllib.txt

# 如果仍有问题，手动安装：
pip install ray[rllib]==2.9.0
pip install pettingzoo==1.24.3
```

**Q: 训练中断如何恢复？**

RLlib会自动保存检查点：
```bash
# 查看检查点
ls -lh logs/experiment_multi_dc_5_*/checkpoints/

# 恢复训练（功能开发中）
# TODO: 添加checkpoint恢复功能
```

**Q: 如何对比不同碳排放系数的效果？**

运行多个实验并对比：
```bash
# 实验A：低惩罚
# 修改config.yml: carbon_emission_penalty_coef: 500
export OUTPUT_DIR="logs/carbon_500"
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5

# 实验B：高惩罚
# 修改config.yml: carbon_emission_penalty_coef: 2000
export OUTPUT_DIR="logs/carbon_2000"
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5

# 对比TensorBoard
tensorboard --logdir=logs/
```

---

## 💡 最佳实践

### 实验设计

#### 1. 迭代开发流程

```
阶段1: 快速原型（1000-5000 steps）
  ├─> 验证配置正确
  ├─> 检查奖励设计
  └─> 确认环境稳定

阶段2: 中等规模测试（10000-50000 steps）
  ├─> 观察学习曲线
  ├─> 调整超参数
  └─> 对比不同算法

阶段3: 完整训练（100000+ steps）
  ├─> 最终性能评估
  ├─> 多次运行取平均
  └─> 发表级别实验
```

#### 2. 实验命名规范

```yaml
# 清晰的命名
experiment_ppo_50k_poisson:
  simulation_name: "PPO_50k_Poisson_v1"
  experiment_name: "ppo_50k_poisson_20250120"
  experiment_type_dir: "Algorithm_Comparison"
```

**建议：**
- `experiment_id`: 简短、描述性
- `experiment_name`: 详细、包含日期/版本
- `simulation_name`: 人类可读

#### 3. 组织实验配置

```yaml
# 按研究目标分组
experiment_baseline_ppo:
  experiment_type_dir: "Baselines"
  # ...

experiment_ablation_no_cost:
  experiment_type_dir: "Ablation_Studies"
  reward_cost_coef: 0.0  # 移除成本奖励
  # ...

experiment_production:
  experiment_type_dir: "Production_Tests"
  # ...
```

### 超参数调优

#### 学习率调整

```yaml
# 保守学习（更稳定）
experiment_conservative:
  learning_rate: 0.0001

# 标准学习
experiment_standard:
  learning_rate: 0.0003

# 激进学习（可能更快但不稳定）
experiment_aggressive:
  learning_rate: 0.001
```

#### 批量大小

```yaml
# 小批量（更新频繁，但噪声大）
experiment_small_batch:
  batch_size: 32
  n_steps: 512

# 大批量（更新稳定，但速度慢）
experiment_large_batch:
  batch_size: 256
  n_steps: 4096
```

### 奖励函数设计

```yaml
# 重视性能
experiment_performance_focused:
  reward_wait_time_coef: 1.5
  reward_throughput_coef: 1.2
  reward_cost_coef: 0.2

# 重视成本
experiment_cost_focused:
  reward_wait_time_coef: 0.5
  reward_throughput_coef: 0.8
  reward_cost_coef: 1.0

# 平衡型
experiment_balanced:
  reward_wait_time_coef: 0.75
  reward_throughput_coef: 0.85
  reward_cost_coef: 0.5
```

### 可复现性

#### 固定随机种子

```yaml
experiment_reproducible:
  seed: 42  # 固定种子
```

#### 保存完整配置

系统会自动保存：
- `config_used.yml` - 实际使用的配置
- `seed_used.txt` - 使用的随机种子

在论文/报告中引用这些文件以确保可复现。

### 性能优化

#### JVM优化

编辑 `cloudsimplus-gateway/build.gradle`:

```groovy
applicationDefaultJvmArgs = [
    "-Xmx16g",                   // 最大堆内存16GB
    "-Xms4g",                    // 初始堆内存4GB
    "-XX:+UseG1GC",              // 使用G1垃圾收集器
    "-server"                    // 服务器模式
]
```

#### Python优化

```yaml
experiment_optimized:
  n_steps: 1024       // 减小以更频繁更新
  batch_size: 256     // 增大以提升GPU利用率
  device: "cuda"      // 使用GPU
```

### 调试技巧

#### 启用详细日志

```yaml
experiment_debug:
  verbose: 1           // SB3详细输出
  log_interval: 1      // 每个episode都记录
```

#### 监控实时日志

```bash
# Git Bash
tail -f logs/MyExp/current_run.log

# PowerShell
Get-Content logs\MyExp\current_run.log -Wait -Tail 50
```

#### 快速测试配置

```yaml
experiment_quick_test:
  timesteps: 1000
  max_episode_length: 50
  save_experiment: false  # 不保存，快速测试
```

---

## 📞 获取帮助

### 日志位置

1. **Python日志**: `logs/{experiment_type_dir}/{experiment_name}/current_run.log`
2. **Java日志**: `cloudsimplus-gateway/logs/cloudsimplus/cspg.current.log`

### 错误排查

| 错误信息 | 可能原因 | 解决方法 |
|---------|---------|---------|
| `Configuration file not found` | 路径错误 | 从项目根目录运行 |
| `Gateway connection failed` | Java未启动 | 先运行 `gradlew run` |
| `ModuleNotFoundError` | 未激活venv | 激活虚拟环境 |
| `Failed to load configuration` | YAML语法错误 | 检查config.yml格式 |
| `Model spaces mismatch` | 环境配置不一致 | 评估时使用训练时的配置 |

### 文档资源

- **快速参考**: `QUICK_REFERENCE.md` - 常用命令速查
- **英文文档**: `README.md` - 完整英文文档
- **本文档**: `README_CN.md` - 中文完整指南

### 社区支持

- GitHub Issues: 报告bug或提问
- 项目文档: 查看详细说明

---

## 📝 快速命令参考

### 单数据中心训练（传统方式）

```bash
# === 启动系统 ===
# Java Gateway（单DC）
cd cloudsimplus-gateway && ./gradlew run

# Python训练（单DC）
export EXPERIMENT_ID="experiment_1"
python ./drl-manager/entrypoint.py

# === 查看结果 ===
# TensorBoard
tensorboard --logdir=logs

# 查看日志
tail -f logs/MyExp/current_run.log

# 查看评估结果
cat logs/Evaluations/my_eval/evaluation_summary.csv

# === 生成工作负载 ===
cd data-analysis
python generate_workload.py --type poisson --duration 600 --output ../cloudsimplus-gateway/src/main/resources/traces/my_workload.csv
```

### 🆕 多数据中心 PettingZoo 并行训练（推荐）

```bash
# === 启动 MultiDC Java Gateway ===
cd cloudsimplus-gateway
./gradlew build -x test
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# === 安装 RLlib 依赖（首次运行）===
cd drl-manager
source .venv/Scripts/activate  # Windows: .venv/Scripts/activate
pip install -r requirements_rllib.txt

# === 运行实验5（5个数据中心 + 碳排放优化）⭐推荐 ===
# 方式 1: 使用环境变量（最简单）
export EXPERIMENT_ID="experiment_multi_dc_5"
export NUM_WORKERS=0        # Windows建议设为0
export TOTAL_TIMESTEPS=32000
python entrypoint_pettingzoo.py

# 方式 2: 使用命令行参数（完整控制）
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 \
    --total-timesteps 32000 \
    --num-gpus 1

# 方式 3: 快速测试（5000步，约30分钟）
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 \
    --total-timesteps 5000 \
    --num-gpus 1

# === 运行实验3（3个数据中心，快速测试）===
export EXPERIMENT_ID="experiment_multi_dc_3"
export TOTAL_TIMESTEPS=100000
python entrypoint_pettingzoo.py

# === 仅测试环境（不训练）===
python entrypoint_pettingzoo.py --test --experiment experiment_multi_dc_5

# === 监控训练（实验5）===
# 实时日志
tail -f logs/experiment_multi_dc_5_*/current_run.log

# TensorBoard 可视化（查看碳排放、绿色能源比例）
tensorboard --logdir=logs/experiment_multi_dc_5_*

# === 查看训练结果 ===
# 查看最终检查点
ls -lh logs/experiment_multi_dc_5_*/checkpoints/

# 查看配置和种子
cat logs/experiment_multi_dc_5_*/config_used.yml
cat logs/experiment_multi_dc_5_*/seed_used.txt

# 查看碳排放统计
grep "carbon" logs/experiment_multi_dc_5_*/current_run.log
```

### PowerShell 版本（Windows）

```powershell
# === 启动 MultiDC Java Gateway ===
cd cloudsimplus-gateway
.\gradlew.bat build -x test
.\gradlew.bat run "-PappMainClass=giu.edu.cspg.MainMultiDC"

# === 运行实验5（5个数据中心 + 碳排放优化）⭐推荐 ===
cd drl-manager
.venv\Scripts\Activate.ps1

# 方式 1: 使用环境变量（最简单）
$env:EXPERIMENT_ID = "experiment_multi_dc_5"
$env:NUM_WORKERS = 0
$env:TOTAL_TIMESTEPS = 32000
python entrypoint_pettingzoo.py

# 方式 2: 使用命令行参数
python entrypoint_pettingzoo.py `
    --experiment experiment_multi_dc_5 `
    --num-workers 0 `
    --total-timesteps 32000 `
    --num-gpus 1

# 方式 3: 快速测试（5000步）
python entrypoint_pettingzoo.py `
    --experiment experiment_multi_dc_5 `
    --num-workers 0 `
    --total-timesteps 5000 `
    --num-gpus 1

# === 运行实验3（3个数据中心）===
$env:EXPERIMENT_ID = "experiment_multi_dc_3"
$env:TOTAL_TIMESTEPS = 100000
python entrypoint_pettingzoo.py

# === 监控训练（实验5）===
# 实时日志
Get-Content logs\experiment_multi_dc_5_*\current_run.log -Wait -Tail 50

# TensorBoard
tensorboard --logdir=logs\experiment_multi_dc_5_*

# 查看碳排放
Select-String -Path "logs\experiment_multi_dc_5_*\current_run.log" -Pattern "carbon"
```

### 多数据中心层次化训练（Alternating/Simultaneous）

```bash
# === 启动 MultiDC Java Gateway ===
cd cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# === 运行层次化联合训练 ===
cd drl-manager
source .venv/Scripts/activate

# 方式 1: 使用 entrypoint（推荐）
export EXPERIMENT_ID="experiment_multi_dc_3"
export STRATEGY="alternating"  # 或 "simultaneous"
export SEED="2025"
python entrypoint_multidc.py

# 方式 2: 直接调用训练脚本
python -m src.training.train_hierarchical_multidc_joint \
    --config ../config.yml \
    --experiment experiment_multi_dc_3 \
    --strategy alternating \
    --seed 2025

# === 监控训练 ===
tail -f logs/multi_dc_training/*/current_run.log
tensorboard --logdir=logs/multi_dc_training/*/tensorboard
```

---

## 📄 许可证

本项目采用 **GNU General Public License v3.0** 许可证 - 详见 [LICENSE](LICENSE) 文件。

---

## 🙏 致谢

- **[CloudSim Plus](https://cloudsimplus.org/)** - 云模拟框架
- **[Stable Baselines3](https://stable-baselines3.readthedocs.io/)** - RL算法库
- **[Gymnasium](https://gymnasium.farama.org/)** - RL环境接口
- **[Py4J](https://www.py4j.org/)** - Java-Python桥接

---

<div align="center">

**⭐ 如果这个项目对你有帮助，请给一个Star！**

**[🔝 返回顶部](#drl-cloudsim-绿色调度---中文使用指南)**

</div>
