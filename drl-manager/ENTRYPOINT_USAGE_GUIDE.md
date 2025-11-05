# Entrypoint Usage Guide - Unified Entry Point for All Modes

## TL;DR

✅ **不需要单独的entrypoint！** 现有的`entrypoint.py`已经自动集成了hierarchical multi-DC训练。

---

## 统一的Entrypoint架构

```
                    entrypoint.py
                         |
                         | 检查配置
                         ↓
        +----------------+----------------+
        |                                 |
    单数据中心模式              多数据中心模式
        |                                 |
        ↓                                 ↓
multi_datacenter_enabled: false    multi_datacenter_enabled: true
        |                                 |
        ↓                                 ↓
     train.py                    train_hierarchical_multidc.py
     test.py                     (自动选择)
     transfer.py
```

---

## 自动路由机制

### entrypoint.py (lines 157-165)

```python
# --- Check for Multi-Datacenter Mode ---
is_multi_dc = params.get("multi_datacenter_enabled", False)
if is_multi_dc and mode == "train":
    logger.info("Multi-datacenter mode detected, using hierarchical training module")
    mode_module = "train_hierarchical_multidc"
    func_name = "train_hierarchical_multidc"
else:
    mode_module = mode
    func_name = mode

# Dynamically import and execute
module = importlib.import_module(mode_module)
func = getattr(module, func_name)
func(params)  # 调用训练函数
```

**判断逻辑：**
```python
if multi_datacenter_enabled == True AND mode == "train":
    使用 train_hierarchical_multidc
else:
    使用 train.py / test.py / transfer.py
```

---

## 使用方式

### 🚀 方式1: 通过统一Entrypoint（推荐）

#### **单数据中心训练**

```bash
# 1. 启动Java Gateway
cd cloudsimplus-gateway
./gradlew run

# 2. 新终端，运行训练
export EXPERIMENT_ID="experiment_11_long"  # 单DC实验
python drl-manager/mnt/entrypoint.py
```

**执行流程：**
```
entrypoint.py
  ↓ 加载 config.yml[experiment_11_long]
  ↓ multi_datacenter_enabled: false (或不存在)
  ↓ mode: "train"
  ↓ 路由到 train.py
  ↓ 使用 LoadBalancingEnv
  ↓ 单DC训练
```

#### **多数据中心训练**

```bash
# 1. 启动Java Gateway
cd cloudsimplus-gateway
./gradlew run

# 2. 新终端，运行训练
export EXPERIMENT_ID="experiment_multi_dc_3"  # 多DC实验
python drl-manager/mnt/entrypoint.py
```

**执行流程：**
```
entrypoint.py
  ↓ 加载 config.yml[experiment_multi_dc_3]
  ↓ multi_datacenter_enabled: true ✅
  ↓ mode: "train"
  ↓ 路由到 train_hierarchical_multidc.py ← 自动选择！
  ↓ 使用 HierarchicalMultiDCEnv
  ↓ Phase 1: 训练Local Agents
  ↓ Phase 2: 训练Global Agent
  ↓ 多DC训练完成
```

---

### 🔧 方式2: 直接调用训练脚本（适合调试）

#### **单DC训练**

```bash
cd drl-manager/mnt
python train.py  # 需要手动配置params
```

#### **多DC训练**

```bash
cd drl-manager/mnt
python train_hierarchical_multidc.py  # 需要手动配置params
```

**注意：** 这种方式需要修改脚本内的配置，不推荐。

---

## 配置文件示例

### 单数据中心实验

```yaml
# config.yml
experiment_11_long:
  # 不设置multi_datacenter_enabled，或设置为false
  simulation_name: "Exp11_LongEpisode"
  experiment_name: "exp11_3600s"

  # 单DC配置
  hosts_count: 32
  initial_s_vm_count: 20
  initial_m_vm_count: 10
  initial_l_vm_count: 5

  # RL配置
  mode: "train"
  algorithm: "MaskablePPO"
  timesteps: 1000000
  # ...
```

**使用：**
```bash
export EXPERIMENT_ID="experiment_11_long"
python drl-manager/mnt/entrypoint.py
# → 自动使用 train.py
```

### 多数据中心实验

```yaml
# config.yml
experiment_multi_dc_3:
  # ✅ 启用多数据中心模式
  multi_datacenter_enabled: true
  py4j_port: 25333
  max_arriving_cloudlets: 50

  # 定义数据中心配置
  datacenters:
    - datacenter_id: 0
      name: "DC_HighPerformance"
      hosts_count: 20
      initial_s_vm_count: 20
      # ...

    - datacenter_id: 1
      name: "DC_EnergyEfficient"
      hosts_count: 24
      initial_s_vm_count: 18
      # ...

  # RL配置
  mode: "train"
  global_agent:
    algorithm: "PPO"
    learning_rate: 0.0003
    # ...

  local_agents:
    algorithm: "MaskablePPO"
    parameter_sharing: true
    # ...
```

**使用：**
```bash
export EXPERIMENT_ID="experiment_multi_dc_3"
python drl-manager/mnt/entrypoint.py
# → 自动使用 train_hierarchical_multidc.py
```

---

## 模式切换对比

### 传统方式（需要手动选择）

```bash
# ❌ 需要记住不同的命令
python train.py              # 单DC
python train_multi_dc.py     # 多DC
python train_hierarchical.py # Hierarchical
```

### 统一Entrypoint方式（自动选择）

```bash
# ✅ 统一入口，自动路由
export EXPERIMENT_ID="experiment_11_long"
python entrypoint.py  # 自动识别为单DC

export EXPERIMENT_ID="experiment_multi_dc_3"
python entrypoint.py  # 自动识别为多DC
```

---

## 完整工作流

### 单DC实验完整流程

```bash
# 1. 启动Java
cd cloudsimplus-gateway
./gradlew run

# 2. 新终端，激活Python环境
cd drl-manager
source .venv/bin/activate  # Linux/Mac
# 或 .venv\Scripts\activate  # Windows

# 3. 运行训练
export EXPERIMENT_ID="experiment_11_long"
python mnt/entrypoint.py

# 4. 查看结果
ls logs/exp11_3600s/
```

### 多DC实验完整流程

```bash
# 1. 启动Java
cd cloudsimplus-gateway
./gradlew run

# 2. 新终端，激活Python环境
cd drl-manager
source .venv/bin/activate

# 3. 运行训练
export EXPERIMENT_ID="experiment_multi_dc_3"
python mnt/entrypoint.py

# 4. 查看结果
ls logs/hierarchical_3dc/
# local_agent.zip
# global_agent.zip
# progress.csv
```

---

## 日志输出对比

### 单DC模式日志

```
[INFO] --- DRL Manager Entrypoint Starting ---
[INFO] Loaded configuration for experiment 'experiment_11_long'
[INFO] Selected mode: train
[INFO] Executing mode 'train'
[INFO] Creating Gym environment: LoadBalancingScaling-v0
[INFO] Training single-datacenter agent...
```

### 多DC模式日志

```
[INFO] --- DRL Manager Entrypoint Starting ---
[INFO] Loaded configuration for experiment 'experiment_multi_dc_3'
[INFO] Selected mode: train
[INFO] Multi-datacenter mode detected, using hierarchical training module
[INFO] ============================================================
[INFO] Starting Hierarchical Multi-Datacenter MARL Training
[INFO] ============================================================
[INFO] PHASE 1: Training Local Agents (VM Scheduling)
[INFO] Creating local agent (MaskablePPO)...
```

---

## 高级功能

### 环境变量覆盖

```bash
# 使用不同的配置文件
export CONFIG_FILE="config_dev.yml"
export EXPERIMENT_ID="experiment_test"
python drl-manager/mnt/entrypoint.py
```

### 指定模式

```yaml
# config.yml
experiment_test:
  mode: "test"  # 测试模式
  # 或
  mode: "transfer"  # 迁移学习模式
```

```bash
export EXPERIMENT_ID="experiment_test"
python drl-manager/mnt/entrypoint.py
# → 自动调用 test.py 或 transfer.py
```

---

## 故障排除

### 问题1: 使用了错误的训练脚本

**症状：**
```
[ERROR] Mode script 'train_hierarchical_multidc.py' not found
```

**原因：** `train_hierarchical_multidc.py`不在`mnt/`目录

**解决：**
```bash
ls drl-manager/mnt/train_hierarchical_multidc.py
# 确认文件存在
```

### 问题2: 配置未生效

**症状：** 多DC实验使用了单DC训练

**原因：** 配置中`multi_datacenter_enabled`未设置或为false

**解决：**
```yaml
experiment_multi_dc_3:
  multi_datacenter_enabled: true  # ← 确保这行存在且为true
```

### 问题3: 模块导入失败

**症状：**
```
ModuleNotFoundError: No module named 'train_hierarchical_multidc'
```

**原因：** Python路径问题

**解决：**
```bash
# 确保从正确的目录运行
cd /path/to/rl-cloudsimplus-greenscheduling
python drl-manager/mnt/entrypoint.py
```

---

## 架构优势

### ✅ 统一入口的优势

1. **简化使用**
   - 用户只需记住一个命令
   - 自动根据配置选择正确的训练模式

2. **降低错误**
   - 不会因为选错脚本而训练失败
   - 配置驱动，减少人为失误

3. **易于扩展**
   - 添加新模式只需修改entrypoint.py
   - 新模式自动集成到统一入口

4. **一致的接口**
   - 所有训练脚本使用相同的函数签名：`train(params)`
   - 所有配置统一在config.yml中管理

### ✅ 与其他系统对比

| 特性 | 传统方式 | 统一Entrypoint |
|------|---------|---------------|
| **命令数** | 多个（train.py, train_multi.py...） | 1个（entrypoint.py） |
| **配置方式** | 命令行参数 | config.yml |
| **模式切换** | 手动更改命令 | 更改EXPERIMENT_ID |
| **易用性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **可维护性** | ⭐⭐ | ⭐⭐⭐⭐⭐ |
| **扩展性** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |

---

## 总结

### ✅ 不需要单独的entrypoint

- 现有的`entrypoint.py`已经完美集成了所有训练模式
- 通过`multi_datacenter_enabled`参数自动路由
- 统一的配置管理和命令接口

### 🚀 推荐使用方式

```bash
# 1. 配置实验（config.yml）
experiment_multi_dc_3:
  multi_datacenter_enabled: true
  # ... 其他配置

# 2. 运行（统一命令）
export EXPERIMENT_ID="experiment_multi_dc_3"
python drl-manager/mnt/entrypoint.py
```

### 📚 更多文档

- `HIERARCHICAL_MULTIDC_GUIDE.md` - 多DC训练详细指南
- `ERROR_HANDLING_IMPROVEMENTS.md` - 错误处理文档
- `config.yml` - 配置文件示例

---

**总结：统一的entrypoint.py已经完美支持所有模式，不需要额外的entrypoint！** 🎉
