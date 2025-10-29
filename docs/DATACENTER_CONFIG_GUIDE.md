# DatacenterConfig 配置指南

## 📋 目录
1. [配置流程概览](#配置流程概览)
2. [单数据中心配置](#单数据中心配置)
3. [多数据中心配置](#多数据中心配置)
4. [配置参数详解](#配置参数详解)
5. [实验配置示例](#实验配置示例)
6. [Python端使用方法](#python端使用方法)
7. [常见问题](#常见问题)

---

## 配置流程概览

```
┌─────────────────────────────────────────────────────────────┐
│  1. config.yml (YAML配置文件)                               │
│     - 定义实验参数                                           │
│     - 包含datacenter配置                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  2. Python (load_config)                                     │
│     - 使用 yaml.safe_load() 读取配置                        │
│     - 转换为 Python dict                                     │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  3. Python Environment (HierarchicalMultiDCEnv)             │
│     - 传递 config dict 到 Java Gateway                      │
└────────────────┬────────────────────────────────────────────┘
                 │ Py4J
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  4. Java (HierarchicalMultiDCGateway)                        │
│     - configureSimulation(Map<String, Object> params)       │
│     - 解析配置参数                                           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  5. Java (parseDatacenterConfigs)                            │
│     - 检查 multi_datacenter_enabled                         │
│     - 单DC模式：从通用参数创建1个config                     │
│     - 多DC模式：从 datacenters 列表创建多个config           │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  6. Java (DatacenterConfig.builder())                        │
│     - 使用 Lombok @Builder 模式创建配置对象                 │
│     - 每个datacenter一个 DatacenterConfig 实例              │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────────┐
│  7. Java (MultiDatacenterSimulationCore)                    │
│     - 使用 DatacenterConfig 创建 DatacenterInstance         │
│     - 创建 hosts, VMs, GreenEnergyProvider                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 单数据中心配置

### 方式1: 使用通用参数（默认行为）

如果 `multi_datacenter_enabled` 为 `false` 或未设置，系统会从通用参数创建一个默认数据中心。

**config.yml 示例**:

```yaml
experiment_single_dc:
  simulation_name: "SingleDC_Test"

  # 不设置或设置为 false
  multi_datacenter_enabled: false

  # 通用基础设施参数（会用于创建单个DC）
  hosts_count: 32
  host_pes: 16
  host_pe_mips: 50000
  host_ram: 65536
  host_bw: 50000
  host_storage: 100000

  # VM配置
  small_vm_pes: 2
  small_vm_ram: 8192
  small_vm_bw: 1000
  small_vm_storage: 4000
  medium_vm_multiplier: 2
  large_vm_multiplier: 4

  # 初始VM数量
  initial_s_vm_count: 20
  initial_m_vm_count: 10
  initial_l_vm_count: 5

  # 绿色能源
  green_energy_enabled: true
  turbine_id: 57
  wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

  # VM生命周期
  vm_startup_delay: 0.0
  vm_shutdown_delay: 0.0
```

**Java端解析逻辑** (`HierarchicalMultiDCGateway.java:68-97`):

```java
if (!enabled) {
    // Single datacenter mode - create one config from common params
    DatacenterConfig config = DatacenterConfig.builder()
            .datacenterId(0)
            .datacenterName("DC_0")
            .hostsCount(getIntParam(params, "hosts_count", 16))
            .hostPes(getIntParam(params, "host_pes", 16))
            .hostPeMips(getLongParam(params, "host_pe_mips", 50000))
            // ... 其他参数
            .build();
    configs.add(config);
    return configs;
}
```

---

## 多数据中心配置

### 启用多DC模式

设置 `multi_datacenter_enabled: true`，并在 `datacenters` 列表中定义每个数据中心的配置。

**config.yml 示例**:

```yaml
experiment_multi_dc_3:
  simulation_name: "MultiDC_3_Heterogeneous"

  # 启用多DC模式
  multi_datacenter_enabled: true

  # 数据中心列表配置
  datacenters:
    # 数据中心 0: 高性能DC
    - datacenter_id: 0
      name: "DC_HighPerformance"

      # 基础设施配置
      hosts_count: 20
      host_pes: 24                    # 24核/主机
      host_pe_mips: 60000             # 高MIPS
      host_ram: 131072                # 128 GB
      host_bw: 100000                 # 100 Gbps
      host_storage: 200000

      # VM配置
      small_vm_pes: 2
      small_vm_ram: 8192
      small_vm_bw: 1000
      small_vm_storage: 4000
      medium_vm_multiplier: 2
      large_vm_multiplier: 4

      # 初始VM数量
      initial_s_vm_count: 20
      initial_m_vm_count: 10
      initial_l_vm_count: 6

      # 绿色能源（独立的风力涡轮机）
      green_energy_enabled: true
      turbine_id: 57                  # 涡轮机57
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

      # VM延迟
      vm_startup_delay: 0.0
      vm_shutdown_delay: 0.0

    # 数据中心 1: 能源高效DC
    - datacenter_id: 1
      name: "DC_EnergyEfficient"

      # 基础设施（更多主机，较低功耗）
      hosts_count: 24
      host_pes: 16                    # 16核/主机
      host_pe_mips: 50000             # 标准MIPS
      host_ram: 65536                 # 64 GB
      host_bw: 50000                  # 50 Gbps
      host_storage: 100000

      # VM配置
      small_vm_pes: 2
      small_vm_ram: 8192
      small_vm_bw: 1000
      small_vm_storage: 4000
      medium_vm_multiplier: 2
      large_vm_multiplier: 4

      # 初始VM数量
      initial_s_vm_count: 18
      initial_m_vm_count: 8
      initial_l_vm_count: 4

      # 绿色能源（不同的涡轮机）
      green_energy_enabled: true
      turbine_id: 58                  # 涡轮机58
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

      vm_startup_delay: 0.0
      vm_shutdown_delay: 0.0

    # 数据中心 2: 边缘DC（低容量）
    - datacenter_id: 2
      name: "DC_Edge"

      # 基础设施（较小，边缘位置）
      hosts_count: 12
      host_pes: 12                    # 12核/主机
      host_pe_mips: 40000             # 较低MIPS
      host_ram: 32768                 # 32 GB
      host_bw: 25000                  # 25 Gbps
      host_storage: 50000

      # VM配置（较小的VM）
      small_vm_pes: 2
      small_vm_ram: 4096              # 更小的RAM
      small_vm_bw: 500
      small_vm_storage: 2000
      medium_vm_multiplier: 2
      large_vm_multiplier: 3          # 较小的Large VM

      # 初始VM数量
      initial_s_vm_count: 12
      initial_m_vm_count: 6
      initial_l_vm_count: 2

      # 绿色能源
      green_energy_enabled: true
      turbine_id: 59                  # 涡轮机59
      wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"

      vm_startup_delay: 0.0
      vm_shutdown_delay: 0.0

  # 其他仿真设置
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/poisson_05_300.csv"
  simulation_timestep: 1.0
  max_episode_length: 2000
```

**Java端解析逻辑** (`HierarchicalMultiDCGateway.java:99-110`):

```java
// Multi-datacenter mode - parse list of DC configs
Object dcListObj = params.get("datacenters");
if (dcListObj instanceof List) {
    List<Map<String, Object>> dcList = (List<Map<String, Object>>) dcListObj;
    for (Map<String, Object> dcParams : dcList) {
        DatacenterConfig config = parseDatacenterConfig(dcParams);
        configs.add(config);
    }
}
```

---

## 配置参数详解

### DatacenterConfig 所有参数

| 参数名 | 类型 | 说明 | 默认值 | 示例 |
|--------|------|------|--------|------|
| **标识** |
| `datacenter_id` | int | 数据中心唯一ID | 0 | 0, 1, 2 |
| `name` | String | 数据中心名称 | "DC_0" | "DC_HighPerformance" |
| **主机配置** |
| `hosts_count` | int | 物理主机数量 | 16 | 20, 32 |
| `host_pes` | int | 每个主机的核心数 | 16 | 12, 16, 24 |
| `host_pe_mips` | long | 每个核心的MIPS容量 | 50000 | 40000, 50000, 60000 |
| `host_ram` | long | 主机RAM (MB) | 65536 | 32768, 65536, 131072 |
| `host_bw` | long | 主机带宽 (Mbps) | 50000 | 25000, 50000, 100000 |
| `host_storage` | long | 主机存储 (MB) | 100000 | 50000, 100000, 200000 |
| **VM配置（Small VM基准）** |
| `small_vm_pes` | int | Small VM的核心数 | 2 | 2, 4 |
| `small_vm_ram` | long | Small VM的RAM (MB) | 8192 | 4096, 8192 |
| `small_vm_bw` | long | Small VM的带宽 (Mbps) | 1000 | 500, 1000 |
| `small_vm_storage` | long | Small VM的存储 (MB) | 4000 | 2000, 4000 |
| **VM倍数** |
| `medium_vm_multiplier` | int | Medium VM = Small × multiplier | 2 | 2 |
| `large_vm_multiplier` | int | Large VM = Small × multiplier | 4 | 3, 4 |
| **初始VM舰队** |
| `initial_s_vm_count` | int | 初始Small VM数量 | 10 | 10, 20, 30 |
| `initial_m_vm_count` | int | 初始Medium VM数量 | 5 | 5, 10 |
| `initial_l_vm_count` | int | 初始Large VM数量 | 3 | 2, 5, 10 |
| **绿色能源** |
| `green_energy_enabled` | boolean | 启用绿色能源 | true | true, false |
| `turbine_id` | int | 风力涡轮机ID | 57 | 57, 58, 59 |
| `wind_data_file` | String | 风力数据文件路径 | "windProduction/..." | 路径字符串 |
| **VM生命周期** |
| `vm_startup_delay` | double | VM启动延迟（秒） | 0.0 | 0.0, 10.0 |
| `vm_shutdown_delay` | double | VM关闭延迟（秒） | 0.0 | 0.0, 5.0 |

### 计算的派生值

`DatacenterConfig` 提供了一些辅助方法来计算派生值：

```java
// 获取总VM数量
int totalVms = config.getTotalInitialVmCount();
// = initial_s_vm_count + initial_m_vm_count + initial_l_vm_count

// 获取总核心数
int totalCores = config.getTotalCores();
// = hosts_count * host_pes

// 获取Medium VM的核心数
int mediumPes = config.getMediumVmPes();
// = small_vm_pes * medium_vm_multiplier

// 获取Large VM的核心数
int largePes = config.getLargeVmPes();
// = small_vm_pes * large_vm_multiplier

// 获取Medium VM的RAM
long mediumRam = config.getMediumVmRam();
// = small_vm_ram * medium_vm_multiplier

// 获取Large VM的RAM
long largeRam = config.getLargeVmRam();
// = small_vm_ram * large_vm_multiplier
```

---

## 实验配置示例

### 示例1: 同构3数据中心（用于基准测试）

```yaml
experiment_homogeneous_3dc:
  simulation_name: "Homogeneous_3DC"
  multi_datacenter_enabled: true

  datacenters:
    - datacenter_id: 0
      name: "DC_0"
      hosts_count: 20
      host_pes: 16
      host_pe_mips: 50000
      host_ram: 65536
      # ... 其他参数相同
      turbine_id: 57

    - datacenter_id: 1
      name: "DC_1"
      hosts_count: 20        # 相同配置
      host_pes: 16
      host_pe_mips: 50000
      host_ram: 65536
      turbine_id: 58         # 不同涡轮机

    - datacenter_id: 2
      name: "DC_2"
      hosts_count: 20        # 相同配置
      host_pes: 16
      host_pe_mips: 50000
      host_ram: 65536
      turbine_id: 59         # 不同涡轮机
```

### 示例2: 异构5数据中心（真实场景）

```yaml
experiment_heterogeneous_5dc:
  simulation_name: "Heterogeneous_5DC"
  multi_datacenter_enabled: true

  datacenters:
    # 中央高性能DC
    - datacenter_id: 0
      name: "DC_Central_HighPerf"
      hosts_count: 32
      host_pes: 32
      host_pe_mips: 80000
      host_ram: 262144        # 256 GB
      initial_s_vm_count: 30
      initial_m_vm_count: 20
      initial_l_vm_count: 10
      turbine_id: 57

    # 西部数据中心
    - datacenter_id: 1
      name: "DC_West"
      hosts_count: 24
      host_pes: 16
      host_pe_mips: 50000
      host_ram: 65536
      initial_s_vm_count: 20
      initial_m_vm_count: 10
      initial_l_vm_count: 5
      turbine_id: 58

    # 东部数据中心
    - datacenter_id: 2
      name: "DC_East"
      hosts_count: 24
      host_pes: 16
      host_pe_mips: 50000
      host_ram: 65536
      initial_s_vm_count: 20
      initial_m_vm_count: 10
      initial_l_vm_count: 5
      turbine_id: 59

    # 北部边缘DC
    - datacenter_id: 3
      name: "DC_Edge_North"
      hosts_count: 12
      host_pes: 8
      host_pe_mips: 30000
      host_ram: 32768
      initial_s_vm_count: 10
      initial_m_vm_count: 5
      initial_l_vm_count: 2
      turbine_id: 60

    # 南部边缘DC
    - datacenter_id: 4
      name: "DC_Edge_South"
      hosts_count: 12
      host_pes: 8
      host_pe_mips: 30000
      host_ram: 32768
      initial_s_vm_count: 10
      initial_m_vm_count: 5
      initial_l_vm_count: 2
      turbine_id: 61
```

### 示例3: 绿色能源优化（不同涡轮机）

```yaml
experiment_green_optimization:
  simulation_name: "Green_Optimization_3DC"
  multi_datacenter_enabled: true

  datacenters:
    # DC0: 强风区域（高绿色能源）
    - datacenter_id: 0
      name: "DC_HighWind"
      turbine_id: 10          # 选择强风涡轮机
      green_energy_enabled: true
      hosts_count: 16
      # ... 其他配置

    # DC1: 中等风力区域
    - datacenter_id: 1
      name: "DC_MediumWind"
      turbine_id: 50          # 中等风力涡轮机
      green_energy_enabled: true
      hosts_count: 20
      # ... 其他配置

    # DC2: 低风区域（主要依赖电网）
    - datacenter_id: 2
      name: "DC_LowWind"
      turbine_id: 100         # 低风力涡轮机
      green_energy_enabled: true
      hosts_count: 24
      # ... 其他配置
```

---

## Python端使用方法

### 方法1: 直接使用环境（推荐用于训练）

```python
import yaml
from hierarchical_multidc_env import HierarchicalMultiDCEnv

# 1. 加载配置
with open('config.yml', 'r') as f:
    all_configs = yaml.safe_load(f)

# 2. 选择实验配置
config = all_configs['experiment_multi_dc_3']

# 3. 创建环境
env = HierarchicalMultiDCEnv(config)

# 4. 重置环境（配置在reset时传递给Java）
observations, info = env.reset(seed=42)

# 5. 运行step
action = {
    "global": [0, 1, 2],  # 将3个到达的cloudlet分配到DC 0, 1, 2
    "local": {
        0: 5,  # DC 0: 分配到VM 5
        1: 3,  # DC 1: 分配到VM 3
        2: 7   # DC 2: 分配到VM 7
    }
}
observations, rewards, terminated, truncated, info = env.step(action)
```

### 方法2: 使用训练脚本

```python
# train_hierarchical_marl.py 中的用法

def load_config(config_path: str) -> Dict[str, Any]:
    """加载配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

# 命令行使用
# python train_hierarchical_marl.py \
#     --config ../config.yml \
#     --experiment experiment_multi_dc_3 \
#     --phase all

# 在脚本内部
config = load_config(args.config)
experiment_config = config[args.experiment]

# 创建trainer
trainer = HierarchicalTrainer(experiment_config, log_dir)
trainer.create_env()  # 内部会创建 HierarchicalMultiDCEnv
```

### 方法3: 编程式配置（用于测试/调试）

```python
# 不从文件加载，直接在代码中定义配置
config = {
    "simulation_name": "Test_2DC",
    "multi_datacenter_enabled": True,
    "datacenters": [
        {
            "datacenter_id": 0,
            "name": "DC_Test_0",
            "hosts_count": 10,
            "host_pes": 8,
            "host_pe_mips": 40000,
            "host_ram": 32768,
            "host_bw": 25000,
            "host_storage": 50000,
            "small_vm_pes": 2,
            "small_vm_ram": 4096,
            "small_vm_bw": 500,
            "small_vm_storage": 2000,
            "medium_vm_multiplier": 2,
            "large_vm_multiplier": 4,
            "initial_s_vm_count": 5,
            "initial_m_vm_count": 3,
            "initial_l_vm_count": 2,
            "green_energy_enabled": True,
            "turbine_id": 57,
            "wind_data_file": "windProduction/sdwpf_2001_2112_full.csv",
            "vm_startup_delay": 0.0,
            "vm_shutdown_delay": 0.0
        },
        {
            "datacenter_id": 1,
            "name": "DC_Test_1",
            # ... 类似的配置
        }
    ],
    "workload_mode": "CSV",
    "cloudlet_trace_file": "traces/three_60max_8maxcores.csv",
    "simulation_timestep": 1.0,
    "max_episode_length": 500
}

env = HierarchicalMultiDCEnv(config)
```

---

## 常见问题

### Q1: 如何添加新的数据中心？

**A**: 在 `datacenters` 列表中添加新的配置块：

```yaml
datacenters:
  - datacenter_id: 0
    name: "DC_0"
    # ... 配置

  - datacenter_id: 1
    name: "DC_1"
    # ... 配置

  # 添加第3个数据中心
  - datacenter_id: 2
    name: "DC_New"
    hosts_count: 16
    # ... 其他参数
```

**注意**:
- `datacenter_id` 必须唯一且连续（0, 1, 2, ...）
- 每个DC必须有不同的 `turbine_id`（如果启用绿色能源）

---

### Q2: 如何禁用某个数据中心的绿色能源？

**A**: 设置 `green_energy_enabled: false`:

```yaml
datacenters:
  - datacenter_id: 0
    name: "DC_NoGreen"
    green_energy_enabled: false  # 禁用绿色能源
    turbine_id: 57               # 仍需提供，但不会使用
    # ... 其他配置
```

---

### Q3: 如何配置不同的VM大小比例？

**A**: 调整 `medium_vm_multiplier` 和 `large_vm_multiplier`:

```yaml
# 例子1: 保守的VM大小（适合低功耗DC）
small_vm_pes: 2
medium_vm_multiplier: 2    # Medium = 4 PEs
large_vm_multiplier: 3     # Large = 6 PEs

# 例子2: 激进的VM大小（适合高性能DC）
small_vm_pes: 4
medium_vm_multiplier: 2    # Medium = 8 PEs
large_vm_multiplier: 4     # Large = 16 PEs
```

---

### Q4: 如何验证配置是否正确？

**方法1: 使用Java日志**

启动Java gateway后，查看日志：

```
INFO: Created datacenter: DC_HighPerformance (ID: 0) with 20 hosts
INFO: GreenEnergyProvider created for DC 0 with turbine 57
INFO: Created 20 hosts for datacenter 0 (DC_HighPerformance)
```

**方法2: 使用Python环境检查**

```python
env = HierarchicalMultiDCEnv(config)
env.reset()

# 检查数据中心数量
num_dcs = env.get_num_datacenters()
print(f"Number of datacenters: {num_dcs}")

# 获取观测空间
print(f"Global observation space: {env.global_observation_space}")
print(f"Local observation space: {env.local_observation_space}")
```

---

### Q5: 如何使用不同的风力数据文件？

**A**: 在每个数据中心配置中指定不同的文件：

```yaml
datacenters:
  - datacenter_id: 0
    wind_data_file: "windProduction/region_north.csv"
    turbine_id: 10

  - datacenter_id: 1
    wind_data_file: "windProduction/region_south.csv"
    turbine_id: 20
```

**注意**: 文件路径相对于 `cloudsimplus-gateway/src/main/resources/`

---

### Q6: 单DC模式和多DC模式可以共存吗？

**A**: 可以！在同一个 `config.yml` 中定义多个实验：

```yaml
# 单DC实验
experiment_single_dc:
  simulation_name: "Single_DC"
  multi_datacenter_enabled: false
  hosts_count: 32
  # ... 单DC参数

# 多DC实验
experiment_multi_dc:
  simulation_name: "Multi_DC"
  multi_datacenter_enabled: true
  datacenters:
    - datacenter_id: 0
      # ... DC0配置
    - datacenter_id: 1
      # ... DC1配置
```

使用时选择不同的实验：

```python
# 单DC
config = all_configs['experiment_single_dc']

# 多DC
config = all_configs['experiment_multi_dc']
```

---

### Q7: 如何快速创建测试配置？

**A**: 使用 Java 的 `createDefault()` 工厂方法（适用于单元测试）：

```java
// 在Java测试代码中
DatacenterConfig testConfig = DatacenterConfig.createDefault(0);
// 使用默认值创建DC 0

List<DatacenterConfig> configs = Arrays.asList(
    DatacenterConfig.createDefault(0),
    DatacenterConfig.createDefault(1),
    DatacenterConfig.createDefault(2)
);
// 快速创建3个DC用于测试
```

---

## 配置最佳实践

### ✅ 推荐做法

1. **使用描述性名称**
   ```yaml
   name: "DC_HighPerformance_US_West"  # 好
   name: "DC_0"                         # 不太好
   ```

2. **保持turbine_id唯一**
   ```yaml
   # 好：每个DC不同的涡轮机
   - datacenter_id: 0
     turbine_id: 57
   - datacenter_id: 1
     turbine_id: 58
   ```

3. **注释配置的意图**
   ```yaml
   hosts_count: 32  # Provisioned for peak load
   ```

4. **从小规模开始测试**
   ```yaml
   # 开发/测试配置
   hosts_count: 4
   initial_s_vm_count: 2
   initial_m_vm_count: 1
   initial_l_vm_count: 1
   ```

5. **使用版本控制跟踪配置变化**
   - Git commit message: "config: Increase DC0 capacity for experiment X"

### ❌ 避免的做法

1. **硬编码路径**
   ```yaml
   # 不好
   wind_data_file: "C:/Users/xxx/windProduction/data.csv"

   # 好（相对路径）
   wind_data_file: "windProduction/sdwpf_2001_2112_full.csv"
   ```

2. **DC配置不一致**
   ```yaml
   # 不好：容易混淆
   - datacenter_id: 0
     name: "DC_East"
     turbine_id: 100
   - datacenter_id: 5     # 跳过了1-4
     name: "DC_West"
     turbine_id: 100      # 重复的turbine_id
   ```

3. **过度配置资源**
   ```yaml
   # 可能不切实际
   hosts_count: 10000
   host_ram: 10485760  # 10 TB RAM per host
   ```

---

## 总结

`DatacenterConfig` 的配置流程：

1. **定义**: 在 `config.yml` 中定义实验配置
2. **加载**: Python使用 `yaml.safe_load()` 加载为dict
3. **传递**: 通过Py4J传递给Java gateway
4. **解析**: Java解析为 `DatacenterConfig` 对象
5. **使用**: 用于创建仿真环境中的数据中心实例

**关键点**:
- 单DC模式：使用通用参数
- 多DC模式：使用 `datacenters` 列表
- 每个DC可以有完全不同的配置
- 支持异构数据中心（不同容量、不同绿色能源）
- 配置是声明式的，在运行时解析

**下一步**:
- 参考 `config.yml` 中的 `experiment_multi_dc_3` 示例
- 根据实验需求调整配置参数
- 使用 `hierarchical_multidc_env.py` 运行实验
