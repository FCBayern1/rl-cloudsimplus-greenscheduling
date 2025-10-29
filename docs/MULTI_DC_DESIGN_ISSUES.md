# Multi-Datacenter 设计问题分析报告

## 📋 检查概况

**检查日期**: 2025-10-29
**检查范围**: Multi-Datacenter 分层强化学习系统
**关键组件**:
- Python: `hierarchical_multidc_env.py`
- Java: `MultiDatacenterSimulationCore.java`, `HierarchicalMultiDCGateway.java`
- 接口: Py4J Gateway, ObservationState, HierarchicalStepResult

**结论**: 发现 **7 个关键问题**，包括 **2 个 P0 严重问题** 和 **5 个 P1 高优先级问题**。

---

## 🚨 P0 级别问题（阻塞性）

### 问题 1: ❌ **全局观测字段映射错误**

#### **位置**
- 文件: `hierarchical_multidc_env.py`
- 行数: 297-304

#### **问题描述**

Python 端使用了**错误的字段**来解析全局观测：

```python
# hierarchical_multidc_env.py lines 297-304
global_obs = {
    "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),  # ❌ 错误！
    "dc_queue_sizes": np.array(global_obs_java.getHostRamUsageRatio(), dtype=np.float32),  # ❌ 错误！
    "dc_utilizations": np.array(global_obs_java.getHostLoads(), dtype=np.float32),  # ✅ 正确
    "dc_available_pes": np.array(global_obs_java.getVmAvailablePes(), dtype=np.float32),  # ❌ 可能错误
    "upcoming_cloudlets_count": global_obs_java.getWaitingCloudlets(),
    "next_cloudlet_pes": global_obs_java.getNextCloudletPes(),
}
```

**Java 端实际填充** (MultiDatacenterSimulationCore.java lines 416-432):

```java
return new ObservationState(
    dcUtilizations,      // hostLoads ← "dc_utilizations" 使用这个 ✅
    dcUtilizations,      // hostRamUsageRatio ← "dc_queue_sizes" 误用这个 ❌
    dcGreenPower,        // vmLoads ← "dc_green_power" 误用这个 ❌
    new int[numDatacenters],  // vmTypes
    new int[numDatacenters],  // vmHostMap
    infrastructureObs,   // infrastructureObservation
    upcomingCount,
    nextCloudletPes,
    convertDoubleArrayToIntArray(dcAvailablePes),  // vmAvailablePes
    numDatacenters,
    numDatacenters,
    0L, 0.0, new int[3], 0
);
```

#### **正确的映射关系**

| Python 字段 | 应该使用 Java Getter | 当前错误使用 | 实际数据 |
|-------------|---------------------|-------------|----------|
| `dc_green_power` | `getVmLoads()` ✅ | `getVmLoads()` ✅ | DC 绿色能源功率 (kW) |
| `dc_queue_sizes` | 应该用 `infrastructureObs` 的一部分 | `getHostRamUsageRatio()` ❌ | **重复的 dcUtilizations！** |
| `dc_utilizations` | `getHostLoads()` ✅ | `getHostLoads()` ✅ | DC CPU 利用率 |
| `dc_available_pes` | `getVmAvailablePes()` ✅ | `getVmAvailablePes()` ✅ | DC 可用 PEs |

#### **问题影响**

```
🔴 严重性: P0 (Critical)
- "dc_queue_sizes" 实际得到的是 dcUtilizations（利用率），不是队列大小
- 全局 Agent 看到的是错误的状态信息
- 训练会失败或学到错误的策略
- 路由决策完全基于错误数据
```

#### **修复方案**

##### **方案 A: 修改 Java 填充逻辑（推荐）**

```java
// MultiDatacenterSimulationCore.java - getGlobalObservation()
// 创建正确的数组
double[] dcGreenPower = new double[numDatacenters];
double[] dcQueueSizes = new double[numDatacenters];  // ← 新增！
double[] dcUtilizations = new double[numDatacenters];
double[] dcAvailablePes = new double[numDatacenters];

for (int i = 0; i < numDatacenters; i++) {
    DatacenterInstance dc = datacenterInstances.get(i);
    dcGreenPower[i] = dc.getCurrentGreenPowerW(currentClock) / 1000.0;
    dcQueueSizes[i] = dc.getWaitingCloudletCount();  // ← 正确的队列大小！
    dcUtilizations[i] = dc.getAverageHostUtilization();
    dcAvailablePes[i] = dc.getTotalAvailablePes();
}

// 正确填充 ObservationState
return new ObservationState(
    dcUtilizations,      // hostLoads → dc_utilizations
    dcQueueSizes,        // hostRamUsageRatio → dc_queue_sizes ✅ 修正！
    dcGreenPower,        // vmLoads → dc_green_power
    new int[numDatacenters],
    new int[numDatacenters],
    infrastructureObs,
    upcomingCount,
    nextCloudletPes,
    convertDoubleArrayToIntArray(dcAvailablePes),
    numDatacenters,
    numDatacenters,
    0L, 0.0, new int[3], 0
);
```

##### **方案 B: 修改 Python 解析逻辑（不推荐）**

如果不想改 Java，也可以改 Python（但这样意义不清晰）：

```python
# 临时修正，不推荐
global_obs = {
    "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),
    "dc_queue_sizes": np.array(global_obs_java.getHostRamUsageRatio(), dtype=np.float32),
    # 注意：这里拿到的其实是 dcUtilizations 的副本，不是真正的队列大小
    ...
}
```

---

### 问题 2: ❌ **局部动作空间设计不灵活**

#### **位置**
- 文件: `hierarchical_multidc_env.py`
- 行数: 155-164

#### **问题描述**

当前局部动作空间是**单个 Discrete**，但应该是**每个 DC 一个动作**：

```python
# 当前实现 - 错误
self.local_action_space = spaces.Discrete(max_vms)

# 期望的实现
self.local_action_space = spaces.Dict({
    0: spaces.Discrete(max_vms_dc0),
    1: spaces.Discrete(max_vms_dc1),
    ...
})
```

或者使用 MultiDiscrete：

```python
self.local_action_space = spaces.MultiDiscrete([
    num_vms_dc0,
    num_vms_dc1,
    num_vms_dc2,
    ...
])
```

#### **问题影响**

```
🔴 严重性: P0 (Critical)
- 当前定义无法表示"每个 DC 选一个 VM"的语义
- RL 算法无法正确采样动作
- step() 函数期望 Dict[int, int]，但 action_space 是 Discrete(max_vms)
- 类型不匹配会导致运行时错误
```

#### **修复方案**

```python
def _setup_action_spaces(self):
    """
    Define action spaces for global and local agents.
    """
    # Global action space: Select datacenter for each arriving cloudlet
    self.global_action_space = spaces.MultiDiscrete(
        [self.num_datacenters] * self.max_arriving_cloudlets
    )

    # Local action spaces: Dict of {dc_id: Discrete(num_vms)}
    # ✅ 修正：每个 DC 一个动作空间
    dc_action_spaces = {}
    for i, dc_config in enumerate(self.config.get("datacenters", [])):
        num_vms = (
            dc_config.get("initial_s_vm_count", 10) +
            dc_config.get("initial_m_vm_count", 5) +
            dc_config.get("initial_l_vm_count", 3)
        )
        dc_action_spaces[i] = spaces.Discrete(num_vms)

    self.local_action_space = spaces.Dict(dc_action_spaces)
```

---

## ⚠️ P1 级别问题（高优先级）

### 问题 3: ⚠️ **观测空间使用 Dict 导致 RL 训练困难**

#### **位置**
- `hierarchical_multidc_env.py` lines 75-143

#### **问题描述**

当前观测空间使用嵌套的 `Dict` 结构：

```python
observation_space = {
    "global": Dict({
        "dc_green_power": Box(...),
        "dc_queue_sizes": Box(...),
        ...
    }),
    "local": Dict({
        0: Dict({...}),
        1: Dict({...}),
        ...
    })
}
```

**问题**:
- 大多数 RL 算法（PPO, DQN, A2C）**不原生支持** Dict 观测空间
- Stable-Baselines3 需要特殊处理（`MultiInputPolicy`）
- 分层 MARL 训练复杂度大幅增加

#### **影响**

```
⚠️ 严重性: P1 (High)
- 需要自定义 RL 算法包装器
- 训练代码复杂度高
- 调试困难
```

#### **修复方案**

##### **方案 A: 扁平化观测（推荐用于快速测试）**

```python
def _get_flat_observation(self, hierarchical_obs):
    """将分层观测扁平化为单个向量"""
    global_obs = hierarchical_obs["global"]
    local_obs = hierarchical_obs["local"]

    # 拼接全局观测
    global_vec = np.concatenate([
        global_obs["dc_green_power"],
        global_obs["dc_queue_sizes"],
        global_obs["dc_utilizations"],
        global_obs["dc_available_pes"],
        [global_obs["upcoming_cloudlets_count"]],
        [global_obs["next_cloudlet_pes"]]
    ])

    # 拼接所有 DC 的局部观测
    local_vecs = []
    for dc_id in sorted(local_obs.keys()):
        dc_obs = local_obs[dc_id]
        local_vec = np.concatenate([
            dc_obs["host_loads"],
            dc_obs["host_ram_usage"],
            dc_obs["vm_loads"],
            dc_obs["vm_types"],
            dc_obs["vm_available_pes"],
            [dc_obs["waiting_cloudlets"]],
            [dc_obs["next_cloudlet_pes"]]
        ])
        local_vecs.append(local_vec)

    # 完整扁平观测
    return np.concatenate([global_vec] + local_vecs)
```

##### **方案 B: 保持分层但使用标准格式**

```python
# 只返回 numpy 数组，不用 Dict
observation_space = spaces.Box(
    low=-np.inf,
    high=np.inf,
    shape=(total_obs_dim,),
    dtype=np.float32
)
```

---

### 问题 4: ⚠️ **Py4J Map 转换问题**

#### **位置**
- `hierarchical_multidc_env.py` lines 307-321, 343-347

#### **问题描述**

Py4J 的 Java Map 到 Python dict 转换需要小心处理：

```python
# 当前代码 - 可能有问题
local_obs_map_java = result.getLocalObservations()  # Java Map<Integer, ObservationState>

for dc_id in range(self.num_datacenters):
    if dc_id in local_obs_map_java:  # ← 这个 'in' 操作可能不工作！
        ...
```

**Py4J 的 Map 不完全兼容 Python dict**，特别是：
- `in` 操作符可能不可靠
- 键类型转换（Integer ↔ int）可能有问题
- 迭代方式不同

#### **影响**

```
⚠️ 严重性: P1 (High)
- 运行时可能抛出 KeyError
- 某些 DC 的观测可能丢失
- 奖励解析也有同样问题
```

#### **修复方案**

```python
def _parse_hierarchical_observation(self, result) -> Dict[str, Any]:
    """
    Parse hierarchical step result into observation dict.
    """
    # Parse global observation
    global_obs_java = result.getGlobalObservation()
    global_obs = {
        "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),
        "dc_queue_sizes": np.array(global_obs_java.getHostRamUsageRatio(), dtype=np.float32),
        "dc_utilizations": np.array(global_obs_java.getHostLoads(), dtype=np.float32),
        "dc_available_pes": np.array(global_obs_java.getVmAvailablePes(), dtype=np.float32),
        "upcoming_cloudlets_count": global_obs_java.getWaitingCloudlets(),
        "next_cloudlet_pes": global_obs_java.getNextCloudletPes(),
    }

    # Parse local observations
    local_obs_map_java = result.getLocalObservations()
    local_obs = {}

    # ✅ 修正：使用 Java Map 的正确迭代方式
    for dc_id_obj in local_obs_map_java.keySet():  # ← 使用 keySet()
        dc_id = int(dc_id_obj)  # ← 显式转换为 Python int
        local_obs_java = local_obs_map_java.get(dc_id_obj)  # ← 使用原始 Java 对象作为键

        local_obs[dc_id] = {
            "host_loads": np.array(local_obs_java.getHostLoads(), dtype=np.float32),
            "host_ram_usage": np.array(local_obs_java.getHostRamUsageRatio(), dtype=np.float32),
            "vm_loads": np.array(local_obs_java.getVmLoads(), dtype=np.float32),
            "vm_types": np.array(local_obs_java.getVmTypes(), dtype=np.int32),
            "vm_available_pes": np.array(local_obs_java.getVmAvailablePes(), dtype=np.int32),
            "waiting_cloudlets": local_obs_java.getWaitingCloudlets(),
            "next_cloudlet_pes": local_obs_java.getNextCloudletPes(),
        }

    return {
        "global": global_obs,
        "local": local_obs
    }
```

同样的修复应用到 `_parse_hierarchical_rewards()`:

```python
def _parse_hierarchical_rewards(self, result) -> Dict[str, Any]:
    """
    Parse hierarchical rewards from step result.
    """
    global_reward = result.getGlobalReward()

    local_rewards_java = result.getLocalRewards()
    local_rewards = {}

    # ✅ 修正：使用 keySet() 迭代
    for dc_id_obj in local_rewards_java.keySet():
        dc_id = int(dc_id_obj)
        local_rewards[dc_id] = float(local_rewards_java.get(dc_id_obj))

    return {
        "global": global_reward,
        "local": local_rewards
    }
```

---

### 问题 5: ⚠️ **GlobalBroker 方法缺失检查**

#### **位置**
- 需要检查 `GlobalBroker.java`

#### **问题描述**

Python 代码调用了 `getArrivingCloudlets(currentClock, timestepSize)`，但需要确认：
1. `GlobalBroker` 是否实现了这个方法
2. 方法签名是否正确
3. 返回值是否可以通过 Py4J 传递

#### **需要验证的方法**

```java
// GlobalBroker.java - 需要有这些方法
public List<Cloudlet> getArrivingCloudlets(double startTime, double endTime)
public int getRemainingCloudletCount()
public boolean routeCloudletToDatacenter(Cloudlet cloudlet, int targetDcIndex)
```

#### **检查步骤**

```bash
# 搜索 GlobalBroker 的实现
grep -n "getArrivingCloudlets" cloudsimplus-gateway/src/main/java/giu/edu/cspg/GlobalBroker.java
```

---

### 问题 6: ⚠️ **配置参数传递验证**

#### **位置**
- `HierarchicalMultiDCGateway.java` lines 44-55

#### **问题描述**

Python 传递 `config` 字典到 Java，需要确保：

```python
# Python 端
config = {
    "multi_datacenter_enabled": True,
    "datacenters": [
        {"datacenter_id": 0, "name": "DC0", "hosts_count": 16, ...},
        {"datacenter_id": 1, "name": "DC1", "hosts_count": 20, ...},
    ],
    "simulation_timestep": 1.0,
    "max_episode_length": 1000,
    ...
}
```

Java 端的 `configureSimulation(Map<String, Object> params)` 需要正确解析所有参数。

#### **潜在问题**

1. **嵌套数据结构**: `datacenters` 是 `List[Dict]`，Py4J 可能转换为 `List<Map<String, Object>>`
2. **类型转换**: Python `int` vs Java `Integer`, Python `bool` vs Java `Boolean`
3. **缺失字段**: 如果 config 缺少某些字段，Java 会用默认值还是抛异常？

#### **验证方法**

添加日志检查：

```java
// HierarchicalMultiDCGateway.java
public void configureSimulation(Map<String, Object> params) {
    LOGGER.info("Configuring simulation with params: {}", params.keySet());

    // 验证必需字段
    if (!params.containsKey("multi_datacenter_enabled")) {
        LOGGER.warn("Missing 'multi_datacenter_enabled' in config, defaulting to false");
    }

    Object dcListObj = params.get("datacenters");
    if (dcListObj == null) {
        LOGGER.warn("Missing 'datacenters' in config");
    } else {
        LOGGER.info("Datacenters config type: {}", dcListObj.getClass());
        LOGGER.info("Datacenters config: {}", dcListObj);
    }

    // ... 继续配置
}
```

---

### 问题 7: ⚠️ **错误处理不完整**

#### **位置**
- 多处

#### **问题描述**

#### **A. reset() 错误处理**

```python
# 当前代码
result = self.java_env.reset(seed if seed is not None else 0)

# 如果 Java 抛异常会怎样？
# 需要 try-catch
try:
    result = self.java_env.reset(seed if seed is not None else 0)
except Exception as e:
    logger.error(f"Java reset failed: {e}")
    # 重新连接？重试？
    raise
```

#### **B. step() 边界情况**

```python
# 问题1: 如果 num_arriving == 0？
num_arriving = self.java_env.getArrivingCloudletsCount()
if len(global_actions) > num_arriving:
    global_actions = global_actions[:num_arriving]

# 问题2: 如果 local_actions_map 是空的？
# 问题3: 如果 DC ID 不存在？
```

#### **C. 数组长度不匹配**

```python
# 如果 Java 返回的数组长度与预期不符？
global_obs = {
    "dc_green_power": np.array(global_obs_java.getVmLoads(), dtype=np.float32),
    # 假设应该是 num_datacenters 长度，但实际返回的不是？
}
```

#### **修复方案**

添加完整的错误处理和验证：

```python
def step(self, action: Dict[str, Any]):
    if self.java_env is None:
        raise RuntimeError("Environment not initialized. Call reset() first.")

    try:
        # Extract and validate actions
        global_actions = action.get("global", [])
        local_actions_map = action.get("local", {})

        # Validate global actions
        num_arriving = self.java_env.getArrivingCloudletsCount()
        if num_arriving < 0:
            raise ValueError(f"Invalid arriving cloudlets count: {num_arriving}")

        if len(global_actions) > num_arriving:
            logger.warning(f"Trimming global actions from {len(global_actions)} to {num_arriving}")
            global_actions = global_actions[:num_arriving]
        elif len(global_actions) < num_arriving:
            logger.warning(f"Not enough global actions: {len(global_actions)} < {num_arriving}")
            # Pad with default action (e.g., DC 0)
            global_actions = global_actions + [0] * (num_arriving - len(global_actions))

        # Validate local actions
        local_actions_java = {}
        for dc_id, vm_id in local_actions_map.items():
            if dc_id < 0 or dc_id >= self.num_datacenters:
                logger.warning(f"Invalid datacenter ID: {dc_id}, skipping")
                continue
            local_actions_java[int(dc_id)] = int(vm_id)

        # Execute step
        result = self.java_env.step(global_actions, local_actions_java)

        # Validate result
        if result is None:
            raise RuntimeError("Java step() returned None")

        # Parse results with validation
        observations = self._parse_hierarchical_observation(result)
        rewards = self._parse_hierarchical_rewards(result)
        terminated = bool(result.isTerminated())
        truncated = bool(result.isTruncated())
        info = self._parse_info(result)

        # Validate observation shapes
        self._validate_observation(observations)

        # Update state
        self.current_step += 1
        self.episode_reward += rewards["global"]
        self.done = terminated or truncated

        return observations, rewards, terminated, truncated, info

    except Exception as e:
        logger.error(f"Error in step(): {e}", exc_info=True)
        # Optionally: try to recover or raise
        raise

def _validate_observation(self, obs):
    """Validate observation structure and shapes"""
    global_obs = obs.get("global")
    local_obs = obs.get("local")

    if global_obs is None or local_obs is None:
        raise ValueError("Missing global or local observations")

    # Validate global obs shapes
    if global_obs["dc_green_power"].shape[0] != self.num_datacenters:
        raise ValueError(
            f"Invalid dc_green_power shape: {global_obs['dc_green_power'].shape}, "
            f"expected ({self.num_datacenters},)"
        )

    # Validate local obs count
    if len(local_obs) != self.num_datacenters:
        logger.warning(
            f"Expected {self.num_datacenters} local observations, got {len(local_obs)}"
        )
```

---

## 📊 问题优先级总结

| 问题 | 严重性 | 影响 | 修复难度 |
|------|--------|------|---------|
| **1. 全局观测字段映射错误** | P0 🔴 | 训练失败 | 简单 |
| **2. 局部动作空间设计错误** | P0 🔴 | 运行时错误 | 中等 |
| **3. Dict 观测空间训练困难** | P1 ⚠️ | 训练复杂 | 中等 |
| **4. Py4J Map 转换问题** | P1 ⚠️ | 运行时错误 | 简单 |
| **5. GlobalBroker 方法缺失** | P1 ⚠️ | 编译/运行时错误 | 需验证 |
| **6. 配置参数传递** | P1 ⚠️ | 初始化失败 | 需验证 |
| **7. 错误处理不完整** | P1 ⚠️ | 调试困难 | 中等 |

---

## 🔧 推荐修复顺序

### **阶段 1: 立即修复（必须）**
1. ✅ **修复问题 1**: 全局观测字段映射
2. ✅ **修复问题 2**: 局部动作空间设计

### **阶段 2: 高优先级（推荐）**
3. ✅ **修复问题 4**: Py4J Map 转换
4. ✅ **验证问题 5**: 检查 GlobalBroker 方法
5. ✅ **验证问题 6**: 配置参数传递

### **阶段 3: 优化（可选）**
6. ⚪ **改进问题 3**: 简化观测空间（或提供扁平化选项）
7. ⚪ **增强问题 7**: 完善错误处理

---

## 🧪 测试建议

### **单元测试**

```python
# test_hierarchical_env.py

def test_observation_fields_mapping():
    """测试观测字段映射正确性"""
    env = HierarchicalMultiDCEnv(config)
    obs, info = env.reset()

    # 检查全局观测
    assert "dc_green_power" in obs["global"]
    assert obs["global"]["dc_green_power"].shape == (num_datacenters,)

    # 检查局部观测
    assert len(obs["local"]) == num_datacenters
    for dc_id in range(num_datacenters):
        assert dc_id in obs["local"]

def test_action_space_structure():
    """测试动作空间结构"""
    env = HierarchicalMultiDCEnv(config)

    # 全局动作空间
    assert isinstance(env.global_action_space, spaces.MultiDiscrete)

    # 局部动作空间
    assert isinstance(env.local_action_space, (spaces.Dict, spaces.MultiDiscrete))

    # 采样测试
    global_action = env.global_action_space.sample()
    local_action = env.local_action_space.sample()

    # Step 测试
    obs, reward, done, truncated, info = env.step({
        "global": global_action,
        "local": local_action
    })

def test_py4j_map_conversion():
    """测试 Py4J Map 转换"""
    # Mock Java Map
    # 测试 keySet() 迭代
    # 测试 get() 访问
```

### **集成测试**

```python
def test_full_episode():
    """测试完整 episode"""
    env = HierarchicalMultiDCEnv(config)
    obs, info = env.reset(seed=42)

    for step in range(100):
        # 简单随机策略
        num_arriving = env.get_arriving_cloudlets_count()
        global_actions = [0] * num_arriving  # 全部路由到 DC 0
        local_actions = {i: 0 for i in range(env.num_datacenters)}  # 全部调度到 VM 0

        obs, rewards, terminated, truncated, info = env.step({
            "global": global_actions,
            "local": local_actions
        })

        if terminated or truncated:
            break

    env.close()
```

---

## 📝 其他建议

### **1. 添加调试模式**

```python
class HierarchicalMultiDCEnv(gym.Env):
    def __init__(self, config: Dict[str, Any]):
        # ...
        self.debug = config.get("debug", False)

    def step(self, action):
        if self.debug:
            logger.debug(f"Action: {action}")
            logger.debug(f"Num arriving: {self.java_env.getArrivingCloudletsCount()}")

        result = self.java_env.step(...)

        if self.debug:
            logger.debug(f"Result: {result}")
            logger.debug(f"Global reward: {result.getGlobalReward()}")
```

### **2. 创建环境工厂**

```python
def make_hierarchical_multidc_env(config_path: str, experiment_id: str):
    """
    Environment factory with validation
    """
    config = load_config(config_path, experiment_id)

    # Validate config
    assert "multi_datacenter_enabled" in config
    assert config["multi_datacenter_enabled"] == True
    assert "datacenters" in config
    assert len(config["datacenters"]) > 1

    # Create environment
    env = HierarchicalMultiDCEnv(config)

    # Optional: wrap with monitors
    env = Monitor(env, ...)

    return env
```

### **3. 文档完善**

创建 `HIERARCHICAL_MARL_USAGE_GUIDE.md`，包含：
- 环境接口说明
- 观测空间详解
- 动作空间详解
- 示例训练脚本
- 常见问题 FAQ

---

## ✅ 检查清单

在运行 multi-DC 训练前，请确认：

- [ ] **问题 1** 已修复：全局观测字段映射正确
- [ ] **问题 2** 已修复：局部动作空间使用 Dict 或 MultiDiscrete
- [ ] **问题 4** 已修复：Py4J Map 使用 keySet() 迭代
- [ ] **问题 5** 已验证：GlobalBroker 有所需方法
- [ ] **问题 6** 已验证：配置参数正确传递
- [ ] 单元测试通过
- [ ] 集成测试通过
- [ ] 可以完成至少 1 个完整 episode
- [ ] Java 日志无 ERROR
- [ ] Python 日志无 WARNING（除了预期的）

---

需要我立即开始修复这些问题吗？我建议按照优先级顺序逐个修复并测试。
