# Error Handling Improvements for HierarchicalMultiDCEnv

## 问题描述

用户发现`HierarchicalMultiDCEnv`的错误处理机制不够robust，与`LoadBalancingEnv`相比缺少重试机制。

## 改进对比

### 改进前（脆弱）

```python
def _connect_to_java(self):
    if self.gateway is None:
        try:
            self.gateway = JavaGateway(...)
            self.java_env = self.gateway.entry_point
            self.java_env.configureSimulation(self.config)
        except Exception as e:  # ❌ 捕获所有异常
            logger.error(f"Failed to connect: {e}")
            raise RuntimeError(...) from e  # ❌ 直接失败，没有重试！
```

**问题：**
- ❌ 没有重试机制
- ❌ 没有连接测试
- ❌ 配置失败时不清理资源
- ❌ 错误信息不够详细

### 改进后（Robust）

```python
def _connect_to_java(self):
    if self.gateway is None:
        max_retries = self.config.get("gateway_max_retries", 5)
        retry_delay = self.config.get("gateway_retry_delay", 5.0)

        retries = max_retries
        while retries > 0:
            try:
                # ✅ 尝试连接
                self.gateway = JavaGateway(...)

                # ✅ 测试连接
                self.gateway.jvm.System.out.println(
                    f"Python HierarchicalMultiDCEnv connected on port {self.py4j_port}!"
                )

                self.java_env = self.gateway.entry_point
                logger.info("Successfully connected to Java gateway")
                break  # ✅ 成功，退出重试循环

            except (ConnectionRefusedError, Py4JNetworkError) as e:
                # ✅ 捕获特定异常
                retries -= 1
                if retries > 0:
                    logger.warning(f"Gateway connection failed: {e}. "
                                   f"Retrying in {retry_delay} seconds... "
                                   f"({retries} retries left)")
                    time.sleep(retry_delay)  # ✅ 等待后重试
                else:
                    logger.error("Max retries reached.")
                    raise RuntimeError(
                        f"Could not connect after {max_retries} attempts. "
                        f"Make sure Java gateway is running:\n"
                        f"  cd cloudsimplus-gateway && ./gradlew run"
                    ) from e

            except Exception as e:
                # ✅ 意外错误不重试
                logger.error(f"Unexpected error: {e}")
                raise RuntimeError(f"Unexpected error: {e}") from e

        # ✅ 配置仿真（分离的错误处理）
        try:
            logger.info("Configuring multi-datacenter simulation...")
            self.java_env.configureSimulation(self.config)
            logger.info("Simulation configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure simulation: {e}")
            self._cleanup_gateway()  # ✅ 失败时清理资源
            raise RuntimeError(
                "Failed to configure simulation. Check Java logs."
            ) from e
```

## 完整改进列表

### 1. 连接错误处理（`_connect_to_java()`）

#### 新增功能：
- ✅ **重试机制**：默认5次重试，可通过配置调整
- ✅ **重试延迟**：每次重试等待5秒，可配置
- ✅ **连接测试**：通过`System.out.println()`测试连接
- ✅ **特定异常捕获**：区分`ConnectionRefusedError`和`Py4JNetworkError`
- ✅ **详细日志**：显示剩余重试次数
- ✅ **清晰的错误信息**：提供具体的解决方案（如何启动Java）
- ✅ **资源清理**：配置失败时调用`_cleanup_gateway()`

#### 配置参数：
```yaml
# config.yml中可选配置
gateway_max_retries: 5        # 最大重试次数（默认5）
gateway_retry_delay: 5.0      # 重试延迟秒数（默认5.0）
```

### 2. Reset方法错误处理

#### 改进：
```python
def reset(self, *, seed=None, options=None):
    super().reset(seed=seed)

    # ✅ 连接Java（带重试）
    self._connect_to_java()

    # ✅ Reset仿真（错误处理）
    try:
        logger.debug(f"Resetting Java simulation with seed {seed}...")
        result = self.java_env.reset(seed if seed is not None else 0)
    except Exception as e:
        logger.error(f"Failed to reset Java simulation: {e}")
        raise RuntimeError(
            "Failed to reset multi-datacenter simulation. "
            "Check Java logs for details."
        ) from e

    # Reset episode state
    self.current_step = 0
    self.episode_reward = 0.0
    self.done = False

    # ✅ 解析结果（错误处理）
    try:
        observations = self._parse_hierarchical_observation(result)
        info = self._parse_info(result)
    except Exception as e:
        logger.error(f"Failed to parse reset result: {e}")
        raise RuntimeError(
            "Failed to parse observations from Java. "
            "Check observation structure compatibility."
        ) from e

    logger.info(f"Environment reset successfully (seed={seed})")
    return observations, info
```

#### 新增：
- ✅ 详细的错误日志
- ✅ 分离的错误处理（reset失败 vs 解析失败）
- ✅ 清晰的错误信息
- ✅ 成功日志

### 3. Step方法错误处理

#### 改进：
```python
def step(self, action):
    # ✅ 检查环境是否初始化
    if self.java_env is None:
        raise RuntimeError(
            "Environment not initialized. Call reset() first."
        )

    # ✅ 验证action格式
    try:
        global_actions = action.get("global", [])
        local_actions_map = action.get("local", {})

        if not isinstance(global_actions, (list, np.ndarray)):
            raise ValueError(
                f"'global' actions must be list/array, got {type(global_actions)}"
            )
        if not isinstance(local_actions_map, dict):
            raise ValueError(
                f"'local' actions must be dict, got {type(local_actions_map)}"
            )
    except Exception as e:
        logger.error(f"Invalid action format: {e}")
        raise ValueError(
            "Invalid action format. Expected dict with 'global' and 'local' keys."
        ) from e

    # ✅ 获取arriving cloudlets（错误处理）
    try:
        num_arriving = self.java_env.getArrivingCloudletsCount()
    except Exception as e:
        logger.error(f"Failed to get arriving cloudlets count: {e}")
        num_arriving = 0  # ✅ 失败时使用默认值

    # ✅ 转换local actions（错误处理）
    try:
        local_actions_java = {}
        for dc_id, vm_id in local_actions_map.items():
            local_actions_java[int(dc_id)] = int(vm_id)
    except Exception as e:
        logger.error(f"Failed to convert local actions: {e}")
        raise ValueError(
            "Invalid local action format. DC/VM IDs must be integers."
        ) from e

    # ✅ 执行step（错误处理）
    try:
        logger.debug(f"Executing step {self.current_step + 1}...")
        result = self.java_env.step(global_actions, local_actions_java)
    except Exception as e:
        logger.error(f"Failed to execute step: {e}")
        raise RuntimeError(
            "Failed to execute simulation step. Check Java logs."
        ) from e

    # ✅ 解析结果（错误处理）
    try:
        observations = self._parse_hierarchical_observation(result)
        rewards = self._parse_hierarchical_rewards(result)
        terminated = result.isTerminated()
        truncated = result.isTruncated()
        info = self._parse_info(result)
    except Exception as e:
        logger.error(f"Failed to parse step result: {e}")
        raise RuntimeError(
            "Failed to parse step results. "
            "Check observation/reward structure compatibility."
        ) from e

    # Update episode state
    self.current_step += 1
    self.episode_reward += rewards["global"]
    self.done = terminated or truncated

    return observations, rewards, terminated, truncated, info
```

#### 新增：
- ✅ Action格式验证
- ✅ 类型检查（list/array, dict）
- ✅ 多层次错误处理
- ✅ 降级处理（如获取cloudlets count失败时用0）
- ✅ 详细的错误信息

### 4. 资源清理方法

#### 新增`_cleanup_gateway()`：
```python
def _cleanup_gateway(self):
    """
    Clean up gateway connection resources.
    """
    if self.gateway is not None:
        try:
            self.gateway.close()
            logger.info("Java gateway connection closed")
        except Exception as e:
            logger.warning(f"Error closing gateway: {e}")
        finally:
            self.gateway = None
            self.java_env = None
```

#### 用途：
- 配置失败时清理资源
- 防止资源泄漏
- 保证清理操作的原子性（finally块）

### 5. Close方法改进

#### 改进：
```python
def close(self):
    """
    Close the environment and cleanup resources.
    """
    # Close Java simulation environment
    if self.java_env is not None:
        try:
            logger.info("Closing Java simulation environment...")
            self.java_env.close()
            logger.info("Java simulation environment closed successfully")
        except Exception as e:
            logger.warning(f"Error closing Java environment: {e}")

    # Shutdown Py4J gateway
    if self.gateway is not None:
        try:
            logger.info("Shutting down Py4J gateway...")
            self.gateway.shutdown()
            logger.info("Py4J gateway shutdown successfully")
        except Exception as e:
            logger.warning(f"Error shutting down gateway: {e}")
        finally:
            self.gateway = None
            self.java_env = None
```

#### 新增：
- ✅ 详细的日志（开始、成功、失败）
- ✅ finally块确保资源置空
- ✅ 分离Java环境和Gateway的关闭

## 使用示例

### 基本使用（自动重试）

```python
import gymnasium as gym
import gym_cloudsimplus

# 创建环境
env = gym.make("HierarchicalMultiDC-v0", config=config)

# Reset会自动重试连接（最多5次）
try:
    obs, info = env.reset()
except RuntimeError as e:
    print(f"Failed to initialize: {e}")
    # 检查Java是否启动：cd cloudsimplus-gateway && ./gradlew run
```

### 自定义重试配置

```python
config = {
    "multi_datacenter_enabled": True,
    "py4j_port": 25333,
    "gateway_max_retries": 10,     # 更多重试
    "gateway_retry_delay": 3.0,     # 更短延迟
    "datacenters": [...],
}

env = gym.make("HierarchicalMultiDC-v0", config=config)
```

### 错误处理最佳实践

```python
# 1. 连接失败
try:
    obs, info = env.reset()
except RuntimeError as e:
    print(f"Connection failed: {e}")
    # 提示：确保Java gateway正在运行
    # cd cloudsimplus-gateway && ./gradlew run

# 2. Action格式错误
try:
    obs, rewards, term, trunc, info = env.step({
        "global": [0, 1, 2],  # 必须是list/array
        "local": {0: 5, 1: 12}  # 必须是dict
    })
except ValueError as e:
    print(f"Invalid action format: {e}")

# 3. 仿真执行失败
try:
    obs, rewards, term, trunc, info = env.step(action)
except RuntimeError as e:
    print(f"Simulation error: {e}")
    # 检查Java日志获取详情
```

## 错误类型说明

### RuntimeError
- 连接失败（经过所有重试后）
- Reset失败
- Step执行失败
- 解析结果失败

**处理方式：** 检查Java是否运行，查看Java日志

### ValueError
- Action格式错误
- Action类型错误
- 配置参数错误

**处理方式：** 检查action格式，确保符合API

### ConnectionRefusedError / Py4JNetworkError
- Java gateway未启动
- 端口被占用
- 网络问题

**处理方式：** 启动Java gateway，检查端口配置

## 日志级别

### INFO级别日志：
- ✅ 连接成功
- ✅ 配置完成
- ✅ Reset成功
- ✅ Close完成

### DEBUG级别日志：
- 🔍 执行每个step
- 🔍 Step详细信息（reward, terminated, truncated）

### WARNING级别日志：
- ⚠️ 重试连接
- ⚠️ 清理资源时出错

### ERROR级别日志：
- ❌ 连接失败
- ❌ 配置失败
- ❌ Reset失败
- ❌ Step失败
- ❌ 解析失败

## 总结

### 改进对比表

| 特性 | 改进前 | 改进后 |
|-----|-------|--------|
| **重试机制** | ❌ 无 | ✅ 5次重试 + 可配置 |
| **连接测试** | ❌ 无 | ✅ System.out.println测试 |
| **特定异常** | ❌ 捕获所有Exception | ✅ 区分ConnectionRefused等 |
| **资源清理** | ❌ 配置失败不清理 | ✅ _cleanup_gateway() |
| **错误信息** | ❌ 简单 | ✅ 详细 + 解决方案 |
| **Action验证** | ❌ 无 | ✅ 类型和格式检查 |
| **降级处理** | ❌ 无 | ✅ cloudlets count失败用0 |
| **日志层次** | ⚠️ 简单 | ✅ INFO/DEBUG/WARNING/ERROR |
| **文档化** | ❌ 简单docstring | ✅ 详细docstring + Raises说明 |

### 现在与LoadBalancingEnv一致

HierarchicalMultiDCEnv现在具有与LoadBalancingEnv同等的robust错误处理机制，甚至更好：

- ✅ 重试机制
- ✅ 连接测试
- ✅ 资源清理
- ✅ 详细日志
- ✅ Action验证
- ✅ 降级处理
- ✅ 可配置重试参数

### 用户体验改进

**改进前：**
```
❌ Failed to connect to Java gateway: Connection refused
RuntimeError: Could not connect to Java gateway
```

**改进后：**
```
⚠️ Gateway connection failed: Connection refused.
   Retrying in 5 seconds... (4 retries left)
⚠️ Gateway connection failed: Connection refused.
   Retrying in 5 seconds... (3 retries left)
✅ Successfully connected to Java gateway on port 25333
✅ Multi-datacenter simulation configured successfully
```

用户现在有时间启动Java，而不是立即失败！
