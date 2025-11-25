# 多智能体强化学习重构计划

> **状态**: 计划中
> **预计时间**: 6-10周
> **目标**: 从Gymnasium伪MARL迁移到RLlib真正的多智能体学习

---

## 📋 背景：架构问题分析

### 当前系统的核心问题

经过系统性分析，发现当前Gymnasium+SB3架构存在以下**关键缺陷**：

#### ❌ Critical Issues

1. **分层协调失效** (train_hierarchical_multidc_joint.py:211-370)
   - Global Agent训练时，Local actions是随机采样
   - Local Agent训练时，Global actions是随机采样
   - **结果**: 两层agent无法学习真正的协调策略

2. **奖励错位** (MultiDatacenterSimulationCore.java:696-970)
   - Global reward包含50%绿色能源奖励
   - Local reward **完全没有**绿色能源项
   - **结果**: Local agent不优化绿色能源，导致整体目标无法达成

3. **非真正MARL**
   - 当前实现是"交替单智能体训练"
   - 缺少MADDPG/QMIX/MAPPO等真正的MARL算法
   - 缺少中心化Critic和agent间通信

#### ⚠️ Major Issues

4. **观测空间不完整** (hierarchical_multidc_env.py)
   - Local agents无法看到绿色能源可用性
   - 即使添加绿色能源奖励，也无法做出绿色优化决策

5. **训练不稳定**
   - 交替训练导致非平稳环境（non-stationary）
   - 一个agent的策略变化会破坏另一个agent的学习

---

## 🎯 解决方案：迁移到 RLlib + MAPPO

### 为什么选择这个方案？

**RLlib优势**:
- ✅ 原生多智能体支持（Multi-Agent API）
- ✅ 内置MAPPO、QMIX、MADDPG算法
- ✅ 参数共享（Parameter Sharing）
- ✅ Ray分布式训练
- ✅ 与Gymnasium兼容

**MAPPO优势**:
- ✅ 适合合作式任务（Global + Local协同优化）
- ✅ 中心化训练 + 去中心化执行（CTDE）
- ✅ 处理部分可观测环境（POMDP）
- ✅ 稳定性好，收敛快

---

## 📅 分阶段实施计划

### 阶段 1: 评估基线 + 环境准备（Week 1）

#### ✅ 已完成：

**1.1 评估工具创建**
- 文件: `drl-manager/src/evaluation/metrics_evaluator.py`
- 功能:
  - 绿色能源利用率追踪
  - 任务性能指标（完成率、等待时间）
  - 训练稳定性分析（CV、趋势）
  - 基线对比功能
  - CLI工具: `python -m src.evaluation.metrics_evaluator <experiment_dir>`

**1.2 可视化工具创建**
- 文件: `drl-manager/src/evaluation/visualize_training.py`
- 功能:
  - 训练进度图（rewards over time）
  - 绿色能源优化图（利用率、浪费量）
  - 多智能体协调分析图（global vs local correlation）
  - CLI工具: `python -m src.evaluation.visualize_training <experiment_dir>`

#### 🔄 进行中：

**1.2 基线测试运行**
- 当前训练: `logs/experiment_multi_dc_3/20251110_235733`
- 配置: Gymnasium + SB3 + 交替训练
- 进度: Cycle 2/10（20,000/200,000 timesteps）
- 完成后保存到: `baselines/gymnasium_alternating/`

**1.3 RLlib依赖安装**
```bash
pip install "ray[rllib]==2.9.0"
```
- 状态: 后台安装中

---

### 阶段 2: 环境适配 RLlib（Week 2-3）

#### 2.1 创建 RLlib MultiAgentEnv 适配器

**文件**: `drl-manager/gym_cloudsimplus/envs/rllib_multidc_env.py`（新建）

**关键改动**:
1. 继承 `ray.rllib.env.MultiAgentEnv`
2. 返回格式: `{agent_id: obs}` 而非单一obs
3. 定义 agent IDs:
   - `"global_agent"`
   - `"local_agent_dc0"`, `"local_agent_dc1"`, `"local_agent_dc2"`
4. 实现:
   - `observation_space_dict`
   - `action_space_dict`
   - `observation_space_contains()`
   - `action_space_contains()`

**示例代码结构**:
```python
class RLlibHierarchicalMultiDCEnv(MultiAgentEnv):
    def __init__(self, config):
        super().__init__()
        # 复用现有的 HierarchicalMultiDCEnv 作为后端
        self.base_env = HierarchicalMultiDCEnv(config)

        # 定义 agent IDs
        self._agent_ids = {"global_agent"}
        for i in range(self.base_env.num_datacenters):
            self._agent_ids.add(f"local_agent_dc{i}")

    def reset(self, *, seed=None, options=None):
        # 返回 {agent_id: obs} 格式
        ...

    def step(self, action_dict):
        # 接收 {agent_id: action}
        # 返回 obs_dict, reward_dict, done_dict, truncated_dict, info_dict
        ...
```

#### 2.2 修复 Local Agent 观测空间（添加绿色能源）

**Java端修改**:

**文件**: `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/ObservationState.java`
```java
// 添加字段
private final double currentGreenPowerW;
private final double currentPowerW;
private final double greenRatio;

// 添加getter方法
public double getCurrentGreenPowerW() { return currentGreenPowerW; }
public double getCurrentPowerW() { return currentPowerW; }
public double getGreenRatio() { return greenRatio; }
```

**文件**: `MultiDatacenterSimulationCore.java` (line 583: buildLocalObservation)
```java
return new ObservationState(
    // ... 现有字段 ...
    dc.getCurrentGreenPowerW(currentClock),
    dc.getCurrentPowerW(),
    dc.getGreenEnergyRatio()
);
```

**Python端修改**:

**文件**: `hierarchical_multidc_env.py` (line 191: local_observation_space)
```python
self.local_observation_space = spaces.Dict({
    # ... 现有字段 ...
    "current_green_power_w": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
    "current_power_w": spaces.Box(low=0, high=10000, shape=(1,), dtype=np.float32),
    "green_ratio": spaces.Box(low=0, high=1, shape=(1,), dtype=np.float32),
})
```

**文件**: `hierarchical_multidc_env.py` (_convert_local_observation)
```python
def _convert_local_observation(self, dc_id, obs_state):
    return {
        # ... 现有字段 ...
        "current_green_power_w": np.array([obs_state.getCurrentGreenPowerW()], dtype=np.float32),
        "current_power_w": np.array([obs_state.getCurrentPowerW()], dtype=np.float32),
        "green_ratio": np.array([obs_state.getGreenRatio()], dtype=np.float32),
    }
```

#### 2.3 修复 Local Agent 奖励函数（添加绿色能源项）

**Java端修改**:

**文件**: `MultiDatacenterSimulationCore.java` (line 887: calculateSingleLocalReward)
```java
private double calculateSingleLocalReward(DatacenterInstance dc) {
    // 现有组件
    double waitTimePenalty = ...;
    double utilizationPenalty = ...;
    double queuePenalty = ...;
    double completionBonus = ...;

    // 新增：绿色能源组件
    double greenEnergyReward = 0.0;
    EnergyMetricsDelta delta = dc.getLatestEnergyDelta();
    if (delta != null) {
        double greenRatio = delta.getGreenRatio();
        double wastePenalty = delta.getNormalizedWastePenalty();
        greenEnergyReward = greenRatio * 0.5 - wastePenalty * 0.3;
    }

    double totalReward = waitTimePenalty + utilizationPenalty +
                         queuePenalty + completionBonus +
                         greenEnergyReward * settings.getRewardEnergyCoef();
    return totalReward;
}
```

**配置文件修改**:

**文件**: `config.yml`
```yaml
# 在 datacenter_configs 下添加
local_agents:
  reward_energy_coef: 0.5  # 绿色能源奖励权重
```

---

### 阶段 3: 实现 MAPPO 训练（Week 4-5）

#### 3.1 创建 MAPPO 训练脚本

**文件**: `drl-manager/src/training/train_mappo.py`（新建）

```python
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.policy.policy import PolicySpec

# 配置 MAPPO
config = (
    PPOConfig()
    .environment(env="hierarchical_multidc")
    .framework("torch")
    .rollouts(num_rollout_workers=4)
    .multi_agent(
        policies={
            # 全局策略
            "global_policy": PolicySpec(
                observation_space=env.observation_space_dict["global_agent"],
                action_space=env.action_space_dict["global_agent"],
            ),
            # 局部策略（参数共享）
            "local_policy": PolicySpec(
                observation_space=env.observation_space_dict["local_agent_dc0"],
                action_space=env.action_space_dict["local_agent_dc0"],
            ),
        },
        policy_mapping_fn=lambda agent_id, episode, worker, **kwargs: (
            "global_policy" if agent_id == "global_agent" else "local_policy"
        ),
        policies_to_train=["global_policy", "local_policy"],
    )
    .training(
        train_batch_size=4096,
        sgd_minibatch_size=128,
        num_sgd_iter=10,
        lr=3e-4,
        gamma=0.99,
        lambda_=0.95,
        entropy_coeff=0.01,
        clip_param=0.2,
    )
    .resources(num_gpus=1)
)

# 训练
algo = config.build()
for i in range(1000):
    result = algo.train()
    print(f"Iteration {i}: reward={result['episode_reward_mean']}")

    if i % 10 == 0:
        checkpoint = algo.save()
```

#### 3.2 配置中心化Critic（MAPPO核心）

**文件**: `drl-manager/src/models/centralized_critic.py`（新建）

实现中心化Value Function:
- Critic看到所有agents的观测和动作
- Actors只看到各自的观测
- 使用`TorchCentralizedCriticModel`基类

```python
config.training(
    model={
        "custom_model": "centralized_critic_model",
        "custom_model_config": {
            "central_vf_share_layers": False,
        },
    }
)
```

#### 3.3 集成现有回调系统

适配 `SaveOnBestRewardHierarchicalCallback` 到 RLlib callback API:
```python
from ray.rllib.algorithms.callbacks import DefaultCallbacks

class HierarchicalMACallback(DefaultCallbacks):
    def on_episode_end(self, *, worker, base_env, policies, episode, **kwargs):
        # 保存绿色能源指标
        # 保存最佳模型
        # 写入CSV
        ...
```

---

### 阶段 4: 实验验证和调优（Week 6-7）

#### 4.1 对比实验

| Baseline | 描述 | 预期绿色能源利用率 |
|----------|------|-------------------|
| Random | 随机策略 | ~50% |
| Greedy | 总是选绿色能源最多的DC | ~60% |
| Gymnasium (current) | 交替训练 | ~60% (Local不优化绿色) |
| **MAPPO (target)** | 真正MARL | **>80%** |

#### 4.2 关键指标追踪

**绿色能源指标**:
- 绿色能源利用率（目标: >80%）
- 棕色能源使用量（目标: 最小化）
- 绿色能源浪费量（目标: <10%）

**性能指标**:
- 任务等待时间（目标: 保持低）
- 完成率（目标: >95%）
- 负载均衡（目标: variance低）

**训练指标**:
- 收敛速度
- 稳定性（reward CV < 0.1）
- Sample efficiency

#### 4.3 超参数调优

使用Ray Tune进行自动调优:
```python
from ray import tune

config = PPOConfig().training(
    lr=tune.grid_search([1e-4, 3e-4, 1e-3]),
    entropy_coeff=tune.grid_search([0.0, 0.01, 0.05]),
    train_batch_size=tune.choice([2048, 4096, 8192]),
)
```

---

### 阶段 5: 高级优化（Week 8-9）

#### 5.1 添加通信机制（可选）

实现CommNet风格的agent间通信:
- Global agent向Local agents发送"建议"
- Local agents上报负载状态
- 共享embedding层

#### 5.2 改进奖励塑形

- Intrinsic motivation（好奇心奖励）
- Auxiliary tasks（辅助任务）
- Reward shaping（基于专家知识）

#### 5.3 分布式训练加速

利用Ray的分布式能力:
```python
config.rollouts(
    num_rollout_workers=8,
    num_envs_per_worker=4,
)
```

---

### 阶段 6: 文档和部署（Week 10）

#### 6.1 更新文档

创建以下文档:
- `docs/MULTI_DC_MAPPO_TRAINING.md` - MAPPO训练指南
- `docs/ARCHITECTURE_COMPARISON.md` - 新旧架构对比
- `docs/GREEN_ENERGY_OPTIMIZATION.md` - 绿色能源优化原理

#### 6.2 创建评估脚本

**文件**: `drl-manager/evaluate_mappo.py`
```bash
python evaluate_mappo.py --checkpoint <path> --episodes 100
```

#### 6.3 对比报告

生成详细的对比报告:
- 绿色能源指标改善百分比
- 训练时间和资源消耗
- 可视化图表

---

## 📊 预期收益

### 绿色能源优化
- **当前问题**: Local不优化绿色能源 → 利用率 <60%
- **MAPPO预期**: 协同优化 → 利用率 >80%
- **额外收益**: 减少浪费 10-20%

### 训练稳定性
- **当前问题**: 交替训练，reward震荡
- **MAPPO预期**: 联合训练，收敛更快更稳定

### 架构清晰度
- **当前问题**: "伪MARL"，难以扩展
- **MAPPO预期**: 真正MARL框架，易于添加新agents

---

## ⚠️ 风险和缓解

| 风险 | 缓解措施 |
|------|---------|
| RLlib学习曲线陡峭 | 先完成toy example，再迁移完整系统 |
| 性能可能不如预期 | 保留Gymnasium基线，逐步迁移 |
| 调试困难（分布式） | 先单worker训练，确认正确性 |
| 时间超预期 | 分阶段交付，每阶段可独立验证 |

---

## 🔄 回滚计划

如果MAPPO效果不好:
1. **保留修复**: 观测和奖励的改进（阶段2.2、2.3）是独立的
2. **改进基线**: 回到Gymnasium，但使用改进的联合训练策略
3. **尝试其他算法**: QMIX、MADDPG

---

## 📝 关键文件清单

### 新建文件
- `drl-manager/gym_cloudsimplus/envs/rllib_multidc_env.py` - RLlib环境适配器
- `drl-manager/src/training/train_mappo.py` - MAPPO训练脚本
- `drl-manager/src/models/centralized_critic.py` - 中心化Critic
- `drl-manager/src/evaluation/metrics_evaluator.py` - ✅ 已创建
- `drl-manager/src/evaluation/visualize_training.py` - ✅ 已创建
- `drl-manager/evaluate_mappo.py` - 评估脚本

### 修改文件
- `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/ObservationState.java`
- `cloudsimplus-gateway/src/main/java/giu/edu/cspg/multidc/MultiDatacenterSimulationCore.java`
- `drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py`
- `config.yml`

---

## 📈 进度跟踪

- [x] 阶段1.1: 创建评估指标模块
- [x] 阶段1.1: 创建可视化工具
- [ ] 阶段1.2: 运行基线测试（进行中 - Cycle 2/10）
- [ ] 阶段1.3: 安装RLlib依赖（进行中）
- [ ] 阶段2.1: 创建RLlib适配器
- [ ] 阶段2.2: 修复Local观测空间
- [ ] 阶段2.3: 修复Local奖励函数
- [ ] 阶段3.1: MAPPO训练脚本
- [ ] 阶段3.2: 中心化Critic
- [ ] 阶段3.3: 回调系统集成
- [ ] 阶段4: 实验验证
- [ ] 阶段5: 高级优化
- [ ] 阶段6: 文档和部署

---

**最后更新**: 2025-11-11
**当前阶段**: Week 1 - 评估基线 + 环境准备
