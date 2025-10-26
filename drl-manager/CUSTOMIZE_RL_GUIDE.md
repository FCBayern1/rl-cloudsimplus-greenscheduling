# 强化学习算法自定义完整指南

本指南涵盖所有可以自定义 RL 算法的层面，从简单到复杂。

---

## 📋 目录

1. [Level 1: 配置文件自定义](#level-1-配置文件自定义) ⭐ 最简单
2. [Level 2: 网络架构自定义](#level-2-网络架构自定义) ⭐⭐
3. [Level 3: 自定义特征提取器](#level-3-自定义特征提取器) ⭐⭐⭐
4. [Level 4: 自定义 Policy](#level-4-自定义-policy) ⭐⭐⭐⭐
5. [Level 5: 自定义算法](#level-5-自定义算法) ⭐⭐⭐⭐⭐
6. [Level 6: 自定义奖励塑造](#level-6-自定义奖励塑造) ⭐⭐
7. [Level 7: 自定义 Callback](#level-7-自定义-callback) ⭐⭐
8. [实用案例集合](#实用案例集合)

---

## Level 1: 配置文件自定义 ⭐

**难度**: 最简单，只需修改 `config.yml`

### 1.1 超参数调整

你**已经在做**的：

```yaml
experiment_3:
  # 算法超参数
  learning_rate: 0.0005
  n_steps: 512
  batch_size: 128
  n_epochs: 15
  gamma: 0.995
  gae_lambda: 0.98
  clip_range: 0.2
  ent_coef: 0.02
  vf_coef: 0.5
  max_grad_norm: 0.5

  # 奖励权重
  reward_wait_time_coef: 2.0
  reward_queue_penalty_coef: 1.0
  # ...
```

### 1.2 添加自定义网络架构（简单版）

**在 `config.yml` 中添加**:

```yaml
experiment_3_custom_network:
  simulation_name: "Exp3_CustomNetwork"
  experiment_name: "exp3_custom_net"
  experiment_type_dir: "CustomNN"

  # ... 其他配置 ...

  # 🌟 自定义网络架构
  policy_kwargs:
    net_arch:
      # Actor 网络: [256, 256, 128]
      # Critic 网络: [256, 256, 128]
      pi: [256, 256, 128]  # Policy (Actor) network
      vf: [256, 256, 128]  # Value Function (Critic) network

    # 激活函数
    activation_fn: "torch.nn.ReLU"  # 可选: Tanh, LeakyReLU, ELU

    # 网络初始化
    ortho_init: true  # 正交初始化（推荐用于 PPO）
```

**支持的网络架构**:

```yaml
# 方案 1: 浅层网络（快速训练）
policy_kwargs:
  net_arch:
    pi: [128, 128]
    vf: [128, 128]

# 方案 2: 深层网络（更强表达能力）
policy_kwargs:
  net_arch:
    pi: [512, 512, 256, 128]
    vf: [512, 512, 256, 128]

# 方案 3: Actor 和 Critic 不同架构
policy_kwargs:
  net_arch:
    pi: [256, 256]       # Actor: 更简单
    vf: [512, 512, 256]  # Critic: 更复杂（更准确的价值估计）

# 方案 4: 共享层 + 独立层
policy_kwargs:
  net_arch:
    - 256  # 共享层
    - 256  # 共享层
    - dict(pi=[128, 64], vf=[128, 64])  # 独立层
```

### 1.3 使用不同激活函数

```yaml
policy_kwargs:
  net_arch:
    pi: [256, 256]
    vf: [256, 256]

  # 激活函数选项
  activation_fn: "torch.nn.Tanh"      # Tanh (传统)
  # activation_fn: "torch.nn.ReLU"    # ReLU (最常用)
  # activation_fn: "torch.nn.LeakyReLU"  # LeakyReLU
  # activation_fn: "torch.nn.ELU"     # ELU (更平滑)
```

---

## Level 2: 网络架构自定义 ⭐⭐

**难度**: 中等，需要修改 Python 代码

### 2.1 创建自定义网络架构文件

**创建文件**: `drl-manager/mnt/custom_networks.py`

```python
import torch
import torch.nn as nn
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class CustomMLP(BaseFeaturesExtractor):
    """
    自定义多层感知机（MLP）特征提取器

    用途: 从观察空间提取特征，用于 Actor 和 Critic 网络
    """

    def __init__(self, observation_space, features_dim=256):
        super(CustomMLP, self).__init__(observation_space, features_dim)

        # 计算输入维度
        # observation_space 是 Dict，包含 vm_loads, vm_available_pes, waiting_cloudlets, next_cloudlet_pes
        input_dim = 0
        for key, space in observation_space.spaces.items():
            input_dim += space.shape[0]

        # 定义网络层
        self.network = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.1),  # 添加 Dropout 防止过拟合

            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        # 将 Dict 观察展平成一个向量
        obs_list = []
        for key in sorted(observations.keys()):
            obs_list.append(observations[key])

        # Concatenate along feature dimension
        flat_obs = torch.cat(obs_list, dim=1)

        return self.network(flat_obs)


class AttentionNetwork(BaseFeaturesExtractor):
    """
    带注意力机制的特征提取器

    用途: 让模型关注重要的 VM（例如负载高的 VM）
    """

    def __init__(self, observation_space, features_dim=256):
        super(AttentionNetwork, self).__init__(observation_space, features_dim)

        # VM 特征编码器
        self.vm_encoder = nn.Sequential(
            nn.Linear(2, 64),  # vm_load + vm_available_pes
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 注意力机制
        self.attention = nn.MultiheadAttention(
            embed_dim=128,
            num_heads=4,
            batch_first=True
        )

        # Cloudlet 特征编码器
        self.cloudlet_encoder = nn.Sequential(
            nn.Linear(2, 64),  # waiting_cloudlets + next_cloudlet_pes
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # 融合网络
        self.fusion = nn.Sequential(
            nn.Linear(256, 256),  # 128 (VM) + 128 (Cloudlet)
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations['vm_loads'].shape[0]
        num_vms = observations['vm_loads'].shape[1]

        # 1. 编码 VM 特征 (batch, num_vms, 2)
        vm_features = torch.stack([
            observations['vm_loads'],
            observations['vm_available_pes'].float()
        ], dim=2)

        vm_encoded = self.vm_encoder(vm_features)  # (batch, num_vms, 128)

        # 2. 应用自注意力（VM 之间的关系）
        vm_attended, _ = self.attention(
            vm_encoded, vm_encoded, vm_encoded
        )  # (batch, num_vms, 128)

        # 3. 聚合 VM 特征（平均池化）
        vm_aggregated = vm_attended.mean(dim=1)  # (batch, 128)

        # 4. 编码 Cloudlet 特征
        cloudlet_features = torch.cat([
            observations['waiting_cloudlets'],
            observations['next_cloudlet_pes']
        ], dim=1)  # (batch, 2)

        cloudlet_encoded = self.cloudlet_encoder(cloudlet_features)  # (batch, 128)

        # 5. 融合所有特征
        combined = torch.cat([vm_aggregated, cloudlet_encoded], dim=1)  # (batch, 256)

        return self.fusion(combined)  # (batch, features_dim)


class ResidualNetwork(BaseFeaturesExtractor):
    """
    带残差连接的深层网络

    用途: 训练更深的网络，避免梯度消失
    """

    def __init__(self, observation_space, features_dim=256):
        super(ResidualNetwork, self).__init__(observation_space, features_dim)

        # 计算输入维度
        input_dim = sum([space.shape[0] for space in observation_space.spaces.values()])

        # 输入投影
        self.input_proj = nn.Linear(input_dim, 256)

        # 残差块
        self.res_block1 = self._make_res_block(256, 256)
        self.res_block2 = self._make_res_block(256, 256)
        self.res_block3 = self._make_res_block(256, 256)

        # 输出层
        self.output = nn.Linear(256, features_dim)

    def _make_res_block(self, in_features, out_features):
        """创建残差块"""
        return nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features),
            nn.ReLU()
        )

    def forward(self, observations):
        # 展平观察
        obs_list = [observations[key] for key in sorted(observations.keys())]
        flat_obs = torch.cat(obs_list, dim=1)

        # 输入投影
        x = self.input_proj(flat_obs)

        # 残差连接
        x = x + self.res_block1(x)
        x = x + self.res_block2(x)
        x = x + self.res_block3(x)

        return self.output(x)
```

### 2.2 在训练脚本中使用自定义网络

**修改 `train.py`** (在 line 192 附近):

```python
# --- Model Instantiation ---
policy = "MultiInputPolicy" if isinstance(env.observation_space, spaces.Dict) else "MlpPolicy"

# 🌟 使用自定义网络
from custom_networks import CustomMLP, AttentionNetwork, ResidualNetwork

# 方式 1: 使用自定义特征提取器
policy_kwargs = {
    "features_extractor_class": AttentionNetwork,  # 使用注意力网络
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [256, 256],  # Actor/Critic 的后续网络
    "activation_fn": torch.nn.ReLU,
}

# 方式 2: 从配置文件读取（如果有的话）
if params.get("policy_kwargs"):
    policy_kwargs = params.get("policy_kwargs")
```

**在 config.yml 中配置**:

```yaml
experiment_3_attention:
  simulation_name: "Exp3_AttentionNetwork"
  experiment_name: "exp3_attention"

  # 使用自定义网络（需要在 train.py 中添加逻辑）
  custom_network: "AttentionNetwork"  # 或 "CustomMLP", "ResidualNetwork"

  # 网络参数
  policy_kwargs:
    features_dim: 256
    net_arch: [256, 256]
```

---

## Level 3: 自定义特征提取器 ⭐⭐⭐

**难度**: 中高级，需要理解强化学习架构

### 3.1 为什么需要自定义特征提取器？

你的观察空间是 **Dict**:
```python
{
  "vm_loads": [35个 VM 的负载],
  "vm_available_pes": [35个 VM 的可用 PEs],
  "waiting_cloudlets": [队列中的任务数],
  "next_cloudlet_pes": [下一个任务需要的 PEs]
}
```

**默认行为**: Stable-Baselines3 会简单地把所有值拉平成一个长向量
**问题**: 这样丢失了 VM 之间的关系信息

**解决方案**: 自定义特征提取器，理解 VM 的结构

### 3.2 创建 VM-Aware 特征提取器

```python
class VMAwareExtractor(BaseFeaturesExtractor):
    """
    专门为 VM 调度设计的特征提取器

    特点:
    1. 理解 VM 的类型（Small/Medium/Large）
    2. 考虑 VM 之间的关系
    3. 提取任务-VM 匹配特征
    """

    def __init__(self, observation_space, features_dim=256):
        super(VMAwareExtractor, self).__init__(observation_space, features_dim)

        # 从配置中获取 VM 数量和类型
        # 假设: 20 Small + 10 Medium + 5 Large = 35 VMs
        self.num_small_vms = 20
        self.num_medium_vms = 10
        self.num_large_vms = 5
        self.total_vms = 35

        # 为每种类型的 VM 创建单独的编码器
        self.small_vm_encoder = nn.Sequential(
            nn.Linear(2, 32),  # load + available_pes
            nn.ReLU(),
            nn.Linear(32, 64),
            nn.ReLU()
        )

        self.medium_vm_encoder = nn.Sequential(
            nn.Linear(2, 48),
            nn.ReLU(),
            nn.Linear(48, 64),
            nn.ReLU()
        )

        self.large_vm_encoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 全局 VM 状态编码器
        self.global_vm_encoder = nn.Sequential(
            nn.Linear(64 * 3, 128),  # Small + Medium + Large features
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        # 任务编码器
        self.task_encoder = nn.Sequential(
            nn.Linear(2, 64),  # waiting + pes_required
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU()
        )

        # 最终融合
        self.fusion = nn.Sequential(
            nn.Linear(128 + 64, 256),  # VM global + Task
            nn.ReLU(),
            nn.Linear(256, features_dim),
            nn.ReLU()
        )

    def forward(self, observations):
        batch_size = observations['vm_loads'].shape[0]

        # 1. 分别处理不同类型的 VM
        vm_loads = observations['vm_loads']
        vm_pes = observations['vm_available_pes'].float()

        # Small VMs (前 20 个)
        small_vms = torch.stack([
            vm_loads[:, :self.num_small_vms],
            vm_pes[:, :self.num_small_vms]
        ], dim=2)  # (batch, 20, 2)
        small_features = self.small_vm_encoder(small_vms).mean(dim=1)  # (batch, 64)

        # Medium VMs (中间 10 个)
        start_idx = self.num_small_vms
        end_idx = start_idx + self.num_medium_vms
        medium_vms = torch.stack([
            vm_loads[:, start_idx:end_idx],
            vm_pes[:, start_idx:end_idx]
        ], dim=2)  # (batch, 10, 2)
        medium_features = self.medium_vm_encoder(medium_vms).mean(dim=1)  # (batch, 64)

        # Large VMs (最后 5 个)
        start_idx = end_idx
        large_vms = torch.stack([
            vm_loads[:, start_idx:],
            vm_pes[:, start_idx:]
        ], dim=2)  # (batch, 5, 2)
        large_features = self.large_vm_encoder(large_vms).mean(dim=1)  # (batch, 64)

        # 2. 融合 VM 特征
        vm_global = torch.cat([small_features, medium_features, large_features], dim=1)
        vm_global = self.global_vm_encoder(vm_global)  # (batch, 128)

        # 3. 处理任务特征
        task_features = torch.cat([
            observations['waiting_cloudlets'],
            observations['next_cloudlet_pes']
        ], dim=1)
        task_encoded = self.task_encoder(task_features)  # (batch, 64)

        # 4. 最终融合
        combined = torch.cat([vm_global, task_encoded], dim=1)

        return self.fusion(combined)
```

### 3.3 使用方法

在 `train.py` 中:

```python
from custom_networks import VMAwareExtractor

policy_kwargs = {
    "features_extractor_class": VMAwareExtractor,
    "features_extractor_kwargs": {"features_dim": 256},
    "net_arch": [256, 128],  # Actor 和 Critic 的后续层
}
```

---

## Level 4: 自定义 Policy ⭐⭐⭐⭐

**难度**: 高级，需要深入理解 PPO 算法

### 4.1 为什么需要自定义 Policy？

**场景**:
- 需要特殊的输出分布（例如，偏向选择空闲的 VM）
- 需要添加辅助输出（例如，同时预测任务完成时间）
- 需要修改 Actor-Critic 架构

### 4.2 创建自定义 Policy

**创建文件**: `drl-manager/mnt/custom_policies.py`

```python
import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import CategoricalDistribution
from gymnasium import spaces


class LoadBalancingPolicy(ActorCriticPolicy):
    """
    专门为负载均衡设计的 Actor-Critic Policy

    特点:
    1. Actor 输出动作概率时，考虑 VM 的可用性
    2. Critic 同时预测价值和任务完成时间（辅助任务）
    """

    def __init__(self, *args, **kwargs):
        super(LoadBalancingPolicy, self).__init__(*args, **kwargs)

        # 辅助头: 预测任务完成时间
        self.completion_time_head = nn.Sequential(
            nn.Linear(self.mlp_extractor.latent_dim_vf, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # 输出预测的完成时间
        )

    def forward(self, obs, deterministic=False):
        """
        重写 forward 方法，添加自定义逻辑
        """
        # 提取特征
        features = self.extract_features(obs)

        # 分别计算 Actor 和 Critic 的 latent
        latent_pi, latent_vf = self.mlp_extractor(features)

        # --- Actor (策略网络) ---
        # 计算动作分布
        distribution = self._get_action_dist_from_latent(latent_pi)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        # --- Critic (价值网络) ---
        values = self.value_net(latent_vf)

        # --- 辅助任务: 预测完成时间 ---
        predicted_completion_time = self.completion_time_head(latent_vf)

        return actions, values, log_prob, predicted_completion_time

    def evaluate_actions(self, obs, actions):
        """
        评估动作（训练时使用）
        """
        features = self.extract_features(obs)
        latent_pi, latent_vf = self.mlp_extractor(features)

        distribution = self._get_action_dist_from_latent(latent_pi)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()

        values = self.value_net(latent_vf)
        predicted_completion_time = self.completion_time_head(latent_vf)

        return values, log_prob, entropy, predicted_completion_time


class BiasedActorPolicy(ActorCriticPolicy):
    """
    带偏置的 Actor Policy

    特点: 动作输出时，偏向选择负载低的 VM
    """

    def __init__(self, *args, **kwargs):
        super(BiasedActorPolicy, self).__init__(*args, **kwargs)

    def _get_action_dist_from_latent(self, latent_pi):
        """
        重写此方法，添加偏置
        """
        # 原始 logits
        logits = self.action_net(latent_pi)

        # 🌟 添加偏置: 根据 VM 负载调整 logits
        # 注意: 这里需要访问观察中的 vm_loads
        # 实际实现中，需要将 vm_loads 作为类成员传递进来

        # 示例: 假设我们有 vm_loads
        # vm_loads = self.current_obs['vm_loads']  # (batch, num_vms)
        # bias = -vm_loads * 2.0  # 负载越高，bias 越负（不倾向选择）
        # logits = logits + bias

        return CategoricalDistribution(logits)
```

### 4.3 使用自定义 Policy

**修改 `train.py`**:

```python
from custom_policies import LoadBalancingPolicy

# 创建模型时指定自定义 Policy
model = MaskablePPO(
    policy=LoadBalancingPolicy,  # 使用自定义 Policy
    env=env,
    learning_rate=params.get("learning_rate", 3e-4),
    n_steps=params.get("n_steps", 2048),
    # ... 其他参数
)
```

---

## Level 5: 自定义算法 ⭐⭐⭐⭐⭐

**难度**: 最高级，需要深入理解 RL 算法原理

### 5.1 创建自定义 PPO 变体

**创建文件**: `drl-manager/mnt/custom_algorithms.py`

```python
from sb3_contrib import MaskablePPO
import torch
import numpy as np


class AdaptivePPO(MaskablePPO):
    """
    自适应 PPO 算法

    特点:
    1. 动态调整 clip_range（根据训练进度）
    2. 动态调整 entropy 系数（先探索后利用）
    3. 添加辅助损失函数
    """

    def __init__(self, *args, **kwargs):
        super(AdaptivePPO, self).__init__(*args, **kwargs)

        # 自适应参数
        self.initial_clip_range = self.clip_range
        self.initial_ent_coef = self.ent_coef
        self.min_ent_coef = 0.001  # 最小熵系数

    def _update_learning_rate(self, optimizers):
        """
        重写学习率更新，同时更新 clip_range 和 ent_coef
        """
        # 调用父类方法
        super()._update_learning_rate(optimizers)

        # 计算训练进度 (0 到 1)
        progress = self.num_timesteps / self._total_timesteps

        # 🌟 自适应 Clip Range (线性衰减)
        if callable(self.clip_range):
            # 如果 clip_range 已经是函数，使用它
            self.clip_range_vf = self.clip_range(progress)
        else:
            # 线性衰减: 从初始值衰减到 0.1
            self.clip_range_vf = self.initial_clip_range * (1 - 0.9 * progress)

        # 🌟 自适应 Entropy Coefficient (指数衰减)
        # 前期高熵（多探索），后期低熵（多利用）
        self.ent_coef = self.initial_ent_coef * np.exp(-3 * progress)
        self.ent_coef = max(self.ent_coef, self.min_ent_coef)

        # 打印当前值（调试用）
        if self.num_timesteps % 10000 == 0:
            print(f"Progress: {progress:.2f}, "
                  f"Clip Range: {self.clip_range_vf:.4f}, "
                  f"Ent Coef: {self.ent_coef:.4f}")

    def train(self):
        """
        重写训练方法，添加自定义损失
        """
        # 调用父类的训练方法
        super().train()

        # 🌟 可以在这里添加额外的训练逻辑
        # 例如: 辅助任务的损失、正则化项等


class PrioritizedPPO(MaskablePPO):
    """
    带优先级经验回放的 PPO

    特点: 更频繁地训练高 TD error 的样本
    """

    def __init__(self, *args, alpha=0.6, beta=0.4, **kwargs):
        super(PrioritizedPPO, self).__init__(*args, **kwargs)

        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样修正

    def train(self):
        """
        使用优先级经验回放训练
        """
        # 获取当前 rollout buffer
        rollout_buffer = self.rollout_buffer

        # 计算 TD error 作为优先级
        with torch.no_grad():
            # ... 计算每个样本的 TD error
            # td_errors = ...
            pass

        # 根据优先级采样
        # priorities = np.abs(td_errors) ** self.alpha
        # probabilities = priorities / priorities.sum()

        # ... 使用 probabilities 进行采样和训练

        # 为了简化，这里调用父类方法
        return super().train()
```

### 5.2 使用自定义算法

**修改 `train.py`**:

```python
from custom_algorithms import AdaptivePPO, PrioritizedPPO

# 使用自适应 PPO
model = AdaptivePPO(
    policy="MultiInputPolicy",
    env=env,
    learning_rate=params.get("learning_rate", 3e-4),
    n_steps=params.get("n_steps", 2048),
    # ... 其他参数
)
```

---

## Level 6: 自定义奖励塑造 ⭐⭐

**难度**: 中等，需要理解奖励工程

### 6.1 在 Python 端添加奖励塑造

**创建文件**: `drl-manager/gym_cloudsimplus/reward_shaping.py`

```python
import numpy as np


class RewardShaper:
    """
    奖励塑造器 - 在环境端处理奖励
    """

    def __init__(self, config):
        self.config = config
        self.prev_waiting_cloudlets = 0
        self.prev_vm_loads = None

    def shape_reward(self, reward, obs, info):
        """
        对原始奖励进行塑造

        Args:
            reward: Java 端返回的原始奖励
            obs: 当前观察
            info: 额外信息

        Returns:
            shaped_reward: 塑造后的奖励
        """
        shaped_reward = reward

        # 🌟 Potential-based Shaping: 减少等待任务数
        waiting_cloudlets = obs['waiting_cloudlets'][0]
        if self.prev_waiting_cloudlets > 0:
            # 奖励: 减少等待任务
            queue_reduction_bonus = (self.prev_waiting_cloudlets - waiting_cloudlets) * 10.0
            shaped_reward += queue_reduction_bonus

        self.prev_waiting_cloudlets = waiting_cloudlets

        # 🌟 负载均衡奖励: 鼓励均匀分配
        vm_loads = obs['vm_loads']
        load_std = np.std(vm_loads)  # 负载标准差
        load_balance_bonus = -load_std * 5.0  # 标准差越小，奖励越高
        shaped_reward += load_balance_bonus

        # 🌟 进度奖励: 鼓励分配动作
        if info.get('assignment_success', False):
            shaped_reward += 1.0  # 成功分配任务

        return shaped_reward

    def reset(self):
        """重置塑造器状态"""
        self.prev_waiting_cloudlets = 0
        self.prev_vm_loads = None


class CurriculumRewardShaper:
    """
    课程学习奖励塑造

    特点: 训练初期给更密集的奖励，后期减少辅助奖励
    """

    def __init__(self, config):
        self.config = config
        self.episode_count = 0
        self.curriculum_threshold = 100  # 前 100 episodes 使用密集奖励

    def shape_reward(self, reward, obs, info):
        shaped_reward = reward

        # 训练初期: 密集奖励
        if self.episode_count < self.curriculum_threshold:
            # 每个成功分配都给奖励
            if info.get('assignment_success', False):
                shaped_reward += 5.0

            # 惩罚无效动作
            if info.get('invalid_action_taken', False):
                shaped_reward -= 2.0

        # 训练后期: 稀疏奖励（只使用原始奖励）
        else:
            pass

        return shaped_reward

    def on_episode_end(self):
        """Episode 结束时调用"""
        self.episode_count += 1
```

### 6.2 在环境中使用奖励塑造

**修改 `loadbalancing_env.py`** 的 `step()` 方法:

```python
from reward_shaping import RewardShaper

class LoadBalancingEnv(gym.Env):

    def __init__(self, config_params, render_mode="ansi"):
        # ... 原有初始化代码 ...

        # 🌟 添加奖励塑造器
        self.reward_shaper = RewardShaper(config_params)

    def step(self, action):
        # ... 原有 step 代码 ...

        # 获取原始奖励
        reward = step_result_dict.get("totalReward", 0.0)

        # 🌟 应用奖励塑造
        shaped_reward = self.reward_shaper.shape_reward(
            reward=reward,
            obs=observation,
            info=info_dict
        )

        return observation, shaped_reward, terminated, truncated, info_dict

    def reset(self, seed=None, options=None):
        # ... 原有 reset 代码 ...

        # 🌟 重置奖励塑造器
        self.reward_shaper.reset()

        return observation, info_dict
```

---

## Level 7: 自定义 Callback ⭐⭐

**难度**: 中等，用于监控和干预训练

### 7.1 创建自定义 Callback

**创建文件**: `drl-manager/mnt/callbacks/custom_callbacks.py`

```python
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class AdaptiveLearningRateCallback(BaseCallback):
    """
    动态调整学习率的 Callback

    特点: 根据训练性能自动调整学习率
    """

    def __init__(self, initial_lr=3e-4, min_lr=1e-5, verbose=1):
        super(AdaptiveLearningRateCallback, self).__init__(verbose)
        self.initial_lr = initial_lr
        self.min_lr = min_lr
        self.best_mean_reward = -np.inf
        self.plateau_count = 0
        self.check_freq = 10  # 每 10 episodes 检查一次

    def _on_step(self):
        # 每隔一定步数检查
        if self.n_calls % 10000 == 0:
            # 获取最近的平均奖励
            if len(self.model.ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in self.model.ep_info_buffer])

                # 如果没有改进
                if mean_reward <= self.best_mean_reward:
                    self.plateau_count += 1

                    # 如果连续 3 次没有改进，降低学习率
                    if self.plateau_count >= 3:
                        current_lr = self.model.learning_rate
                        new_lr = max(current_lr * 0.5, self.min_lr)
                        self.model.learning_rate = new_lr

                        if self.verbose > 0:
                            print(f"Reducing learning rate: {current_lr:.6f} -> {new_lr:.6f}")

                        self.plateau_count = 0
                else:
                    self.best_mean_reward = mean_reward
                    self.plateau_count = 0

        return True


class SuccessRateMonitorCallback(BaseCallback):
    """
    监控快速策略成功率的 Callback

    特点: 实时监控 170 步完成的比例
    """

    def __init__(self, fast_strategy_steps=170, log_freq=1000, verbose=1):
        super(SuccessRateMonitorCallback, self).__init__(verbose)
        self.fast_strategy_steps = fast_strategy_steps
        self.log_freq = log_freq
        self.episode_lengths = []
        self.episode_count = 0

    def _on_step(self):
        # 检查是否有新的 episode 结束
        if len(self.model.ep_info_buffer) > self.episode_count:
            # 获取新完成的 episode
            new_episodes = self.model.ep_info_buffer[self.episode_count:]
            for ep in new_episodes:
                self.episode_lengths.append(ep['l'])

            self.episode_count = len(self.model.ep_info_buffer)

            # 每隔一定频率统计
            if self.episode_count % 10 == 0:
                recent_lengths = self.episode_lengths[-20:]  # 最近 20 episodes
                fast_count = sum(1 for l in recent_lengths if l == self.fast_strategy_steps)
                success_rate = fast_count / len(recent_lengths) * 100

                if self.verbose > 0:
                    print(f"\n[Episode {self.episode_count}] "
                          f"Fast Strategy Success Rate (last 20): {success_rate:.1f}%")

        return True


class EnergyEfficiencyCallback(BaseCallback):
    """
    监控能耗效率的 Callback

    特点: 记录和可视化能耗指标
    """

    def __init__(self, log_dir, verbose=1):
        super(EnergyEfficiencyCallback, self).__init__(verbose)
        self.log_dir = log_dir
        self.energy_data = []

    def _on_step(self):
        # 从 info 中获取能耗数据
        if len(self.model.ep_info_buffer) > 0:
            latest_ep = self.model.ep_info_buffer[-1]

            # 假设 info 中有能耗数据
            if 'cumulative_energy_wh' in latest_ep:
                self.energy_data.append({
                    'episode': len(self.model.ep_info_buffer),
                    'energy_wh': latest_ep['cumulative_energy_wh'],
                    'reward': latest_ep['r']
                })

        return True

    def _on_training_end(self):
        # 训练结束时保存能耗数据
        import pandas as pd
        df = pd.DataFrame(self.energy_data)
        csv_path = f"{self.log_dir}/energy_efficiency.csv"
        df.to_csv(csv_path, index=False)

        if self.verbose > 0:
            print(f"Energy efficiency data saved to: {csv_path}")
```

### 7.2 使用自定义 Callback

**修改 `train.py`**:

```python
from callbacks.custom_callbacks import (
    AdaptiveLearningRateCallback,
    SuccessRateMonitorCallback,
    EnergyEfficiencyCallback
)

# 添加 callbacks
callbacks = []

# 1. 原有的 best model callback
if params.get("save_experiment", False) and log_dir:
    save_best_callback = SaveOnBestTrainingRewardCallback(...)
    callbacks.append(save_best_callback)

# 2. 🌟 自适应学习率
adaptive_lr_callback = AdaptiveLearningRateCallback(
    initial_lr=params.get("learning_rate", 3e-4),
    min_lr=1e-5,
    verbose=1
)
callbacks.append(adaptive_lr_callback)

# 3. 🌟 成功率监控
success_rate_callback = SuccessRateMonitorCallback(
    fast_strategy_steps=170,
    log_freq=1000,
    verbose=1
)
callbacks.append(success_rate_callback)

# 4. 🌟 能耗监控
if log_dir:
    energy_callback = EnergyEfficiencyCallback(
        log_dir=log_dir,
        verbose=1
    )
    callbacks.append(energy_callback)

# 训练时传入所有 callbacks
model.learn(
    total_timesteps=total_timesteps,
    callback=callbacks,
    progress_bar=True
)
```

---

## 实用案例集合

### 案例 1: 提高快速策略成功率

**目标**: 从 14.5% 提升到 >80%

**方案**: 组合多个自定义

```yaml
# config.yml
experiment_fast_strategy:
  simulation_name: "Exp_FastStrategy"

  # 1. 使用注意力网络（关注 VM 负载）
  custom_network: "AttentionNetwork"

  # 2. 调整奖励权重
  reward_wait_time_coef: 3.0    # 大幅增加等待时间惩罚
  reward_queue_penalty_coef: 2.0

  # 3. 增加探索
  ent_coef: 0.10  # 高熵系数

  # 4. 课程学习
  max_episode_length: 600  # 先给足够时间

  timesteps: 300000  # 更长训练
```

**Python 端**:

```python
# 使用注意力网络 + 奖励塑造 + 自定义 callback
from custom_networks import AttentionNetwork
from reward_shaping import CurriculumRewardShaper
from callbacks.custom_callbacks import SuccessRateMonitorCallback

# 在环境初始化时
env.reward_shaper = CurriculumRewardShaper(config_params)

# 在训练时
callbacks.append(SuccessRateMonitorCallback(fast_strategy_steps=170))
```

### 案例 2: 降低 Critic Loss

**目标**: 从 2510 降低到 <100

**方案**:

```yaml
# config.yml
experiment_low_critic_loss:
  # 1. 增加 Critic 网络容量
  policy_kwargs:
    net_arch:
      pi: [256, 256]         # Actor 保持中等
      vf: [512, 512, 256]    # Critic 更大

  # 2. 增加 Value Function 权重
  vf_coef: 1.5  # 从 0.5 增加到 1.5

  # 3. 更多训练 epochs
  n_epochs: 25  # 从 15 增加到 25

  # 4. 更大 batch size
  batch_size: 256  # 从 128 增加到 256

  # 5. 更长训练
  timesteps: 400000
```

### 案例 3: 能耗优化

**目标**: 在保持性能的同时降低能耗

**方案**:

```python
# custom_networks.py
class EnergyAwareExtractor(BaseFeaturesExtractor):
    """
    能耗感知的特征提取器

    特点:
    - 编码 VM 的功耗信息
    - 优先选择能效高的 VM
    """

    def __init__(self, observation_space, features_dim=256):
        super(EnergyAwareExtractor, self).__init__(observation_space, features_dim)

        # VM 能效编码器（结合负载和功耗）
        self.vm_efficiency_encoder = nn.Sequential(
            nn.Linear(3, 64),  # load + pes + power_efficiency
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU()
        )

        # ... 其他层

    def forward(self, observations):
        # 计算 VM 能效特征
        # power_efficiency = power_consumption / utilization
        # ...
        pass
```

```yaml
# config.yml
experiment_green:
  custom_network: "EnergyAwareExtractor"

  reward_energy_coef: 2.0  # 高能耗惩罚
  reward_wait_time_coef: 1.0

  timesteps: 250000
```

---

## 📚 总结和建议

### 从简单到复杂的学习路径

**第 1 周**: Level 1 - 配置文件调参
- 熟练掌握超参数调整
- 理解每个参数的作用

**第 2-3 周**: Level 2 - 自定义网络架构
- 尝试不同的网络层数和宽度
- 使用 Dropout、Batch Normalization

**第 4-5 周**: Level 3 - 自定义特征提取器
- 理解观察空间的结构
- 设计特定于问题的编码器

**第 6-8 周**: Level 4-5 - 自定义 Policy 和算法
- 深入理解 PPO 原理
- 实现算法变体

**贯穿始终**: Level 6-7 - 奖励塑造和 Callback
- 持续优化奖励函数
- 使用 Callback 监控训练

### 当前你的问题的最佳自定义方案

**问题 1: 快速策略成功率低 (2.5%)**
→ 使用 **AttentionNetwork** + **CurriculumRewardShaper**

**问题 2: Critic Loss 高 (2510)**
→ 增加 **vf_coef** + 更大的 **Critic 网络** + 更多 **n_epochs**

**问题 3: 训练不稳定**
→ 使用 **AdaptivePPO** (自适应 clip range 和 entropy)

---

需要我帮你实现任何一个自定义方案吗？或者解释某个特定的概念？
