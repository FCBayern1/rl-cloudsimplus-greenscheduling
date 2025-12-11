# 实现计划：Local Agents 参数共享

## 目标
在 RLlib 多数据中心训练中，为所有 Local Agents 实现参数共享，通过 padding 统一异构数据中心的观测空间。

**范围**: 仅修改 RLlib 训练流程 (`train_rllib_multidc.py`)，保留 God's Eye 特征。

---

## 当前架构 vs 目标架构

### 当前（独立策略）
```
Global Agent ─────► global_policy (独立)

Local Agents:
├─ local_agent_0 ─► local_policy_0 (独立, Discrete(225))
├─ local_agent_1 ─► local_policy_1 (独立, Discrete(169))
├─ local_agent_2 ─► local_policy_2 (独立, Discrete(175))
│   ...
└─ local_agent_9 ─► local_policy_9 (独立, Discrete(197))
```

### 目标（参数共享）
```
Global Agent ─────► global_policy (独立)

Local Agents:
├─ local_agent_0 ─┐
├─ local_agent_1 ─┤
├─ local_agent_2 ─┼───► shared_local_policy (共享, Discrete(max_vms+1))
│   ...           │     所有 local agents 共享同一个神经网络
└─ local_agent_9 ─┘
```

---

## 实现方案：全参数共享 + 观测空间 Padding

### 为什么选择这个方案？
1. **样本效率最大化**: 10 个 local agents 的经验共同训练一个策略
2. **泛化能力强**: 策略学习跨不同规模 DC 的通用调度能力
3. **架构简单**: 一个神经网络，易于维护
4. **RLlib 原生支持**: 只需修改 `policy_mapping_fn`

---

## 统一观测空间设计

### 当前观测空间（DC-specific，大小不同）
```python
{
    "observation": {
        "host_loads": Box(shape=(dc_host_count,)),      # 12-24 变化
        "host_ram_usage": Box(shape=(dc_host_count,)),  # 12-24 变化
        "vm_loads": Box(shape=(dc_vm_count,)),          # 84-252 变化
        "vm_types": Box(shape=(dc_vm_count,)),
        "vm_available_pes": Box(shape=(dc_vm_count,)),
        "waiting_cloudlets": Discrete(100000),
        "next_cloudlet_pes": Discrete(256),
    },
    "action_mask": Box(shape=(dc_vm_count + 1,))
}
```

### 统一后观测空间（Padded，所有 DC 相同）
```python
{
    "observation": {
        # Padded 到最大值，用 0 填充
        "host_loads": Box(shape=(max_hosts,)),          # = 24
        "host_ram_usage": Box(shape=(max_hosts,)),      # = 24
        "vm_loads": Box(shape=(max_vms,)),              # = 252
        "vm_types": Box(shape=(max_vms,)),              # = 252
        "vm_available_pes": Box(shape=(max_vms,)),      # = 252
        "waiting_cloudlets": Discrete(100000),
        "next_cloudlet_pes": Discrete(256),

        # 新增 DC 上下文特征
        "dc_id_onehot": Box(shape=(num_dcs,)),          # 标识当前 DC
        "valid_vm_mask": Box(shape=(max_vms,)),         # 1=真实VM, 0=padding
    },
    "action_mask": Box(shape=(max_vms + 1,))            # = 253
}
```

### Action Space 统一
```python
# 所有 local agents 使用相同的 action space
Discrete(max_vms + 1)  # = Discrete(253)

# 通过 action_mask 限制有效动作：
# DC4 (252 VMs): mask[0:253] = [1,1,1,...,1]  # 全部有效
# DC7 (84 VMs):  mask[0:253] = [1,1,..,1,0,0,...,0]  # 85后全为0
```

---

## 实现步骤

### Step 1: 修改 PettingZoo 环境（观测空间 padding）

**文件**: `drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_pettingzoo.py`

```python
# 1. 添加统一维度计算
def __init__(self, config):
    # ... 现有代码 ...
    self.max_hosts = max(self._get_dc_host_count(i) for i in range(self.num_dcs))
    self.max_vms = max(self._get_dc_vm_count(i) for i in range(self.num_dcs))
    self.max_actions = self.max_vms + 1

# 2. 创建统一观测空间
def _create_unified_local_obs_space(self):
    return spaces.Dict({
        "observation": spaces.Dict({
            "host_loads": spaces.Box(0, 1, (self.max_hosts,), np.float32),
            "host_ram_usage": spaces.Box(0, 1, (self.max_hosts,), np.float32),
            "vm_loads": spaces.Box(0, 1, (self.max_vms,), np.float32),
            "vm_types": spaces.Box(0, 3, (self.max_vms,), np.int32),
            "vm_available_pes": spaces.Box(0, 100, (self.max_vms,), np.int32),
            "waiting_cloudlets": spaces.Discrete(100000),
            "next_cloudlet_pes": spaces.Discrete(256),
            "dc_id_onehot": spaces.Box(0, 1, (self.num_dcs,), np.float32),
            "valid_vm_mask": spaces.Box(0, 1, (self.max_vms,), np.float32),
        }),
        "action_mask": spaces.Box(0, 1, (self.max_actions,), np.float32)
    })

# 3. 观测转换函数（padding）
def _transform_local_obs_to_unified(self, dc_id, local_obs, action_mask):
    dc_vm_count = self.dc_vm_counts[dc_id]
    dc_host_count = self.dc_host_counts[dc_id]

    # Pad host observations
    host_loads = np.zeros(self.max_hosts, dtype=np.float32)
    host_loads[:dc_host_count] = local_obs["host_loads"][:dc_host_count]

    # Pad VM observations
    vm_loads = np.zeros(self.max_vms, dtype=np.float32)
    vm_loads[:dc_vm_count] = local_obs["vm_loads"][:dc_vm_count]

    # Valid VM mask
    valid_vm_mask = np.zeros(self.max_vms, dtype=np.float32)
    valid_vm_mask[:dc_vm_count] = 1.0

    # DC ID one-hot
    dc_id_onehot = np.zeros(self.num_dcs, dtype=np.float32)
    dc_id_onehot[dc_id] = 1.0

    # Pad action mask
    unified_action_mask = np.zeros(self.max_actions, dtype=np.float32)
    unified_action_mask[:dc_vm_count + 1] = action_mask[:dc_vm_count + 1]

    return {
        "observation": {
            "host_loads": host_loads,
            "host_ram_usage": host_ram,
            "vm_loads": vm_loads,
            "vm_types": vm_types,
            "vm_available_pes": vm_pes,
            "waiting_cloudlets": local_obs["waiting_cloudlets"],
            "next_cloudlet_pes": local_obs["next_cloudlet_pes"],
            "dc_id_onehot": dc_id_onehot,
            "valid_vm_mask": valid_vm_mask,
        },
        "action_mask": unified_action_mask
    }
```

### Step 2: 修改训练脚本（参数共享策略）

**文件**: `drl-manager/src/training/train_rllib_multidc.py`

```python
# 1. 修改 policy_mapping_fn
def policy_mapping_fn(agent_id, episode, **kwargs):
    if agent_id == "global_agent":
        return "global_policy"
    else:
        return "shared_local_policy"  # 所有 local agents 共享

# 2. 修改 policies 定义
policies = {
    "global_policy": PolicySpec(
        observation_space=global_obs_space,
        action_space=global_action_space,
        config=global_model_cfg,
    ),
    # 单一共享策略
    "shared_local_policy": PolicySpec(
        observation_space=unified_local_obs_space,  # 统一空间
        action_space=unified_local_action_space,    # Discrete(max_vms+1)
        config=masked_model_cfg,
    ),
}
```

### Step 3: 添加 Action 映射验证

**文件**: `drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_pettingzoo.py`

```python
def _flat_to_hierarchical_actions(self, flat_actions):
    local_actions = {}
    for i in range(self.num_datacenters):
        agent_name = f"local_agent_{i}"
        if agent_name in flat_actions:
            action = int(flat_actions[agent_name])
            dc_vm_count = self.dc_vm_counts[i]

            # 安全检查：action 不应超过 DC 的 VM 数量
            if action > dc_vm_count:
                logger.warning(f"Action {action} > DC{i} vm_count {dc_vm_count}")
                action = 0  # 回退到 NoAssign

            local_actions[i] = action
    return {"global": flat_actions.get("global_agent"), "local": local_actions}
```

---

## 关键文件修改清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_pettingzoo.py` | **主要修改** | 统一观测空间、padding 逻辑 |
| `drl-manager/src/training/train_rllib_multidc.py` | **主要修改** | policy_mapping_fn、PolicySpec |
| `drl-manager/src/models/masked_action_model.py` | 验证兼容性 | 确保支持统一空间 |
| `config.yml` | 可选 | 添加 parameter_sharing 配置开关 |

---

## 预期收益

1. **参数量减少**: 从 10 个独立网络 → 1 个共享网络（~90% 参数减少）
2. **样本效率**: 10x 更多的训练数据更新同一个策略
3. **泛化能力**: 策略学习跨 DC 规模的通用调度策略
4. **训练稳定性**: 更多样本 → 更稳定的梯度估计

---

## 潜在问题与缓解

| 问题 | 缓解措施 |
|------|---------|
| Action space 过大（253 vs 实际 85-253） | Action masking 确保只采样有效动作 |
| Padding 值（0）可能被误解 | 添加 `valid_vm_mask` 明确标识 |
| 不同规模 DC 可能需要不同策略 | 添加 `dc_id_onehot` 提供 DC 上下文 |

---

## 测试计划

1. **单元测试**: 验证 padding 正确性
2. **集成测试**: 验证 policy_mapping_fn 返回 "shared_local_policy"
3. **训练验证**: 短期训练确保无报错，检查各 DC 的 reward 曲线
