# Experiment 3 调试方案

## 📊 当前状态分析

### 基本信息
- **实验名称**: Exp3_CSV_QuickTest_Fixed
- **工作负载**: three_60max_8maxcores.csv (165个任务)
- **完成训练**: 117 episodes, 120000 timesteps
- **算法**: MaskablePPO
- **Learning Rate**: 0.0005

### 当前结果 (117 Episodes)

#### ✅ 成功发现
1. **快速完成策略**: 智能体发现了170步完成的策略
   - 快速策略成功率: **14.5%** (17/117 episodes)
   - 快速 episodes 平均奖励: **-469.16**
   - 正常 episodes 平均奖励: **-1032.42**
   - 快速策略比正常策略奖励高 **54.6%**

2. **总体改进**:
   - 平均奖励: -1037 → -936 (**+9.7%**)
   - Best episode: #28 (reward: -459.13)
   - 峰值性能区间: Episode 81-100

#### ⚠️ 发现的问题

1. **快速策略不稳定**
   - 成功率只有 14.5%，85.5% 的 episodes 仍然需要完整 1200 步
   - 说明智能体还没有充分学会这个策略
   - 可能陷入局部最优

2. **学习效率低**
   - Episode 长度 1200 步太长
   - 大部分时间浪费在等待任务完成
   - 样本效率 (sample efficiency) 差

3. **奖励函数问题** (从 training_report.txt)
   - Total reward 在最后 10 episodes 反而下降 (-5.14%)
   - Episode length 从 1097 增加到 1200 (变慢了)
   - Wait Time Penalty 虽然改进 100%，但基数太小 (-0.3679 → 0.0000)

4. **奖励组件失衡**
   - Wait Time Penalty: -0.3679 (太小，几乎没有影响)
   - Unutilization Penalty: -0.5084 (主导因素)
   - Queue Penalty: -0.0506 (太小)
   - 各组件的数量级差异太大

---

## 🎯 调试目标

### 主要目标
1. **提高快速策略成功率**: 从 14.5% 提升到 **>80%**
2. **缩短平均 episode 长度**: 从 1000+ 步降低到 **<300 步**
3. **提高学习效率**: 减少无效探索，加快收敛
4. **稳定训练**: 避免后期性能下降

### 次要目标
- 理解快速策略的机制
- 优化奖励函数的各组件平衡
- 找到最优的初始 VM 配置

---

## 🔬 调试方案

### Phase 1: 根本原因分析 (诊断阶段)

#### 1.1 分析快速 episodes 的特征
**目的**: 理解智能体是如何实现 170 步完成的

**步骤**:
```bash
# 1. 提取所有 170 步完成的 episodes
# 查看这些 episodes 的 Java 结果文件
ls -la cloudsimplus-gateway/results/Exp3_CSV_QuickTest_Fixed/episode_*/

# 2. 分析快速 episodes 的行为模式
# - VM 创建/删除模式
# - 任务分配策略
# - 资源利用率
```

**预期发现**:
- 快速策略可能是：一开始就创建足够的 VMs，快速分配所有任务
- 或者：智能体发现了某种高效的 VM-任务匹配方式

#### 1.2 检查奖励函数的实际值分布
**目的**: 确认各奖励组件的数量级是否合理

**步骤**:
```python
# 创建详细的奖励分析脚本
python drl-manager/analyze_rewards.py --log_dir logs/QuickTests/exp3_csv_quick
```

**需要检查**:
- 各组件的数值范围 (min, max, mean, std)
- 各组件对总奖励的贡献百分比
- 快速 vs 正常 episodes 的奖励差异

#### 1.3 分析学习曲线
**目的**: 找出性能下降的原因

**已知信息**:
- Episode 81-100 是峰值性能区间
- 之后性能下降 (后 10 episodes 奖励变差)

**可能原因**:
- Exploration-exploitation 失衡 (ent_coef = 0.02 可能太高)
- Learning rate decay 导致陷入局部最优
- PPO 的 clip range 限制了进一步优化

---

### Phase 2: 快速修复方案 (立即可试)

#### 方案 A: 缩短 Episode 长度 ⭐⭐⭐
**理由**: 1200 步太长，大量时间浪费在等待

**修改**:
```yaml
experiment_3_debug_v1:
  simulation_name: "Exp3_Debug_ShortEpisode"
  experiment_name: "exp3_debug_short"
  experiment_type_dir: "QuickTests"

  # 继承 experiment_3 的所有配置，只修改以下内容:
  max_episode_length: 300  # 1200 → 300 (缩短 75%)
  timesteps: 150000         # 120000 → 150000 (补偿更多 episodes)

  # 其他配置保持不变
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/three_60max_8maxcores.csv"
  algorithm: "MaskablePPO"
  learning_rate: 0.0005
  # ... (其他保持原样)
```

**预期效果**:
- 强制智能体在 300 步内完成任务
- 如果失败，episode 会被 truncate，获得负奖励
- 迫使智能体学习快速完成策略

**风险**:
- 如果 300 步不够完成所有任务，可能导致训练失败
- 需要观察前几个 episodes 的实际完成时间

---

#### 方案 B: 调整奖励函数权重 ⭐⭐⭐⭐
**理由**: 当前奖励组件失衡，需要重新平衡

**当前问题**:
- Wait Time Penalty: -0.3679 (太小，几乎不起作用)
- Unutilization Penalty: -0.5084 (主导，导致智能体过度关注利用率)

**修改**:
```yaml
experiment_3_debug_v2:
  simulation_name: "Exp3_Debug_RewardBalance"
  experiment_name: "exp3_debug_reward"
  experiment_type_dir: "QuickTests"

  # 奖励权重调整 (关键!)
  reward_wait_time_coef: 2.0          # 0.75 → 2.0 (增加 2.7x)
  reward_throughput_coef: 1.5         # 0.85 → 1.5 (增加吞吐量奖励)
  reward_unutilization_coef: 0.3      # 0.5 → 0.3 (降低，避免过度关注)
  reward_queue_penalty_coef: 1.0      # 0.55 → 1.0 (增加，减少排队)
  reward_invalid_action_coef: 2.0     # 保持

  # Episode 长度也缩短
  max_episode_length: 400
  timesteps: 150000

  # 其他保持不变
```

**预期效果**:
- 增加 wait time 惩罚 → 迫使智能体快速完成任务
- 增加 throughput 奖励 → 鼓励高吞吐量（快速完成更多任务）
- 降低 unutilization 惩罚 → 允许一定的资源浪费换取速度

---

#### 方案 C: 减少初始 VMs，让智能体学习扩展 ⭐⭐
**理由**: 当前初始 34 个 VMs 可能太多，智能体没有学习动态扩展

**修改**:
```yaml
experiment_3_debug_v3:
  simulation_name: "Exp3_Debug_FewVMs"
  experiment_name: "exp3_debug_few_vms"
  experiment_type_dir: "QuickTests"

  # 减少初始 VMs
  initial_s_vm_count: 10   # 20 → 10
  initial_m_vm_count: 5    # 10 → 5
  initial_l_vm_count: 2    # 4 → 2
  # 总共从 34 VMs → 17 VMs

  # 增加扩展激励
  reward_wait_time_coef: 1.5
  reward_queue_penalty_coef: 1.2  # 增加排队惩罚，迫使扩展

  max_episode_length: 400
  timesteps: 150000
```

**预期效果**:
- 强制智能体学习何时创建新 VMs
- 可能提高快速策略的成功率（因为需要主动扩展）

**风险**:
- 初始 VMs 太少可能导致大量排队，初期奖励很差
- 需要足够的探索才能学会创建 VMs

---

#### 方案 D: 增加探索 + 更长训练 ⭐⭐⭐
**理由**: 14.5% 成功率说明探索不足，没有充分发现快速策略

**修改**:
```yaml
experiment_3_debug_v4:
  simulation_name: "Exp3_Debug_MoreExploration"
  experiment_name: "exp3_debug_explore"
  experiment_type_dir: "QuickTests"

  # 增加探索
  ent_coef: 0.08            # 0.02 → 0.08 (4x 熵系数)

  # 更长训练
  timesteps: 200000         # 120000 → 200000

  # 降低学习率，更稳定
  learning_rate: 0.0003     # 0.0005 → 0.0003

  # 缩短 episode
  max_episode_length: 350

  # 奖励调整
  reward_wait_time_coef: 1.5
  reward_queue_penalty_coef: 1.0
```

**预期效果**:
- 更多探索 → 更大概率发现快速策略
- 更长训练 → 有足够时间学习并稳定快速策略

---

### Phase 3: 深度优化方案 (后续迭代)

#### 方案 E: 组合最优配置
**基于 Phase 2 的实验结果，组合最有效的改进**

**建议步骤**:
1. 运行方案 A, B, D (并行训练)
2. 对比结果，找出最有效的改进
3. 组合多个有效改进
4. 最终优化训练

#### 方案 F: Curriculum Learning (课程学习)
**渐进式训练策略**

**思路**:
1. **Stage 1 (Easy)**: max_episode_length=1200, 让智能体先学会完成任务
2. **Stage 2 (Medium)**: max_episode_length=600, 鼓励更快完成
3. **Stage 3 (Hard)**: max_episode_length=300, 强制快速完成

**实现**:
- 需要修改训练脚本，支持多阶段训练
- 或者手动训练 3 个阶段，后一阶段加载前一阶段的模型

---

## 📋 实验执行计划

### 优先级排序

#### 🥇 最高优先级 (立即执行)
1. **方案 B**: 调整奖励函数 (最简单，最可能有效)
2. **方案 D**: 增加探索 + 更长训练 (提高成功率)

#### 🥈 中等优先级 (结果不理想时尝试)
3. **方案 A**: 缩短 Episode 长度
4. **方案 C**: 减少初始 VMs

#### 🥉 低优先级 (深度优化)
5. **方案 E**: 组合优化
6. **方案 F**: Curriculum Learning

### 建议执行顺序

```bash
# Step 1: 运行方案 B (奖励调整)
python drl-manager/mnt/train.py --experiment experiment_3_debug_v2

# Step 2: 同时运行方案 D (更多探索)
python drl-manager/mnt/train.py --experiment experiment_3_debug_v4

# Step 3: 等待结果，分析对比
python drl-manager/analyze_training.py --log_dir logs/QuickTests/exp3_debug_reward
python drl-manager/analyze_training.py --log_dir logs/QuickTests/exp3_debug_explore

# Step 4: 根据结果决定下一步
# - 如果成功率 > 50%: 继续优化参数
# - 如果成功率 < 30%: 尝试方案 A 或 C
```

---

## 🔍 关键指标监控

### 训练过程中需要关注的指标

#### 1. 快速策略成功率
```bash
# 实时监控 (每 30 秒更新)
watch -n 30 'tail -50 logs/QuickTests/exp3_debug_*/monitor.csv | grep ",170," | wc -l'
```

**目标**:
- 前 50 episodes: > 20% (从 14.5% 提升)
- 100 episodes 后: > 50%
- 最终: > 80%

#### 2. Episode 完成时间分布
**监控**:
- 平均 episode 长度
- 中位数 episode 长度
- 170 步 episodes 的比例

**目标**:
- 平均 < 300 步
- 至少 80% episodes 在 300 步内完成

#### 3. 奖励改进趋势
**监控**:
- Rolling mean reward (窗口 10 episodes)
- Best reward 是否持续刷新
- 后期是否出现性能下降

**目标**:
- Best reward < -400 (当前 -459)
- 平均 reward < -500 (当前 -936)
- 无性能下降

#### 4. 奖励组件平衡
**监控**:
- 各组件的数值范围
- 各组件对总奖励的贡献比例

**目标**:
- Wait Time Penalty 应该是主要组件 (30-40%)
- Unutilization Penalty 不应主导 (< 30%)
- 各组件数量级接近 (同一个数量级)

---

## 💾 配置文件模板

### 完整配置 - 方案 B (推荐优先尝试)

```yaml
# 添加到 config.yml

experiment_3_debug_v2:
  simulation_name: "Exp3_Debug_RewardBalance"
  experiment_name: "exp3_debug_reward"
  experiment_type_dir: "QuickTests"

  # --- Workload ---
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/three_60max_8maxcores.csv"

  # --- Training ---
  mode: "train"
  algorithm: "MaskablePPO"
  timesteps: 150000          # 增加训练时间
  learning_rate: 0.0005      # 保持

  # --- PPO Hyperparameters ---
  n_steps: 512
  batch_size: 128
  n_epochs: 15
  gamma: 0.995
  gae_lambda: 0.98
  clip_range: 0.2
  ent_coef: 0.02             # 先保持，后续可调整

  # --- VM Fleet ---
  initial_s_vm_count: 20     # 保持
  initial_m_vm_count: 10     # 保持
  initial_l_vm_count: 4      # 保持

  # --- Reward Weights (关键调整!) ---
  reward_wait_time_coef: 2.0          # 0.75 → 2.0 ⬆️
  reward_throughput_coef: 1.5         # 0.85 → 1.5 ⬆️
  reward_unutilization_coef: 0.3      # 0.5 → 0.3 ⬇️
  reward_cost_coef: 0.3                # 0.5 → 0.3 ⬇️
  reward_queue_penalty_coef: 1.0      # 0.55 → 1.0 ⬆️
  reward_invalid_action_coef: 2.0     # 保持

  # --- Episode Configuration ---
  max_episode_length: 400    # 1200 → 400 (缩短 67%)

  # --- Other Settings ---
  seed: 43                   # 改变 seed
  save_experiment: true
  verbose: 1
  device: "auto"
```

### 完整配置 - 方案 D (更多探索)

```yaml
experiment_3_debug_v4:
  simulation_name: "Exp3_Debug_MoreExploration"
  experiment_name: "exp3_debug_explore"
  experiment_type_dir: "QuickTests"

  # --- Workload ---
  workload_mode: "CSV"
  cloudlet_trace_file: "traces/three_60max_8maxcores.csv"

  # --- Training ---
  mode: "train"
  algorithm: "MaskablePPO"
  timesteps: 200000          # 大幅增加
  learning_rate: 0.0003      # 降低 LR

  # --- PPO Hyperparameters ---
  n_steps: 512
  batch_size: 128
  n_epochs: 15
  gamma: 0.995
  gae_lambda: 0.98
  clip_range: 0.2
  ent_coef: 0.08             # 0.02 → 0.08 (4x 探索)

  # --- VM Fleet ---
  initial_s_vm_count: 20
  initial_m_vm_count: 10
  initial_l_vm_count: 4

  # --- Reward Weights ---
  reward_wait_time_coef: 1.5
  reward_throughput_coef: 1.2
  reward_unutilization_coef: 0.4
  reward_cost_coef: 0.4
  reward_queue_penalty_coef: 1.0
  reward_invalid_action_coef: 2.0

  # --- Episode Configuration ---
  max_episode_length: 350

  # --- Other Settings ---
  seed: 44
  save_experiment: true
  verbose: 1
  device: "auto"
```

---

## 📊 预期结果

### 成功标准

#### Tier 1 - 基本成功
- ✅ 快速策略成功率 > 50%
- ✅ 平均 episode 长度 < 400 步
- ✅ 平均 reward > -600

#### Tier 2 - 良好表现
- ✅ 快速策略成功率 > 70%
- ✅ 平均 episode 长度 < 300 步
- ✅ 平均 reward > -500
- ✅ Best reward < -400

#### Tier 3 - 优秀表现
- ✅ 快速策略成功率 > 90%
- ✅ 平均 episode 长度 < 200 步
- ✅ 平均 reward > -450
- ✅ Best reward < -380

### 失败指标
- ❌ 快速策略成功率 < 20% (没有改进)
- ❌ 平均 episode 长度仍然 > 1000 步
- ❌ 训练后期性能严重下降 (> 20%)

---

## 🚀 快速开始

### 1. 添加配置到 config.yml

```bash
# 将上面的 experiment_3_debug_v2 和 experiment_3_debug_v4 配置
# 复制粘贴到 config.yml 文件末尾
```

### 2. 运行调试实验

```bash
# 方案 B: 奖励平衡
python drl-manager/mnt/train.py --experiment experiment_3_debug_v2

# 方案 D: 更多探索 (可以并行运行)
python drl-manager/mnt/train.py --experiment experiment_3_debug_v4
```

### 3. 实时监控

```bash
# 监控训练进度
tail -f logs/QuickTests/exp3_debug_reward/training.log

# 查看 tensorboard (如果有)
tensorboard --logdir logs/QuickTests/
```

### 4. 分析结果

```bash
# 训练完成后分析
python drl-manager/analyze_training.py --log_dir logs/QuickTests/exp3_debug_reward

# 对比原实验
python drl-manager/analyze_training.py --log_dir logs/QuickTests/exp3_csv_quick
```

---

## 📝 后续优化建议

### 如果方案 B/D 成功
1. 微调奖励权重
2. 尝试更小的 episode 长度 (250, 200)
3. 调整 VM 初始配置
4. 尝试不同的 workload (更多任务)

### 如果方案 B/D 失败
1. 分析为什么失败 (检查日志和奖励曲线)
2. 尝试方案 A 或 C
3. 考虑 Curriculum Learning (方案 F)
4. 深入分析快速 episodes 的共同特征

### 长期优化
1. 实现自动超参数调优 (Optuna)
2. 添加更详细的日志记录
3. 可视化智能体的决策过程
4. 对比不同算法 (A2C, SAC, etc.)

---

## 🔗 相关文档

- [训练分析工具](analyze_training.py)
- [成功率监控工具](monitor_success_rate.py)
- [原始配置](config.yml - experiment_3)
- [训练结果报告](logs/QuickTests/exp3_csv_quick/analysis/training_report.txt)

---

**最后更新**: 2025-10-24
**状态**: ✅ 方案已就绪，等待执行
