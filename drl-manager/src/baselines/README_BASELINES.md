## Baseline & RL 推理实验使用说明

本目录提供了多种 **基准调度算法（heuristics）**，以及使用 **RLlib 训练好的多智能体模型** 进行推理评估的工具。

- 全局调度（选择目标数据中心）：`global_schedulers.py`
- 本地调度（在数据中心内选择 VM）：`local_schedulers.py`
- 统一评估脚本（基线对比 + RLlib 推理）：`evaluate.py`
- 单独加载 RLlib 模型进行推理测试：`load_rllib_model.py`

在运行任何实验前，请确保：

- 已经按项目根目录的 `Linux_SETUP.md` / `README.md` 启动 Java CloudSimPlus 网关；
- 已安装 `drl-manager` 依赖（推荐在虚拟环境中）：

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
pip install -e .
```

---

## 1. 运行单一组合的基线推理实验

脚本：`src/baselines/evaluate.py`  
环境：`gym_cloudsimplus.envs.hierarchical_multidc_env.HierarchicalMultiDCEnv`

示例：运行 **Random Global + Random Local** 在 `experiment_multi_dc_10` 上评估 1 个 episode：

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling

# 1. 启动 Java 网关（单独终端，Multi-DC 网关）
cd cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 2. 在 drl-manager 中运行评估（另一个终端）
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager
python -m src.baselines.evaluate \
  --global random \
  --local random \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
```

常用调度器名称（与 `GLOBAL_SCHEDULERS` / `LOCAL_SCHEDULERS` 中的 key 对应）：

- Global：`random`, `round_robin`, `min_queue`, `green_aware`, `green_queue_balanced`
- Local：`random`, `round_robin`, `first_fit`, `best_fit`, `worst_fit`, `min_load`

你可以自由组合，例如：

```bash
python -m src.baselines.evaluate \
  --global green_queue_balanced \
  --local min_load \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
```

脚本会：

- 使用 `config.yml` 中对应 experiment 的配置创建环境；
- 在同一随机种子下运行指定 episodes；
- 输出每 episode 的能耗 / 完成率等指标到 CSV，并在控制台打印汇总。

---

## 2. 对比多种 heuristic 组合（批量基线对比）

如果想一次性比较多种基准组合，可以使用 `--compare` 选项。  
当前脚本内置了以下组合（在 `compare_baselines` 中定义）：

- `random + random`
- `round_robin + round_robin`
- `min_queue + first_fit`
- `green_aware + best_fit`
- `green_queue_balanced + min_load`

运行示例（每个组合跑 3 个 episode）：

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.evaluate \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
  --compare
```

行为说明：

- 对每个组合依次调用 `run_evaluation`，使用同一 `config.yml` / seed / episodes；
- 为每个组合生成一个单独的 CSV（目录形如：`results/baselines/<timestamp>/<global>_<local>.csv`）；
- 最后在控制台打印一个 **COMPARISON TABLE**，展示平均完成率、绿色比例和碳排放。

如需自定义组合，可以直接修改 `evaluate.py` 最下方 `combinations` 列表。

---

## 3. 使用 RLlib 训练好的模型进行推理评估（Global + Local 都是 RL）

脚本：`run_rllib_evaluation`（在 `evaluate.py` 中）  
模型：由 `src/training/train_rllib_multidc.py` 训练得到的 RLlib checkpoint

### 3.1 从现有实验目录自动找到最新 checkpoint

假设你已经在 `logs/experiment_multi_dc_10/20251201_204434` 下完成过 RLlib 训练，  
目录中包含 `multidc_training/PPO_*/checkpoint_*`：

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.evaluate \
  --global rllib \
  --local rllib \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
  --checkpoint ../logs/experiment_multi_dc_10/20251201_204434/multidc_training/PPO_multidc_env_*/checkpoint_000100 \
```

注意：

- 当 `--global` 或 `--local` 中任意一个为 `rllib` 时，脚本会调用 `run_rllib_evaluation`，即 **Global + Local 都使用 RLlib 模型**；
- `--checkpoint` 必须是 RLlib 的 checkpoint 目录路径（包含 `params.pkl` / `algorithm_state.pkl` 等）。

### 3.2 仅想快速测试 RLlib 推理（不跑完整环境）

可以使用 `load_rllib_model.py` 提供的简单包装器进行 mock 推理：

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.load_rllib_model \
  --experiment /home/joshua/rl-cloudsimplus-greenscheduling/logs/experiment_multi_dc_10/20251201_204434 \
  --test
```

该脚本会：

- 在给定实验目录下自动搜索最新的 PPO checkpoint；
- 加载多智能体 RLlib 算法；
- 构造一组模拟的全局观测 `mock_global_obs`，调用 `RLlibInferenceWrapper.get_global_action(...)`；
- 打印出模型输出的 batch 路由动作。

---

### 3.3 将 3 条 PPO 策略作为调度器接入基线对比（与 `rllib_rllib` 风格一致）

下面是 3 条 PPO 算法如何以“Global + Local 都用 RLlib 策略”的方式，接入 Java 环境做对比实验的示例命令。

> 注意：请将示例中的时间戳和 checkpoint 路径替换为你自己实际训练得到的目录。

- **(1) PPO_baseline（有 God's Eye，使用 `experiment_multi_dc_10`）**

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling

# 启动 Java Multi-DC 网关（一个终端）
cd cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 另一个终端，在 drl-manager 中运行 RLlib 评估
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.evaluate \
  --global rllib \
  --local rllib \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
  --checkpoint ../logs/experiment_multi_dc_10/20251203_105113/multidc_training/PPO_multidc_env_*/checkpoint_000006 \
  --output /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/compare_result/ppo_baseline_rllib_$(date +%Y%m%d_%H%M%S).csv
```

- **(2) PPO_ParameterSharing（局部参数共享，仍然是 `experiment_multi_dc_10`，但训练配置里开启了 parameter_sharing）**

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.evaluate \
  --global rllib \
  --local rllib \
  --experiment experiment_multi_dc_10 \
  --episodes 3 \
  --seed 42 \
  --checkpoint ../logs/experiment_multi_dc_10_PPO_ParameterSharing/20251212_140553/multidc_training/PPO_multidc_env_*/checkpoint_000062 \
  --output /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/compare_result/ppo_param_sharing_rllib_$(date +%Y%m%d_%H%M%S).csv
```

> 说明：虽然训练时启用了“本地 agent 共享策略”，但对 `evaluate.py` 而言，这些细节已经封装在 RLlib checkpoint 里，
> 仍然通过 `--global rllib --local rllib` 这条路径统一接入环境。

- **(3) PPO_simple_no_god_eye（使用简化环境 `experiment_multi_dc_simple`，无 God's Eye 特征）**

`experiment_multi_dc_simple` 在 `config.yml` 中使用 `env_id: "HierarchicalMultiDCSimple-v0"`，评估时脚本会自动使用简化版 Java 环境 `HierarchicalMultiDCEnvSimple`（不包含未来预测特征）。

```bash
cd /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager

python -m src.baselines.evaluate \
  --global rllib \
  --local rllib \
  --experiment experiment_multi_dc_simple \
  --episodes 3 \
  --seed 42 \
  --checkpoint ../logs/experiment_multi_dc_simple/20251206_223544/multidc_training/PPO_multidc_env_*/checkpoint_000019 \
  --output /home/joshua/rl-cloudsimplus-greenscheduling/drl-manager/compare_result/ppo_simple_no_god_eye_rllib_$(date +%Y%m%d_%H%M%S).csv
```

> 小结：通过以上命令，你可以将 3 条 PPO 策略（baseline / parameter sharing / simple 无 God-eye）都以
> “Global + Local = RLlib scheduler” 的方式接入 Java 仿真环境，得到与 `rllib_rllib` 同结构的 CSV 结果文件，
> 方便和 heuristics 以及其他 RL 策略一起做统一对比。

---

## 4. 结果分析与可视化

无论是 heuristic baseline 还是 RLlib 推理评估，`evaluate.py` 都会输出 CSV 文件（每一行是一个 episode 的聚合指标），包括：

- 总绿色能耗、棕色能耗、总能耗、绿色比例、碳排放；
- 总 cloudlet 数、完成率、各 DC 完成数量与平均完成时间等。

你可以使用项目根目录下 `data-analysis/analysis.ipynb` 或自行编写脚本，对不同组合 / RL 模型的结果 CSV 进行对比和可视化，从而评估策略效果。

---

## 5. 各类基线调度算法说明

本项目的调度是 **分层的多数据中心调度**：  
- **全局调度（Global）**：决定每个新到的 cloudlet 发往哪个数据中心（DC）；  
- **本地调度（Local）**：在选定的数据中心内，决定 cloudlet 分配到哪个 VM。

下面对 `GLOBAL_SCHEDULERS` 与 `LOCAL_SCHEDULERS` 中常用的基线算法做简要说明，方便阅读实验结果和设计新策略。

### 5.1 全局调度（Global Schedulers）

对应 `global_schedulers.py` 中的类：

- **`random` – RandomGlobalScheduler**  
  - 思路：在所有数据中心中 **完全随机** 选择目标 DC。  
  - 作用：最简单的参考下界，用于对比“有脑子”的策略能提升多少。

- **`round_robin` – RoundRobinGlobalScheduler**  
  - 思路：按顺序轮流给 DC 分配任务：DC0 → DC1 → … → DC(N−1) → 再回到 DC0。  
  - 特点：追求“表面上的均匀”，但 **不考虑当前队列长度或绿电状态**。

- **`min_queue` – MinQueueGlobalScheduler**  
  - 思路：每次都把新 cloudlet 发给 **当前等待队列最短** 的 DC。  
  - 实现：使用观测中的 `dc_queue_sizes`，选择最小值 `argmin`，并在同一 batch 内本地模拟队列 +1，让分配更均衡。  
  - 特点：偏向 **负载均衡 / 减少等待时间**，但 **不看绿电比例**。

- **`green_aware` – GreenAwareGlobalScheduler**  
  - 思路：始终选择 **绿电比例最高** 的数据中心。  
  - 实现：使用 `dc_green_ratio`，选取最大值 `argmax`，将当前 batch 中的所有 cloudlet 都发过去。  
  - 特点：最大化 **绿色能源利用率**，但可能导致某个 DC **队列过长 / 负载过重**。

- **`green_queue_balanced` – GreenQueueBalancedGlobalScheduler**  
  - 思路：同时考虑 **绿电比例** 和 **队列长度**，做一个加权综合评分。  
  - 实现：对 `dc_green_ratio` 和 `dc_queue_sizes` 做归一化，构造  
    \[
      \text{score} = w \cdot \text{green\_norm} + (1-w) \cdot \text{queue\_norm}
    \]
    默认 `w=0.6`，更偏向绿电，同时惩罚过长队列，并在同一 batch 内动态更新“模拟队列”避免单点拥塞。  
  - 特点：在 **绿色优先** 和 **负载均衡** 之间折中，是一个“较聪明”的 heuristic baseline。

- **`rllib` – RLlibGlobalScheduler**  
  - 思路：不是启发式规则，而是调用 **事先用 RLlib 训练好的全局策略**，从观测中直接输出路由动作。  
  - 用途：在基线对比里，作为“**学习型策略**”与上述 heuristic 进行对照。

### 5.2 本地调度（Local Schedulers）

对应 `local_schedulers.py` 中的类：

- **`random` – RandomLocalScheduler**  
  - 思路：在当前 **可用的 VM**（`action_mask` 为 True）中 **随机** 选择一个。  
  - 特点：不看负载、不看资源，是本地调度的最简单 baseline。

- **`round_robin` – RoundRobinLocalScheduler**  
  - 思路：在 VM 之间做轮询分配：VM1 → VM2 → … → VMk → 再回到 VM1；遇到 `action_mask` 不允许的 VM 会跳过。  
  - 特点：尝试在 VM 间平均摊任务，但 **不区分轻载 / 重载 VM**。

- **`first_fit` – FirstFitLocalScheduler**  
  - 思路：从 VM1 开始扫描，找到第一个 `action_mask` 允许的 VM 就直接用它。  
  - 特点：实现极简，但容易让“靠前的 VM”更忙，**负载可能不均衡**。

- **`best_fit` – BestFitLocalScheduler**  
  - 思路：类似装箱问题中的 **Best Fit**：  
    - 看每个 VM 的 `vm_available_pes`（可用核数）和 cloudlet 的 `next_cloudlet_pes`；  
    - 在所有可用 VM 中，选择“刚好够用、但浪费最少”的那个（剩余核数 ≥0 且尽量小）。  
  - 特点：尝试 **最小化资源浪费**，让每个 VM 被填得更“紧凑”。

- **`worst_fit` – WorstFitLocalScheduler**  
  - 思路：与 Best Fit 相反，优先把任务放到 **剩余资源最多** 的 VM 上（仍需能容纳任务）。  
  - 特点：追求 **负载尽量分散**，防止出现单个 VM 压力过大，但可能产生较多“零碎”资源。

- **`min_load` – MinLoadLocalScheduler**  
  - 思路：根据 `vm_loads`（每个 VM 当前负载）选择 **负载最小** 的 VM。  
  - 特点：直接贪心式地实现 **局部负载均衡**，是很常见的一类调度启发式。

- **`rllib` – RLlibLocalScheduler**  
  - 思路：调用 RLlib 训练出的 **本地策略**，为每个 DC 维护各自的 `local_policy_{dc_id}`，利用观测 + `action_mask` 直接输出 VM 选择。  
  - 特点：作为学习型本地调度，与 `first_fit` / `best_fit` / `min_load` 等 heuristic 做对比。

> 小提示：设计新 heuristic 或调参时，可以优先把结果与  
> `random + random`、`min_queue + first_fit`、`green_queue_balanced + min_load`、以及 `rllib + rllib` 进行对比，  
> 快速判断你的新策略在完成率、绿色比例和碳排放上的优势。

