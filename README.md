# rl-cloudsimplus-greenscheduling

基于 **CloudSim Plus（Java）+ Py4J + Gymnasium（Python）** 的云工作流/云任务调度与**绿色能源/碳排放**感知实验框架。仓库包含：

- **`cloudsimplus-gateway/`**：CloudSim Plus 仿真引擎 + Py4J Gateway（Java 21，Gradle）。
- **`drl-manager/`**：Gym 环境、训练入口、RLlib(PettingZoo) 多智能体训练、baseline 评估、结果对比（Python）。
- **`data-analysis/`**：结果汇总与绘图（Notebook + CSV/PNG）。



### 1) 环境依赖

- **Java**：`cloudsimplus-gateway/build.gradle` 指定 **Java 21**
- **Python**：建议 3.10+（本仓库代码在 3.12 环境下使用较多）

### 2) 构建并启动 Java Gateway（多 DC）

在一个终端：

```bash
cd cloudsimplus-gateway
./gradlew build -x test
./gradlew run -PappMainClass=exe.edu.cspg.MainMultiDC
```

默认端口是 **25333**（Java 侧支持 `PY4J_PORT`/`CSPG_PY4J_PORT` 或 `--port` 参数；见 `exe.edu.cspg.MainMultiDC`）。

### 3) 安装 Python 依赖（RLlib + PettingZoo）

在另一个终端：

```bash
cd drl-manager
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -e .
pip install -r requirements_rllib.txt
```

### 4) 启动训练

`entrypoint_pettingzoo.py` 默认会去找 `../config.yml`：

```bash
cd drl-manager
source .venv/bin/activate
export EXPERIMENT_ID="experiment_multi_dc_5"
python entrypoint_pettingzoo.py
```

可选覆盖参数：

```bash
python entrypoint_pettingzoo.py \
  --config ../config.yml \
  --experiment experiment_multi_dc_5 \
  --num-workers 0 \
  --total-timesteps 400000 \
  --num-gpus 1
```

> 注意：`entrypoint_pettingzoo.py` 在检测不到 Java Gateway 时会 `input()` 询问是否继续（无人值守跑实验时要确保 Java Gateway 已启动）。

---

## 单数据中心训练（SB3 / MaskablePPO 等）

### 1) 启动 Java Gateway（单 DC）

```bash
cd cloudsimplus-gateway
./gradlew build -x test
./gradlew run -PappMainClass=exe.edu.cspg.Main
```

### 2) 启动 Python 训练入口

从仓库根目录运行最不容易出路径问题（确保能找到根目录的 `config.yml`）：

```bash
python drl-manager/entrypoint.py --config config.yml --exp experiment_1
```

或用环境变量：

```bash
export EXPERIMENT_ID="experiment_1"
python drl-manager/entrypoint.py --config config.yml
```

> 单 DC 入口 `drl-manager/entrypoint.py` 会拒绝 `multi_datacenter_enabled: true` 的实验（会提示改用 `entrypoint_pettingzoo.py`）。

---

## 配置说明（`config.yml`）

`config.yml` 结构是：

- **`common:`**：默认参数（仿真、工作负载、奖励权重、训练超参等）
- **`experiment_*:`**：覆盖 `common` 的实验配置（例如 `experiment_1`、`experiment_multi_dc_5`、`experiment_multi_dc_10` 等）

关键字段（多 DC）：

- **`multi_datacenter_enabled: true`**
- **`py4j_port: 25333`**
- **`datacenters:`**：每个 DC 的主机/VM/风机/碳因子配置
- **`global_routing_batch_size`**：全局路由每步处理的 cloudlet 数
- **`training:`**（RLlib）：`algorithm`、`total_timesteps`、`num_workers`、`num_gpus`、`parameter_sharing` 等

关键字段（单 DC）：

- **`env_id: "LoadBalancingScaling-v0"`**（注册入口在 `drl-manager/gym_cloudsimplus/__init__.py`，实际 entry point 是 `LoadBalancingEnv`）
- **`algorithm:`**（SB3）：`MaskablePPO` / `PPO` / `A2C` 等

---

## 结果与日志输出位置

### 多 DC（RLlib）

默认输出目录由 `drl-manager/entrypoint_pettingzoo.py` 生成（可被 `--output-dir` 覆盖），通常类似：

```
logs/<experiment>_<ALGO>[_ParameterSharing]/<timestamp>/
  training.log
  experiment_config.yml
  multidc_training/...
```

Ray Tune 的 TensorBoard/Checkpoint 通常在 `multidc_training/` 子目录下（路径会带 `PPO_multidc_env_*`）。

### 单 DC（SB3）

由 `drl-manager/entrypoint.py` 和 `src/training/train_single_dc.py` 决定，通常在：

```
logs/<experiment_type_dir>/<experiment_name>/<timestamp>/
  current_run.log
  run.log
  config_used.yml
  seed_used.txt
  monitor.csv
  progress.csv
  best_model.zip (或 best_model)
  final_model.zip (或 final_model)
```

---

## TensorBoard

### 多 DC（RLlib / Ray Tune）

把 TensorBoard 指向 **run 的输出目录**（`entrypoint_pettingzoo.py` 会打印 `Output dir`），例如：

```bash
# 示例：查看某个实验的所有 runs（推荐）
tensorboard --logdir logs/experiment_multi_dc_5_PPO_ParameterSharing
```

或者只看单次运行：

```bash
tensorboard --logdir logs/experiment_multi_dc_5_PPO_ParameterSharing/<timestamp>
```

### 单 DC（SB3）

把 TensorBoard 指向 **experiment 文件夹**（这样可以覆盖多个 timestamp 运行），例如：

```bash
# 示例：experiment_1
tensorboard --logdir logs/CSV_Train/Exp1_CSVSimple_GreenEnergy
```

默认端口是 6006；在远程机器上你可能需要：

```bash
tensorboard --logdir logs --host 0.0.0.0 --port 6006
```

---

## Baseline / 算法对比评估

### 1) Heuristic baseline（多 DC）

```bash
cd drl-manager
source .venv/bin/activate
python -m src.baselines.evaluate --global round_robin --local round_robin --experiment experiment_multi_dc_5 --episodes 1
```

### 2) 评估 RLlib checkpoint（多 DC）

```bash
cd drl-manager
source .venv/bin/activate
python -m src.baselines.evaluate \
  --global rllib --local rllib \
  --experiment experiment_multi_dc_5 \
  --checkpoint /abs/path/to/checkpoint_0000xx \
  --episodes 1 \
  --shared-local
```

### 3) 一键对比（脚本）

```bash
cd drl-manager
source .venv/bin/activate
python scripts/compare_algorithms.py --experiment experiment_multi_dc_5 --episodes 1
```

---

## 工作负载（CSV）生成

两份生成器脚本（功能类似）：

- `data-analysis/generate_workload.py`
- `drl-manager/scripts/generate_workload.py`（额外支持 `--min-length/--max-length`）

示例（生成 CSV trace）：

```bash
python drl-manager/scripts/generate_workload.py \
  --type poisson --arrival-rate 1.0 --duration 1000 \
  --output cloudsimplus-gateway/src/main/resources/traces/my_workload.csv \
  --seed 42
```

---

## 目录速览

- **`cloudsimplus-gateway/`**：Java 仿真与 Py4J gateway（`Main` 单 DC，`MainMultiDC` 多 DC）
- **`drl-manager/gym_cloudsimplus/`**：Gym env 注册与实现（含 PettingZoo 并行 env）
- **`drl-manager/src/training/`**：训练入口（SB3 / RLlib）
- **`drl-manager/src/baselines/`**：heuristic scheduler + RLlib checkpoint 评估
- **`logs/`**：训练与评估输出（时间戳目录）
- **`data-analysis/`**：Notebook/图表/汇总 CSV

---

## 常见问题

- **端口冲突**：默认 Py4J 端口 25333；如需并行训练/评估，请给评估传 `--py4j-port` 或启动第二个 Java gateway 使用不同端口。
- **路径问题**：单 DC 入口建议从仓库根目录运行并显式传 `--config config.yml`，避免找不到配置文件。

