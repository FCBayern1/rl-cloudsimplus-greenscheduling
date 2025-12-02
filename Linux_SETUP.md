# Linux环境配置

## 前置要求检查

```bash
# 检查系统
uname -a
cat /etc/os-release

# 检查是否已安装Java
java -version

# 检查是否已安装Python
python3 --version
```

## 1. 安装 Java 21

```bash
# 更新包管理器
sudo apt update

# 安装 OpenJDK 21
sudo apt install -y openjdk-21-jdk

# 验证安装
java -version
# 应该显示: openjdk version "21.x.x"

# 设置JAVA_HOME（添加到~/.bashrc）
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.zshrc
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
source ~/.zshrc

# 验证JAVA_HOME
echo $JAVA_HOME
```

**如果OpenJDK 21不可用（旧版Ubuntu）：**

```bash
# 添加PPA源
sudo apt install -y software-properties-common
sudo add-apt-repository ppa:openjdk-r/ppa
sudo apt update

# 安装Java 21
sudo apt install -y openjdk-21-jdk

# 或者手动下载安装
# wget https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz
# sudo tar -xzf jdk-21_linux-x64_bin.tar.gz -C /opt/
# sudo ln -s /opt/jdk-21 /opt/java
# echo 'export JAVA_HOME=/opt/java' >> ~/.zshrc
# echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.zshrc
# source ~/.zshrc
```

## 2. 配置 Python 环境

```bash
# 安装Python和pip（如果没有）
sudo apt update
sudo apt install -y python3 python3-pip python3-venv

# 进入项目目录
cd ~/projects/rl-cloudsimplus-greenscheduling/drl-manager

# 创建虚拟环境
python3 -m venv .venv

# 激活虚拟环境
source .venv/bin/activate

# 升级pip
pip install --upgrade pip

# 安装依赖
pip install -r requirements_rllib.txt
```

## 3. 编译 Java Gateway

```bash
cd ~/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway

# 修复gradlew权限和行尾符（如果需要）
sed -i 's/\r$//' gradlew
chmod +x gradlew

# 编译项目
./gradlew build

# 验证编译成功
ls -lh build/libs/
```

## 4. GPU配置（可选，如果有NVIDIA GPU）

```bash
# 检查GPU
lspci | grep -i nvidia

# 检查NVIDIA驱动
nvidia-smi

# 如果没有nvidia-smi，需要安装驱动
# Ubuntu 22.04/24.04:
sudo apt update
sudo apt install -y nvidia-driver-535  # 或更新版本

# 重启系统
sudo reboot

# 重启后验证
nvidia-smi

# 验证PyTorch CUDA支持
cd ~/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```



## 从Windows部署到远程服务器

### 前提条件
- VPN已连接（能访问服务器）
- SSH可以连接到服务器：`ssh -p 2222 joshua@144.173.254.94`

### 部署步骤

#### 步骤1：在Windows PowerShell中上传项目

```powershell
# 进入WSL项目目录
cd "\\wsl.localhost\Ubuntu\home\joshua\projects"

# 先在服务器上创建目录
ssh -p 2222 joshua@144.173.254.94 "mkdir -p ~/projects"

# 上传项目（这需要几分钟，取决于网速）
scp -P 2222 -r rl-cloudsimplus-greenscheduling joshua@144.173.254.94:~/projects/

# 等待上传完成...
```

#### 步骤2：SSH登录到服务器

```bash
ssh -p 2222 joshua@144.173.254.94
```

#### 步骤3：在服务器上配置环境

按照上面的"Linux环境配置"章节执行：
1. 安装Java 21
2. 配置Python环境
3. 编译Java Gateway
4. （可选）配置GPU

#### 步骤4：在服务器上运行训练

使用tmux保持会话（即使SSH断开也能继续运行）：

```bash
# 安装tmux
sudo apt install -y tmux

# 创建新会话
tmux new -s training

# 窗口1：启动Java Gateway
cd ~/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 按 Ctrl+B 然后按 C 创建新窗口

# 窗口2：运行训练
cd ~/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_5 --num-workers 0 --num-gpus 1

# 断开tmux会话：Ctrl+B 然后 D
# 重新连接：tmux attach -t training
```

### 后续更新代码

如果在WSL中修改了代码，重新上传：

```powershell
# 在Windows PowerShell中
cd "\\wsl.localhost\Ubuntu\home\joshua\projects"

# 只上传修改的文件（增量同步）
# 如果安装了Git for Windows with rsync:
rsync -avz --progress `
    --exclude ".venv/" --exclude "build/" --exclude "logs/" `
    -e "ssh -p 2222" `
    rl-cloudsimplus-greenscheduling/ `
    joshua@144.173.254.94:~/projects/rl-cloudsimplus-greenscheduling/

# 然后在服务器上重新编译（如果修改了Java代码）
ssh -p 2222 joshua@144.173.254.94
cd ~/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew build
```

## 如何运行

### 方法1：运行单个实验
```bash
# 终端1：启动Java Gateway（如果还没启动）
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 终端2：运行训练
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
EXPERIMENT_ID="experiment_1" python entrypoint.py
```

### 方法2：运行Multi-DC层次化MARL训练
```bash
# 终端1：Java Gateway（同上）

# 终端2：Multi-DC训练（交替训练）
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
EXPERIMENT_ID="experiment_multi_dc_3" python entrypoint_multidc.py
```

### 方法3：运行PettingZoo并行训练（推荐 - 真正的同时执行）
```bash
# 终端1：启动Java Gateway（Multi-DC版本）
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC

# 终端2：运行PettingZoo + RLlib并行训练
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate

# 方式A：使用默认配置
python entrypoint_pettingzoo.py

# 方式B：使用环境变量配置
EXPERIMENT_ID="experiment_multi_dc_5" NUM_WORKERS=4 TOTAL_TIMESTEPS=100000 python entrypoint_pettingzoo.py

# 方式C：使用命令行参数
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 4 \
    --total-timesteps 100000 \
    --num-gpus 1

# 方式D：仅测试环境（不训练）
python entrypoint_pettingzoo.py --test
```

## 当前状态

### 已验证可工作
- Java Gateway成功编译和启动
- Python环境配置正确
- Multi-DC训练可以启动（已测试到Episode 13）
- PettingZoo并行训练环境可以正常运行

###  需要注意
1. **风力预测功能已禁用**（需要D盘的模型文件）
   - 如果需要启用，请：
     - 将SWF_Prediction文件夹复制到WSL
     - 更新`config.yml`中的路径为WSL路径（如`/home/joshua/SWF_Prediction/...`）
     - 设置`wind_prediction.enabled: true`

2. **设备配置默认为CPU**
   - 如果有GPU且驱动正确安装，可以改回`device: "cuda"`

3. **工作负载文件路径**
   - 所有使用`traces/`开头的路径都是相对于Java resources目录的
   - 确保文件存在于：`cloudsimplus-gateway/src/main/resources/traces/`

## 后台运行Java Gateway

当前Gateway运行在后台，日志文件：
```bash
tail -f /home/joshua/projects/rl-cloudsimplus-greenscheduling/gateway_multidc.log
```

停止Gateway：
```bash
pkill -f "gradlew run"
```

## 推荐的测试实验

### 测试1：快速单DC实验
```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
EXPERIMENT_ID="experiment_3" python entrypoint.py
```

这个实验配置了：
- 较少的训练步数（120000）
- CSV工作负载（比SWF更快）
- 合理的VM配置

### 测试2：PettingZoo环境测试（推荐）
```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
python entrypoint_pettingzoo.py --test
```

这个测试会：
- ✓ 验证PettingZoo环境创建
- ✓ 检查风力预测集成
- ✓ 运行几个步骤验证功能
- ✓ 不进行实际训练（快速验证）

## PettingZoo并行训练详解

### 什么是PettingZoo并行训练？

PettingZoo是一个多智能体强化学习环境标准，与传统的交替训练不同：

| 特性 | 交替训练 | PettingZoo并行训练 |
|------|---------|-------------------|
| **执行方式** | 先训练Local Agent，再训练Global Agent | 所有智能体同时执行 |
| **协同优化** | ❌ 无法协同 | ✅ 真正的协同优化 |
| **收敛速度** | 较慢 | 更快 |
| **框架支持** | Stable-Baselines3 | RLlib、CleanRL等 |

### PettingZoo训练参数说明

```bash
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_3 \  # 实验配置名称
    --num-workers 4 \                     # 并行工作进程数（0=单进程）
    --total-timesteps 100000 \            # 总训练步数
    --num-gpus 0                          # GPU数量（0=CPU，1=使用GPU）
```

### 常用命令组合

```bash
# 1. 快速测试（仅验证环境，不训练）
python entrypoint_pettingzoo.py --test

# 2. GPU训练（推荐 - 使用RTX 5080）
python entrypoint_pettingzoo.py --num-workers 0 --num-gpus 1

# 3. CPU训练（如果GPU不可用）
python entrypoint_pettingzoo.py --num-workers 0 --num-gpus 0

# 3. GPU加速训练
python entrypoint_pettingzoo.py --num-gpus 1

# 4. 长时间训练
python entrypoint_pettingzoo.py --total-timesteps 200000

# 5. 使用配置文件中的设置
EXPERIMENT_ID="experiment_multi_dc_5" python entrypoint_pettingzoo.py
```

### 监控训练进度

PettingZoo训练会生成以下日志：

```bash
# 查看训练日志
tail -f logs/rllib_experiment_multi_dc_3_<timestamp>/progress.csv

# 查看详细输出
tail -f logs/rllib_experiment_multi_dc_3_<timestamp>/result.json
```

### 风力预测集成

PettingZoo环境支持风力预测（如果在`config.yml`中启用）：

```yaml
wind_prediction:
  enabled: true  # 启用风力预测
  horizon: 8     # 预测未来8步
```

观察空间会包含：
- `dc_predicted_green_power_w`: 每个数据中心未来8步的风力功率预测

## GPU加速配置（重要！）

### GPU可用性检查

在WSL中验证GPU是否可用：

```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate

# 检查PyTorch是否识别GPU
python -c "import torch; print(f'CUDA可用: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"
```

**预期输出（你的系统）：**
```
CUDA可用: True
GPU: NVIDIA GeForce RTX 5080
```

### 启用GPU训练的关键修复

**问题：** 默认情况下，代码使用了`local_mode=True`（为Windows兼容性），这会强制使用CPU并忽略所有GPU设置。

**解决方案：** 已修复 `src/training/train_rllib_multidc.py` 中的 `local_mode=False`，现在可以正常使用GPU了。

### 多智能体如何共享GPU？

在PettingZoo + RLlib的架构中，**所有智能体（Global Agent + 5个Local Agents）自动共享同一张GPU**：

```bash
# 推荐配置（RTX 5080 16GB）
python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 \     # 单进程模式（避免多进程开销）
    --num-gpus 1 \        # 使用1张GPU
    --total-timesteps 100000
```

**为什么是 `--num-workers 0`？**
- `num_workers=0`：所有智能体在**主进程**中训练，共享GPU内存
- `num_workers>0`：创建多个**子进程**，每个都尝试占用GPU，可能导致OOM或冲突

### GPU使用监控

#### 1. 实时监控GPU利用率

在训练时，打开另一个终端：

```bash
# 每秒刷新一次
nvidia-smi -l 1

# 或使用更友好的监控工具
watch -n 1 nvidia-smi
```

**查看关键指标：**
- **GPU-Util**: GPU计算利用率（应该在70-100%之间）
- **Memory-Usage**: 显存使用量（16GB中使用了多少）
- **Processes**: 正在使用GPU的进程列表

#### 2. 训练中的GPU使用情况

**正常的GPU使用模式：**
```
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 535.xx.xx    Driver Version: 535.xx.xx    CUDA Version: 12.2     |
|-------------------------------+----------------------+----------------------+
| GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
|===============================+======================+======================|
|   0  NVIDIA GeForce ...  Off  | 00000000:01:00.0  On |                  N/A |
| 30%   65C    P2   180W / 285W |   8500MiB / 16384MiB |     95%      Default |
+-------------------------------+----------------------+----------------------+

Processes:
  GPU   GI   CI   PID   Type   Process name                GPU Memory Usage
    0   N/A  N/A  12345   C    python                         8400MiB
```

#### 3. 优化GPU利用率

如果GPU利用率低于50%，可以尝试：

**在 `config.yml` 中调整（`experiment_multi_dc_5`）：**

```yaml
training:
  num_workers: 0                # 保持0（单进程）
  num_gpus: 1                   # 使用1张GPU
  train_batch_size: 4000        # 增大批次大小（如果显存够用）
  sgd_minibatch_size: 512       # 增大到512或1024（RTX 5080可以处理）
  num_sgd_iter: 10              # SGD迭代次数
```

**调优建议：**
1. **显存够用** → 增大 `sgd_minibatch_size` 到 1024
2. **显存不足** → 减小 `train_batch_size` 到 2000
3. **想更快训练** → 增大 `num_sgd_iter` 到 15-20

### 常见GPU问题

**Q: 为什么之前设置 `--num-gpus 0`？**
**A:** 这是保守的默认配置，因为WSL的GPU支持曾经不稳定。但你的系统已经正确配置了CUDA，应该使用 `--num-gpus 1`。

**Q: 显示 "Policy running on CPU"？**
**A:** 检查是否有 `local_mode=True`。已修复，现在应该显示 "Policy running on cuda:0"。

**Q: 6个智能体会不会抢占GPU？**
**A:** 不会。RLlib会自动管理，所有智能体的策略网络和训练都在同一个GPU上，按顺序执行。

**Q: 如何知道GPU是否真的在被使用？**
**A:** 运行 `nvidia-smi`，如果 `GPU-Util` 和 `Memory-Usage` 都很高（>70%），说明GPU正在工作。

## 常见问题

### PettingZoo训练问题

**Q: RLlib导入错误**
```
ImportError: cannot import name 'PPOConfig' from 'ray.rllib.algorithms.ppo'
```
**解决方案：**
```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
pip install -r requirements_rllib.txt
```

**Q: Java Gateway连接失败**
```
Py4JNetworkError: An error occurred while trying to connect to the Java server
```
**解决方案：**
- 确保Gateway已启动：`ps aux | grep MainMultiDC`
- 检查端口25333是否被占用：`netstat -tulpn | grep 25333`
- 确认config.yml中的`py4j_port: 25333`正确

**Q: 风力预测数据不出现**
```
✗ Wind predictions NOT found in observation
```
**解决方案：**
- 检查`config.yml`中`wind_prediction.enabled: true`
- 确认`turbine_csv_paths`路径正确且文件存在
- 查看Java日志是否有风力数据加载错误

**Q: 训练速度慢**

**优化方案：**
- 增加`--num-workers`（根据CPU核心数）
- 使用GPU：`--num-gpus 1`（需要CUDA支持）
- 减少episode长度：修改`config.yml`中的`max_episode_length`
- 使用更小的工作负载文件

### Java Gateway连接失败
```bash
# 检查Gateway是否运行
ps aux | grep java

# 重启Gateway
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew --stop
nohup ./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC > ../gateway_multidc.log 2>&1 &
```

### Python虚拟环境问题
```bash
# 重新激活
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate

# 验证
which python
# 应该输出: /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager/.venv/bin/python
```

### 查看日志
```bash
# Java日志
tail -f /home/joshua/projects/rl-cloudsimplus-greenscheduling/gateway_multidc.log

# Python训练日志（训练开始后）
tail -f logs/<experiment_type_dir>/<experiment_name>/current_run.log
```

## 项目目录结构

```
/home/joshua/projects/rl-cloudsimplus-greenscheduling/
├── cloudsimplus-gateway/          # Java模拟引擎
│   ├── gradlew                    # (已修复行尾符)
│   ├── build.gradle
│   ├── logs/cloudsimplus/         # Java详细日志
│   │   ├── cspg.current.log       # 当前运行日志
│   │   └── 2025-11-23_XX-XX/      # 按时间归档的日志
│   └── src/main/resources/
│       ├── logback.xml            # 日志配置
│       └── traces/                # 工作负载文件
│           └── windProduction/    # 风力数据
├── drl-manager/                   # Python RL环境
│   ├── .venv/                     # (新建的Linux原生venv)
│   ├── entrypoint.py              # 单DC训练入口
│   ├── entrypoint_multidc.py      # Multi-DC交替训练入口
│   ├── entrypoint_pettingzoo.py   # 🆕 PettingZoo并行训练入口
│   ├── requirements_rllib.txt     # RLlib依赖
│   ├── gym_cloudsimplus/          # Gym环境
│   ├── src/training/              # 训练脚本
│   │   ├── train_rllib_multidc.py # RLlib训练脚本
│   │   └── train_hierarchical_multidc_joint.py
│   └── tests/
│       └── test_pettingzoo_wind_prediction.py  # PettingZoo测试
├── config.yml                     # (已修复Windows路径)
├── logs/                          # Python训练日志
│   ├── Multi_Datacenter/          # Multi-DC实验日志
│   └── rllib_experiment_*/        # RLlib训练日志
├── gateway_multidc.log            # Java Gateway控制台日志
└── WSL_SETUP_COMPLETED.md         # 本文件
```

## 快速命令参考

### 启动Java Gateway
```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
# 前台运行（看到日志）
./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC
# 后台运行
nohup ./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC > ../gateway_multidc.log 2>&1 &
```

### Python训练（激活虚拟环境后）
```bash
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate

# 单DC训练
EXPERIMENT_ID="experiment_3" python entrypoint.py

# Multi-DC交替训练
EXPERIMENT_ID="experiment_multi_dc_3" python entrypoint_multidc.py

# PettingZoo并行训练（推荐）
python entrypoint_pettingzoo.py --experiment experiment_multi_dc_3

# 快速测试
python entrypoint_pettingzoo.py --test
```

### 查看日志
```bash
# Java Gateway日志（实时）
tail -f /home/joshua/projects/rl-cloudsimplus-greenscheduling/gateway_multidc.log
# 或
tail -f /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway/logs/cloudsimplus/cspg.current.log

# Python训练日志
tail -f logs/Multi_Datacenter/hierarchical_3dc/current_run.log
```

### 停止训练
```bash
# 停止Python训练
Ctrl+C  # 前台运行时

# 停止Java Gateway
pkill -f "gradlew run"
# 或
cd /home/joshua/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
./gradlew --stop
```

---

## 服务器快速命令参考

### 一键安装脚本（复制粘贴运行）

```bash
# 完整环境安装（Java + Python + 编译）
cd ~/projects/rl-cloudsimplus-greenscheduling

# 安装Java 21
sudo apt update && \
sudo apt install -y openjdk-21-jdk && \
echo 'export JAVA_HOME=/usr/lib/jvm/java-21-openjdk-amd64' >> ~/.bashrc && \
echo 'export PATH=$JAVA_HOME/bin:$PATH' >> ~/.bashrc && \
source ~/.bashrc && \
java -version

# 配置Python环境
cd ~/projects/rl-cloudsimplus-greenscheduling/drl-manager && \
python3 -m venv .venv && \
source .venv/bin/activate && \
pip install --upgrade pip && \
pip install -r requirements_rllib.txt

# 编译Java Gateway
cd ~/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway && \
sed -i 's/\r$//' gradlew && \
chmod +x gradlew && \
./gradlew build

echo "✓ 环境配置完成！"
```

### tmux常用命令

```bash
# 创建会话
tmux new -s training

# 列出会话
tmux ls

# 重新连接会话
tmux attach -t training

# 在tmux中的快捷键：
# Ctrl+B C      - 创建新窗口
# Ctrl+B N      - 下一个窗口
# Ctrl+B P      - 上一个窗口
# Ctrl+B "      - 水平分屏
# Ctrl+B %      - 垂直分屏
# Ctrl+B 方向键  - 切换窗格
# Ctrl+B D      - 断开会话（程序继续运行）
# Ctrl+B [      - 滚动模式（按Q退出）

# 关闭会话
tmux kill-session -t training
```

### GPU监控

```bash
# 实时监控GPU
watch -n 1 nvidia-smi

# 或者
nvidia-smi -l 1

# 查看GPU进程
nvidia-smi pmon -c 10

# 查看详细GPU信息
nvidia-smi --query-gpu=index,name,driver_version,memory.total,memory.used,memory.free,utilization.gpu --format=csv
```

### 后台运行（不使用tmux）

```bash
# 启动Java Gateway（后台）
cd ~/projects/rl-cloudsimplus-greenscheduling/cloudsimplus-gateway
nohup ./gradlew run -PappMainClass=giu.edu.cspg.MainMultiDC > gateway.log 2>&1 &
echo $! > gateway.pid

# 启动训练（后台）
cd ~/projects/rl-cloudsimplus-greenscheduling/drl-manager
source .venv/bin/activate
nohup python entrypoint_pettingzoo.py \
    --experiment experiment_multi_dc_5 \
    --num-workers 0 --num-gpus 1 \
    > training.log 2>&1 &
echo $! > training.pid

# 查看日志
tail -f gateway.log
tail -f training.log

# 停止进程
kill $(cat gateway.pid)
kill $(cat training.pid)
```

---

**配置完成日期：** 2025-11-23  
**Python版本：** 3.12  
**Java版本：** OpenJDK 21.0.8  
**系统：** WSL2 (Ubuntu) / 远程Linux服务器  
**服务器地址：** joshua@144.173.254.94:2222  
**更新日期：** 2025-11-23（添加PettingZoo支持 + 服务器部署指南）

