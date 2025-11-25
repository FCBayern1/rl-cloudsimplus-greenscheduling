# 工作负载生成与使用指南

## 📦 已生成的合成工作负载

| 文件 | 任务数 | 时长 | 模式 | 实验 | 难度 |
|------|--------|------|------|------|------|
| `synthetic_poisson_200s.csv` | 112 | 200s | Poisson (λ=0.5) | Exp 5 | ⭐⭐ 小 |
| `synthetic_uniform_300s.csv` | 150 | 300s | 均匀分布 | Exp 6 | ⭐⭐⭐ 中 |
| `synthetic_bursty_400s.csv` | 200 | 400s | 突发型 | Exp 8 | ⭐⭐⭐⭐ 难 |
| `synthetic_large_600s.csv` | **605** | 600s | Poisson (λ=1.0) | **Exp 7** | ⭐⭐⭐⭐⭐ **推荐** |

---

## 🎯 推荐的实验流程

### 阶段 1: 快速验证（今天，~30分钟）

```powershell
# Experiment 5: 小规模泊松（112 任务，~15 分钟）
$env:EXPERIMENT_ID="experiment_5"
python .\drl-manager\mnt\entrypoint.py
```

**预期**:
- ✅ 任务会实际执行完成
- ✅ 奖励应该变化（不再是固定的 -95）
- ✅ 可以看到学习曲线

---

### 阶段 2: 中等规模（今天，~45分钟）

```powershell
# Experiment 6: 均匀分布（150 任务，~25 分钟）
$env:EXPERIMENT_ID="experiment_6"
python .\drl-manager\mnt\entrypoint.py
```

---

### 阶段 3: 大规模训练（本周，~2-3小时）⭐

```powershell
# Experiment 7: 大规模泊松（605 任务，~2-3 小时）
$env:EXPERIMENT_ID="experiment_7"
python .\drl-manager\mnt\entrypoint.py
```

**这是最重要的实验！**
- 605 个任务（5倍于之前）
- 100,000 训练步数
- 充分的学习时间
- 真实的性能评估

---

### 阶段 4: 压力测试（可选）

```powershell
# Experiment 8: 突发负载（200 任务，~1 小时）
$env:EXPERIMENT_ID="experiment_8"
python .\drl-manager\mnt\entrypoint.py
```

测试模型在流量突发时的表现。

---

## 📊 工作负载特性对比

### Poisson (泊松分布)

**特点**:
- ✅ 符合真实云环境（最常见）
- ✅ 随机到达，符合统计规律
- ✅ 适合学术研究

**使用场景**: 通用负载均衡研究

**文件**: 
- `synthetic_poisson_200s.csv` (λ=0.5, 小规模)
- `synthetic_large_600s.csv` (λ=1.0, **推荐**)

---

### Uniform (均匀分布)

**特点**:
- ✅ 到达时间均匀
- ✅ 可预测性强
- ✅ 便于调试

**使用场景**: 验证算法正确性、调试

**文件**: `synthetic_uniform_300s.csv`

---

### Bursty (突发型)

**特点**:
- ✅ 模拟流量突发（如促销活动）
- ⚠️ 挑战性高
- ⚠️ 需要动态扩展能力

**使用场景**: 压力测试、弹性伸缩研究

**文件**: `synthetic_bursty_400s.csv`

---

## 🔧 生成自定义工作负载

### 基础用法

```bash
cd data-analysis

# 泊松分布（推荐）
python generate_workload.py \
  --type poisson \
  --arrival-rate 0.8 \
  --duration 500 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/my_workload.csv \
  --seed 42
```

### 参数说明

| 参数 | 说明 | 默认值 | 示例 |
|------|------|--------|------|
| `--type` | 到达模式 | `poisson` | `poisson`, `uniform`, `bursty` |
| `--arrival-rate` | 泊松到达率（任务/秒） | `0.5` | `0.5`, `1.0`, `2.0` |
| `--num-jobs` | 任务总数（uniform/bursty） | `100` | `50`, `200`, `1000` |
| `--duration` | 总时长（秒） | `3600` | `300`, `600`, `1800` |
| `--length-dist` | 任务长度分布 | `uniform` | `uniform`, `normal`, `exponential` |
| `--pes-dist` | CPU核心需求分布 | `weighted` | `weighted`, `uniform` |
| `--output` | 输出文件 | - | `traces/my_workload.csv` |
| `--seed` | 随机种子 | `None` | `42`, `123` |

### 高级示例

#### 高负载泊松

```bash
python generate_workload.py \
  --type poisson \
  --arrival-rate 2.0 \
  --duration 300 \
  --length-dist normal \
  --output ../cloudsimplus-gateway/src/main/resources/traces/high_load.csv
```

#### 轻负载均匀

```bash
python generate_workload.py \
  --type uniform \
  --num-jobs 50 \
  --duration 500 \
  --pes-dist uniform \
  --output ../cloudsimplus-gateway/src/main/resources/traces/light_load.csv
```

#### 多突发

```bash
python generate_workload.py \
  --type bursty \
  --num-jobs 500 \
  --duration 1800 \
  --output ../cloudsimplus-gateway/src/main/resources/traces/multi_burst.csv
```

---

## 📋 CSV 工作负载格式

生成的文件格式：

```csv
cloudlet_id,arrival_time,length,pes_required,file_size,output_size
0,0,164044,1,164,82
1,6,542296,1,542,271
2,9,209556,3,209,104
...
```

### 字段说明

| 字段 | 说明 | 单位 |
|------|------|------|
| `cloudlet_id` | 任务唯一 ID | - |
| `arrival_time` | 到达时间 | 秒 |
| `length` | 计算量 | MI (Million Instructions) |
| `pes_required` | 需要的 CPU 核心 | 1-8 |
| `file_size` | 输入文件大小 | KB |
| `output_size` | 输出文件大小 | KB |

---

## 🚀 立即开始

### 推荐方案（最佳实践）

```powershell
# 1. 快速验证（15分钟）
$env:EXPERIMENT_ID="experiment_5"
python .\drl-manager\mnt\entrypoint.py

# 检查结果是否正常（任务有执行）

# 2. 如果正常，运行大规模实验（2-3小时）
$env:EXPERIMENT_ID="experiment_7"
python .\drl-manager\mnt\entrypoint.py
```

---

## 📊 期望的正确行为

### ✅ 正常情况（CSV 工作负载）

```
Episode 1: Reward = -150.23  (任务执行，有变化)
Episode 2: Reward = -142.56  (学习中，奖励改善)
Episode 3: Reward = -138.91  (继续改善)
...
```

Java 日志：
```
Total cost of executing 29 Cloudlets = $5311.09  ✅
Mean CPU Utilization = 11.46%  ✅
```

---

### ❌ 异常情况（SWF 问题）

```
Episode 1: Reward = -95.00  (固定值)
Episode 2: Reward = -95.00  (完全相同)
Episode 3: Reward = -95.00  (没有变化)
```

Java 日志：
```
Total cost of executing 0 Cloudlets = $0.00  ❌
No cloudlets finished  ❌
Arrived Cloudlets: 0  ❌
```

---

## 🎯 实验对比计划

| 实验 | 工作负载 | 任务数 | 模式 | 用途 |
|------|----------|--------|------|------|
| Exp 1 | CSV (原有) | 165 | - | 基准对比 |
| Exp 5 | 泊松 (小) | 112 | Poisson | ✅ **快速验证** |
| Exp 6 | 均匀 | 150 | Uniform | 调试/对比 |
| Exp 7 | 泊松 (大) | **605** | Poisson | ✅ **主要实验** |
| Exp 8 | 突发 | 200 | Bursty | 压力测试 |

---

## 💡 下一步行动

### 立即执行（最重要）

```powershell
# 重启 Java Gateway
# 终端 1
cd cloudsimplus-gateway
.\gradlew.bat run

# 终端 2: 运行 Experiment 5 验证
cd F:\rl-cloudsim-loadbalancer
.\drl-manager\venv\Scripts\Activate.ps1
$env:EXPERIMENT_ID="experiment_5"
python .\drl-manager\mnt\entrypoint.py
```

**预期结果**: 
- ✅ 看到任务执行和完成
- ✅ 奖励有变化
- ✅ 学习曲线出现

---

## 📞 需要更多工作负载？

### 生成更多变体

```bash
cd data-analysis

# 超大规模（1000+ 任务）
python generate_workload.py --type poisson --arrival-rate 2.0 --duration 600

# 超轻负载（调试用）
python generate_workload.py --type uniform --num-jobs 20 --duration 100

# 自定义突发模式
# (需要修改脚本中的 burst 参数)
```

### 混合模式

未来可以创建更复杂的模式：
- 白天高峰 + 夜间低谷
- 周期性波动
- 随机突发

---

**工作负载已准备好！开始运行 Experiment 5 验证吧！** 🚀

