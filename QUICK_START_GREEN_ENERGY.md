# 🚀 Green Energy Quick Start Guide

## 快速启动指南 - 3步搞定Green Energy

---

## TL;DR

```bash
# Step 1: 清洗数据（只需运行一次）
cd drl-manager
python preprocess_wind_data_for_java.py

# Step 2: 修改config.yml（已经修改好了）
# green_energy.wind_data_file 已指向清洗后的文件

# Step 3: 启动实验
cd ../cloudsimplus-gateway
./gradlew run  # Terminal 1

cd ../drl-manager  # Terminal 2
.venv\Scripts\activate
set EXPERIMENT_ID=experiment_1
python mnt/entrypoint.py
```

---

## 📋 详细步骤

### Step 1: 数据预处理 (一次性)

**目的**: 清洗原始风电数据，修复所有异常值和重复时间戳

```bash
cd drl-manager
python preprocess_wind_data_for_java.py
```

**预期输出**:
```
================================================================================
 Wind Data Preprocessing for Java GreenEnergyProvider
================================================================================

[1/7] Loading data...
   ✅ Loaded 11,361,190 rows
   Turbines: 134
   Time range: 2020-01-01 00:10:00 to 2022-01-01 00:00:00

[2/7] Removing rows with insufficient columns...
   ✅ Removed 496,998 incomplete rows

[3/7] Cleaning power values...
   ✅ Fixed XXXX negative power values → 0

[4/7] Cleaning temperature outliers...
   ✅ Fixed XXXX Etmp outliers → XX.XX°C
   ✅ Fixed XXXX Itmp outliers → XX.XX°C

[5/7] Processing per turbine (deduplication + interpolation)...
   Processing turbines: 100%|███████████████████| 134/134
   ✅ Removed XXXX duplicate timestamps
   ✅ Interpolated XXXX missing values

[6/7] Merging and saving...
   ✅ Saved to: ../cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_cleaned.csv

[7/7] Verifying data quality...
   Quality Checks:
   ✅ Negative power values: 0
   ✅ NaN values: 0
   ✅ Duplicate timestamps: 0

================================================================================
 ✅ PREPROCESSING COMPLETE - DATA QUALITY VERIFIED
================================================================================
```

**生成的文件**:
```
cloudsimplus-gateway/src/main/resources/windProduction/
├── sdwpf_2001_2112_full.csv      # 原始数据（保留）
└── sdwpf_2001_2112_cleaned.csv   # 清洗后数据 ✅
```

---

### Step 2: 验证配置

**检查config.yml中的green_energy配置**:

```yaml
# experiment_1配置
green_energy:
  enabled: true                                     # ✅ 启用
  turbine_id: 57                                    # ✅ 使用turbine 57
  wind_data_file: "windProduction/sdwpf_2001_2112_cleaned.csv"  # ✅ 使用清洗后数据
  prediction:
    enabled: false                                  # ❌ 暂不启用预测（Phase 2功能）
```

**✅ 配置已经准备好，无需修改！**

---

### Step 3: 重新构建并启动

#### Terminal 1: Java Gateway

```bash
cd cloudsimplus-gateway
./gradlew build  # 重新构建（包含之前的去重修复）
./gradlew run
```

**预期日志（成功标志）**:
```
INFO  GreenEnergyProvider - Initialising GreenEnergyProvider for turbine 57...
INFO  GreenEnergyProvider - Starting loadCsvData for turbine 57 from file: windProduction/sdwpf_2001_2112_cleaned.csv
INFO  GreenEnergyProvider - ✅ Loading CSV from resources: windProduction/sdwpf_2001_2112_cleaned.csv
INFO  GreenEnergyProvider - CSV Header: TurbID,Tmstamp,Wspd,Wdir,Etmp,Itmp,Ndir,Pab1,Pab2,Pab3,Prtv,T2m,Sp,RelH,Wspd_w,Wdir_w,Tp,Patv
INFO  GreenEnergyProvider - First matched line for turbine 57: ...
INFO  GreenEnergyProvider - CSV loading complete:
INFO  GreenEnergyProvider -   Total lines processed: XXXXXX
INFO  GreenEnergyProvider -   Lines skipped (insufficient columns): 0          ✅
INFO  GreenEnergyProvider -   Lines skipped (different turbine): XXXXXX
INFO  GreenEnergyProvider -   ✅ Matched lines for turbine 57: XXXXX
INFO  GreenEnergyProvider - Loaded XXXXX data points for turbine 57
INFO  GreenEnergyProvider - Building spline interpolator with XXXXX unique points...
INFO  GreenEnergyProvider - ✅ Spline interpolation built successfully!        ✅
INFO  GreenEnergyProvider -    Data points: XXXXX (original: XXXXX)
INFO  GreenEnergyProvider -    Time range: [0.0 s, XXXXXX s] (XXXX hours)
INFO  GreenEnergyProvider -    Power range: [0.0 kW, XXXX.XX kW]
INFO  GreenEnergyProvider - GreenEnergyProvider initialized successfully       ✅
```

**🚨 如果还看到错误**:
```
ERROR GreenEnergyProvider - ❌ Failed to load and build spline: points X and Y are not strictly increasing
```

说明：
1. 清洗后的数据没有被使用（检查文件路径）
2. 或者Java代码需要重新编译（./gradlew build --rerun-tasks）

#### Terminal 2: Python Training

```bash
cd drl-manager
.venv\Scripts\activate
set EXPERIMENT_ID=experiment_1
python mnt/entrypoint.py
```

**预期输出（训练日志）**:
```
INFO  SimulationCore - [X.X]: All Cloudlets: XX
INFO  SimulationCore - [X.X]: SUCCESS: XX
INFO  SimulationCore - VMs running: XX
INFO  SimulationCore - Current Power: XXX.XX W
INFO  SimulationCore - Green Power: XXX.XX W                     ✅
INFO  SimulationCore - Green Ratio: 0.XX                         ✅
INFO  SimulationCore - Cumulative Energy: XXX.XX Wh              ✅
```

---

## ✅ 成功指标

### Java侧成功标志：

- ✅ "Spline interpolation built successfully"
- ✅ "GreenEnergyProvider initialized successfully"
- ✅ 无 "NonMonotonicSequenceException" 错误
- ✅ "Lines skipped (insufficient columns): 0"

### Python侧成功标志：

查看训练日志中的info字典：

```python
{
    'current_power_w': 280.0,              # ✅ 当前功率 > 0
    'current_green_power_w': 150.5,        # ✅ 绿色能源 > 0
    'green_ratio': 0.538,                  # ✅ 绿色比例 0-1
    'cumulative_energy_wh': 1.56,          # ✅ 累积能源递增
    'cumulative_green_energy_wh': 0.84,    # ✅ 绿色累积
    'cumulative_brown_energy_wh': 0.72,    # ✅ 褐色累积
    'total_wasted_green_wh': 0.12          # ✅ 浪费的绿色能源
}
```

---

## 🐛 故障排除

### 问题1: "Power spline not initialized, returning 0"

**原因**:
- Spline构建失败（通常是重复时间戳问题）
- 使用的还是原始数据而不是清洗后的数据

**解决**:
```bash
# 1. 确认清洗后的文件存在
ls cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_cleaned.csv

# 2. 检查config.yml中的文件名
grep "wind_data_file" config.yml
# 应该显示: wind_data_file: "windProduction/sdwpf_2001_2112_cleaned.csv"

# 3. 重新构建Java项目
cd cloudsimplus-gateway
./gradlew clean build
```

### 问题2: "current_green_power_w" 总是0

**原因**:
- GreenEnergyProvider未正确初始化
- Turbine ID不匹配

**解决**:
```bash
# 检查Java日志中的turbine ID
grep "turbine_id" cloudsimplus-gateway/logs/cloudsimplus/*/cspg.log

# 应该显示:
# turbine_id=57
```

### 问题3: Python找不到预处理脚本

**原因**:
- 当前目录不对

**解决**:
```bash
# 确保在drl-manager目录
cd /path/to/rl-cloudsimplus-greenscheduling/drl-manager
python preprocess_wind_data_for_java.py
```

---

## 📊 验证清洗效果

### 快速验证脚本

创建 `drl-manager/verify_cleaned_data.py`:

```python
import pandas as pd

csv_path = "../cloudsimplus-gateway/src/main/resources/windProduction/sdwpf_2001_2112_cleaned.csv"

df = pd.read_csv(csv_path, parse_dates=['Tmstamp'])
turbine_57 = df[df['TurbID'] == 57]

print("="*60)
print("Turbine 57 Data Quality Report")
print("="*60)
print(f"\nRows: {len(turbine_57):,}")
print(f"Time range: {turbine_57['Tmstamp'].min()} to {turbine_57['Tmstamp'].max()}")

print("\n✅ Quality Checks:")
print(f"  Negative power: {(turbine_57['Patv'] < 0).sum()} (should be 0)")
print(f"  NaN values: {turbine_57.isna().sum().sum()} (should be 0)")
print(f"  Duplicate timestamps: {turbine_57['Tmstamp'].duplicated().sum()} (should be 0)")

print("\n📊 Power Statistics:")
print(turbine_57['Patv'].describe())

print("\n📊 Temperature Statistics:")
print("Etmp (External Temperature):")
print(turbine_57['Etmp'].describe())
print("\nItmp (Internal Temperature):")
print(turbine_57['Itmp'].describe())
```

运行：
```bash
python verify_cleaned_data.py
```

---

## 🎯 下一步（可选）

### Phase 2: 集成预测功能

详见 `GREEN_ENERGY_ARCHITECTURE.md` 中的"预测功能集成"章节

1. 训练CViTRNN模型（如果还没有）
2. 创建Python预测服务 (`wind_power_predictor.py`)
3. 修改config启用预测
4. 启动预测服务并测试

---

## 📚 相关文档

- **完整架构设计**: `GREEN_ENERGY_ARCHITECTURE.md`
- **Entrypoint使用**: `drl-manager/ENTRYPOINT_USAGE_GUIDE.md`
- **错误处理**: `drl-manager/ERROR_HANDLING_IMPROVEMENTS.md`
- **预测模型**: `SWF_Prediction/README.md`

---

## 💡 Tips

1. **数据清洗只需运行一次**: 清洗后的CSV可以重复使用
2. **保留原始数据**: 清洗脚本不会修改原始文件
3. **先测试基础功能**: 先确保green energy正常工作，再启用预测
4. **查看详细日志**: Java日志在 `cloudsimplus-gateway/logs/cloudsimplus/*/cspg.log`

---

**创建时间**: 2025-10-30
**版本**: 1.0
**状态**: Ready to Use ✅
