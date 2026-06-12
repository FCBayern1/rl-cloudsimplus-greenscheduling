# 仿真扩展路线 & 公开 Workload 数据集调研

_Status: PLANNED / FUTURE WORK — 等 critic 修复验证 + EU-CRD collapse v2 落地后再动_
_Created: 2026-06-12_

## 0. 原则

- **打包做**:每次仿真改动都会作废 MEASURED baselines、全部 checkpoint 和 Pareto 结果。
  选中的改动一次性进去,baseline 套件只重跑一遍。
- **先修后扩**:当前优先级是 critic 修复(plan Part 6)→ EU-CRD collapse v2,仿真扩展排在其后。
- 按 CLAUDE.md 规矩,新增决策点的改动(如 migration)要同步更新 overhead 评测
  (决策耗时 p95/p99、推理延迟、相对 baseline 开销)。

## 1. 仿真扩展优先级

### Tier 1 — 直接服务研究叙事(先做)

| # | 改动 | 动机 | 成本 |
|---|---|---|---|
| 1 | **可延迟任务 / deadline 任务类** | waste_ratio 钉死 0.937 的机制性原因:风电峰值来时没有需求可挪。给 cloudlet 加 `deadline`,分 latency-sensitive / deferrable 两类 → 调度获得时间维杠杆,**forecast 的价值上限大幅提高**("等风来"才有意义) | 中(Java 队列加延迟提交;reward 接现有 SLA/Lagrangian) |
| 2 | **时变电网碳强度** | 现在 brown_carbon_factor 是常数 → "何时用棕电"无所谓。换真实电网碳强度时序(ENTSO-E / WattTime,每 DC 一条 CSV,复用风电 CSV 管线)→ 时间条件化路由收益空间翻倍,评审吃"真实 trace" | 低 |
| 3 | **随机化工作负载(= Q3 Gate D)** | 按 episode 采样到达 → demand forecast 故事非平凡(否则 forecast ≡ timestep)、消除确定性 trace 软肋、考验 critic 泛化 | 低-中(换负载生成器) |

### Tier 2 — 用户点名的方向(Tier 1 之后)

| # | 改动 | 动机 | 成本 |
|---|---|---|---|
| 4 | **调度/传输延迟** | 全局路由→DC 网络延迟(按地理距离)+ 观测陈旧性(看到 t−δ 状态)→ 预测从加分项变刚需,和 TimeCAP 互相成就 | 中(作废全部 baseline) |
| 5 | **跨 DC 迁移(VM/任务 migration)** | 路由/预测错了可付代价纠正;与 EU-CRD 漂亮互动(迁移成本算 forecast 还是 routing 份额,正是 CRD 能回答的) | 高(网络模型:带宽/迁移时长/停机/传输能耗;动作空间变复杂;per-action reward 重设计) |

### Tier 3 — 备选(评审要求再做)

储能电池(吸收风电峰值,经典但 state 复杂化)、风光混合(solar 日间曲线与负载日间曲线相关)、DVFS/异构功耗模型(local agent 加杠杆)。

## 2. 公开 Workload 数据集(2026-06 调研)

核心需求:**任务级到达 trace**(提交时间+资源量+时长 → cloudlet MI),而非利用率时序。

| 数据集 | 内容 | 对本项目价值 | 链接 |
|---|---|---|---|
| **Alibaba cluster-trace-v2018** ⭐ | 8 天 4000 机,batch_task 表(提交/时长/plan_cpu/实例数),batch+online 共存 | 最适配:真实昼夜到达模式;**batch/online 二分 = 可延迟/延迟敏感两类任务的真实数据依据**(接 Tier 1 #1) | github.com/alibaba/clusterdata (2017-2025 共 8 个 trace) |
| **Azure Public Dataset** | VM trace 2017/2019(260 万 VM 到达/生命周期/规格);**Azure Functions 2019/2021 调用 trace(分钟级,昼夜模式极强)**;LLM 推理 trace 2023/24 | VM trace → "cloudlet=VM 请求";Functions trace 是最干净的到达率昼夜曲线,适合校准到达过程 | github.com/Azure/AzurePublicDataset |
| **Google Borg trace (2011/2019)** | job/task 事件流:提交、资源请求、优先级、抢占 | 引用率最高的 benchmark;2019 在 BigQuery(量大),2011 可直接下载、规模够用 | research/Google "clusterdata" |
| **Parallel Workloads Archive (SWF)** | 38 个真实集群日志,一行一作业(提交时刻/时长/核数) | **工程成本最低**:SWF→cloudlet 直译(时长×核数→MI),CloudSim 经典搭配;缺点 HPC 语义、偏老 | cs.huji.ac.il/labs/parallel/workload/swf.html |
| **Bitbrains GWA-T-12** | 1750 台企业 VM 利用率时序(金融业,季末峰值) | 不适合做到达 trace;适合**训练需求侧预测器**(Q3 demand forecast 数据源) | gwa.ewi.tudelft.nl/datasets/gwa-t-12-bitbrains |
| **Zenodo DataCenter-Traces-Datasets** | Alibaba2018/Google2019/AzureV2 预处理利用率时序,CC-BY 4.0 | workload forecaster 训练数据,已清洗已降采样,最省事 | zenodo.org/records/14564935 |

备查:ACM Computing Surveys《Public Datasets for Cloud Computing》(dl.acm.org/doi/10.1145/3719003,related work 索引);Helios/Philly GPU 训练 trace(若做"绿色 GPU DC"故事)。

## 3. 接入路径(建议)

1. **第一步(成本最低)**:不直接回放 trace——用 Alibaba 2018 / Azure Functions 的到达率曲线
   **拟合时变 Poisson**,每 episode 重采样。同时完成 Gate D + "真实 trace 校准",评审两头堵住。
   只换 Python 侧负载生成器,Java 不动。
2. **第二步**:引入 Alibaba batch/online 二分类作为两类任务依据,配 deadline 字段(Tier 1 #1)。
3. **MI 量级校准**:换负载后重新校准到当前 ~6-8 arrivals/step(或接受全部 baseline 重跑),
   否则 carbon floor、MEASURED baselines 失去可比性。
