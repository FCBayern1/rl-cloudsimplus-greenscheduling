# TB12 预注册表（2026-08-22 20:50 冻结）

**本文件冻结后，T110+111/2021 的判决跑才允许启动。任何参数不得因判决结果回调。**

## 场景（experiment_tb12_iso）

| 参数 | 值 | 冻结来源 |
|---|---|---|
| 机制 | 5 个大作业/24h,runtime 4h,PES 2,不可分割不可抢占,决策=释放时刻 | 相变图断口 (A) |
| trace | `tb12_n5_rt4h.csv`(MI=576e6 每-PE;deadline=arr+15h+rt+2400) | seed 20260822 |
| margin 2400s | 释放量化(≤600)+派发(≤600)+余量,两点定标 120→1200→2400 | 实测 |
| 风电 | SDWPF 原生 10-min,REAL_TIME,timestep 600,horizon 48h(288 步) | Codex 裁定 |
| `green_power_scale` | **9.05562658195e-5**(T100+101/2020 均值 529.136954 kW,ρ=0.5) | 2020 冻结,held-out 不重标 |
| 隔离 A′ | 8 host(RS500A) = 8 small VM(2 PE),RoundRobin 1:1,M/L=0 | Codex 裁定;preflight 六项全绿 |
| M1 | **off**(latest-start 兜底;过期作为独立消融) | Codex 裁定 |

## 臂与冻结参数

| 臂 | 定义 | 冻结 |
|---|---|---|
| nowait | 到达即放 | — |
| **greenfollow(主比较器)** | 因果协调盲:空余实测绿电≥p 则 EDF 放行,latest-start 兜底 | **T100+101/2021 选择跑(12 分层偏移)总碳最低的盲:1.474 vs dpcont 1.692, hazard 1.705** |
| hazard | 窗龄生存盲,θ\*=0.0 | θ/F 表 2020 冻结 |
| dpcont | 连续功率逐状态 DP | cost/转移表 2020 冻结 |
| clair | 在线到达 clairvoyant(读未来风,不读未来作业) | — |

盲的全部拟合只用 T100+101/**2020**;主比较器只用 T100+101/**2021** 选定;
T110+111 在本文件冻结前从未被读取。

## 判决（T110+111/2021,60 episode）

- 偏移:`calib/tb12_offsets.json` 的 `formal_offsets_T110_111_2021`(seed 20260823,每季 15)
- 五臂共享同一 trace/偏移/风电;判决 = **clair vs greenfollow**;其余臂仅诊断
- **配对有效**:两臂 5/5 完成且无截断;任一臂不完成 → 该对无效并单独报因
- **主门**:60 对总碳比 `(Σclair − Σgf)/Σgf ≤ −5%`
- **强门**:≤ −8%
- 方向:≥ **42/60** 对 clair < greenfollow
- **置信**:按季度分块 bootstrap(10000 次)95% CI 上界 ≤ −5%;点估计过线但 CI 跨线 → "有改善、未认证"
- 附报:配对中位数、held-out 实际 ρ(不重标)、ontime/完成 MI 对比

## 选择集参考数字（非判决）

T100+101/2021,12 分层偏移:clair 1.317,greenfollow 1.474(**−10.65%**),
dpcont 1.692,hazard 1.705,nowait 2.067。
