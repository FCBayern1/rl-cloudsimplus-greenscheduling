# GPU 机器任务简报:找一个"预报真正有用"的调度场景

给在 **GPU 机器**(RTX 3060 那台,tailscale)上运行的 Claude Code agent。本机(RTX 5080)跑训练,你和本机**分工不重叠**。读完这份再动手。

---

## ⭐ 当前任务(2026-08-04 更新,先看这段)

**上一轮 dc8 标定 sweep 已回报(dc8calib_results.md,干得很好),两个关键结论:**
1. **dc8_med(1.29 G-MI)命中健康负载锚点**:drain 完成率 1.0,fcast 碳 −38.5%,green_ratio +17pp——但这些是在一个路由 bug 的不利条件下拿到的保守下界。
2. **你抓到的 spread 路由分块 bug 已修复**(commit d4d2854):`_route_batch` 改成平滑加权轮询(SWRR),动作列表现在按比例交错,批次填不满时前缀也不再全是 DC0。诊断精准,修法就是你建议的交错思路。

**现在要做(按顺序):**
```bash
cd ~/rl-cloudsimplus-greenscheduling && git pull    # 拿到路由修复 + sweep 脚本已自带 trace 资源同步
cd drl-manager && nohup ./run_dc8_calib_sweep.sh > /dev/null 2>&1 &   # horizon 已是 1200
tail -f ~/scenario_sweep_dc8calib.summary
```
- 这一轮**同时修好了两件事**:路由交错(per-DC 分布现在可信)+ horizon=1200(预报价值的公平测量)。
- **判读**:修好路由后 (a) drain 完成率曲线——1.29 G-MI 很可能能承载更高负载(med 之前 94% 挤 DC0 都能跑满);(b) fcast vs drain 碳降是否 ≥15% 且完成率代价可接受;(c) per-DC 分布是否终于按绿电比例(DC0 ~34% 而非 94%)。
- **回报**:summary + 一句话:修好路由后哪个负载档命中"健康完成 + 预报省碳 + 分布合理"。若 med 完成率仍满且 light 也上来了,可能需要再加一个更重的档(报回来,本机加)。
- green_ratio/per-DC 那个只读包装(calib_capture.py)继续用,方法对、不进仓库。

下面是完整背景,首次接手或需要细节时再读。

---

## 1. 一句话背景

我们在写一篇论文 **EU-CRD**:碳感知的数据中心调度,用 RL 学"看绿电预报决定把任务路由到哪个机房/是否延迟"。核心卖点是——当预报被污染时,EU-CRD 能把伤害控制住,而普通训练会盲信坏预报、排碳比不用预报还高。

## 2. 你为什么存在(要解决的问题)

论文主考场(代号 C-regime)有个软肋:**在确定性解码(argmax)下,预报本身的价值很小(干净场景只值约 5%)**。于是"EU-CRD 碳吊打对手"这个头条在这个考场上撑不起来——数据是真的,但不够亮。

所以我们需要**另找一个场景,让"有预报 vs 没预报"的差距很大**,也就是预报"载重"(load-bearing)。你的任务就是从候选场景里把这个找出来。**这不是造假**——是找一个预报本该有用的物理条件,然后如实测量。

## 3. 什么叫"预报载重"(判断标准的物理直觉)

预报有用需要三个条件同时成立:
1. **提交不可逆**:任务一旦路由到某机房就得在那跑(有转移成本);
2. **等待有代价**:纯反应式(看到绿电才跑)会吃亏;
3. **当前绿电 ≠ 决策相关的未来绿电**:光看眼前的绿电会被误导,必须靠预报提前量。

C-regime 缺第 2 条。我们造了几个候选场景去补齐这三条,你来测哪个补得最好。

## 4. 你的主任务:跑第二轮"规则测试"

**规则测试 = 不训练**,用三个写死的脚本策略驱动真实 Java 仿真器,几分钟内就能判断一个场景里预报值不值得用。三个策略:
- `drain` 立即派发(不看预报);
- `reactive` 见绿才放(纯反应式);
- `fcast` 用真实未来决定等不等(godeye,预报价值的上界)。

**第一轮(argmax 路由)已经跑过**,发现一个缺陷:oracle 路由 `argmax(当前绿电)` 把 84–89% 的任务全堆到一个机房,把"预报价值"和"路由太烂"混在了一起。**第二轮修好了这个**——新增 `--routing spread`(按各绿电机房的当前绿电**比例分摊**任务),脚本已经写好推上来了。

### 怎么跑

```bash
cd ~/rl-cloudsimplus-greenscheduling && git pull

# 前置:确保 Java 网关 jar 已构建(规则测试要驱动真实仿真器)
cd cloudsimplus-gateway && ./gradlew installDist && cd ..

cd drl-manager
nohup ./run_scenario_sweep_spread.sh > /dev/null 2>&1 &
tail -f ~/scenario_sweep_spread.summary        # 实时看判决行
```

五个场景串行,每个约 15–25 分钟,全程约 1.5 小时。每场景完整日志在 `~/sweep_logs/spread_<名字>.log`,判决行汇总在 `~/scenario_sweep_spread.summary`。

### 五个候选场景

| 名字 | config key | 说明 |
|---|---|---|
| 拥挤轻 1.25x | `experiment_sweep_rwv3l` | 负载 1.25 倍 |
| 拥挤中 1.5x | `experiment_sweep_rwv3m` | 负载 1.5 倍 |
| 缺电版 | `experiment_sweep_scarce` | 绿电减半 |
| 错峰版 | `experiment_sweep_offset` | 各机房绿电曲线时间错开 ← **第一轮的冠军** |
| 8机房版 | `experiment_sweep_dc8` | 8 个机房,规模验证 |

## 5. 怎么读结果(PASS 判据)

对每个场景,看 **fcast vs drain** 这一对(不要看 reactive 当基准——reactive 的低碳是靠只干 29–36% 的活换来的假象):

- **PASS(预报载重)**:`fcast` 相对 `drain` 呈**帕累托改进**——碳明显降(目标 ≥15%),完成率不明显降(掉 ≤2pp),同时超时率下降。
- **最稳的信号是 `green_ratio`(绿电占比)**:五个场景第一轮全都涨,这是预报有用的最稳指标,重点看它 spread 路由后是否还涨、涨多少。
- **冠军 = 碳降幅最大 + 完成率基本不掉 + 超时率显著下降**的那个。第一轮里错峰版做到了碳 −66%、超时 23%→2%,重点确认它在公平路由下是否依然领先。

## 6. 把结果报回来

跑完后,请:
1. 把 `~/scenario_sweep_spread.summary` 完整贴给本机这边(或 push 一份到仓库);
2. 用一句话给出**冠军场景**和它的 fcast-vs-drain 碳降幅/完成率变化;
3. 如果所有场景都不 PASS,如实说,并指出 spread 路由后各场景的 green_ratio 变化——**不要**擅自去设计并训练全新的大场景。

## 7. 纪律(重要,别踩)

- **只做规则测试,不训练**。规则测试不走 learner 路径,在你这台是验证过能跑的。训练(AI 测试)有个未解决的 torch/numpy 环境问题(见下),现在**不碰**,等冠军场景定了、本机 ablation 跑完,再由本机这边统一决定在哪训。
- **对比组不跨机器**:同一个实验/同一组对比只能在一台机器上跑完,不能一半这台一半那台。你负责场景筛选(规则测试),本机负责 ablation(训练),两条线独立。
- **不要碰论文文件**(`paper_materials/`、任何 `.tex`)、不要碰 `references.bib`。
- **不要 push 到 main 之外的分支去改代码逻辑**;你的产出是 summary 数据,不是代码改动。如果发现脚本 bug,先报回来再说。

## 8. 你这台机器的已知坑

- **训练会崩**(如果你尝试的话):RLlib learner 报 `all input arrays must have the same shape`,是 torch/numpy 版本问题。已装 torch 2.11.0 + numpy 2.4.4 到 venv,但**没回验过**。规则测试不受影响(不走 learner)。
- **jar 必须先构建**:`cd cloudsimplus-gateway && ./gradlew installDist`,否则规则测试连不上仿真器。
- **仿真器残留进程**:每个场景开头脚本会 `pkill -9 -f MainMultiDC` 清理,正常现象。
- 退出时可能出现 `Py4JNetworkError` —— **无害噪音**(环境 close 时 SIGTERM 掉 JVM),不影响结果,别当报错处理。

---

**一句话总结**:跑 `run_scenario_sweep_spread.sh`,找出预报载重最强的场景(大概率是错峰版),把 summary 报回来。就这一件事。
