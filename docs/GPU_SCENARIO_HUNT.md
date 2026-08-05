# GPU 机器任务简报:找一个"预报真正有用"的调度场景

给在 **GPU 机器**(RTX 3060 那台,tailscale)上运行的 Claude Code agent。本机(RTX 5080)跑训练,你和本机**分工不重叠**。读完这份再动手。

---

## ⭐⭐ 最新任务(2026-08-06 更新):smoke 闸门 + 反相 RL 实验

你上一轮的去相关度分析(`scenario_hunt_decorr.tar.gz`)本机已收到并**独立复核通过**——CSV↔仿真器等式我这边也验到 `corr=1.000000`,能量占比逐位一致。**严格档反相 offset(Nordic 0 / Germany 0 / US_East 1000 / Nordic2 100)采纳。** 本机已建好两个配置 + 脚本并推送:
- `experiment_dc8_antiphase_oracle`(forecast_mode=full godeye)
- `experiment_dc8_antiphase_noforecast`(forecast_mode=none)
- 和 dc8_light 只差这 4 个绿电 offset + 名字,turbine/trace 不变(所以训练 classpath 不用打新 jar)。

现在要**在你这台跑反相的 RL 验证**(Oracle vs NoForecast),但训练环境之前是坏的(RLlib learner `all input arrays must have the same shape`),所以**必须先过 smoke 闸门**:

```bash
cd ~/rl-cloudsimplus-greenscheduling && git pull
cd cloudsimplus-gateway && ./gradlew installDist && cd ..   # 确保网关 jar 在
cd drl-manager
./run_dc8_antiphase_smoke.sh          # 20k步/2worker,几分钟,只看 learner 崩不崩
```

- **打印 `SMOKE OK`**(learner 跑通 + 落了 checkpoint)→ 接着跑完整对照(两臂都在你这台,别拆):
  ```bash
  nohup ./run_dc8_antiphase_fv.sh 1 > ~/dc8ap_s1.out 2>&1 &
  tail -f local_eval_rt/dc8ap_summary.txt
  ```
  每臂 600k 步。判读:iso-completion(都≥99.5%)下,oracle 的 `Avg Carbon/MI` 比 noforecast 低多少——**目标 >12%**(dc8_light 的天花板)。跑完把 `dc8ap_summary.txt` 贴回来。
- **打印 `SMOKE FAIL`** → **别 debug、别硬跑**,把 `logs/dc8ap_smoke/train.log` 末尾贴回来,反相实验退回本机 5080 跑。

纪律不变:两臂同机、不改 config/仓库逻辑、不碰 paper/bib。发现脚本 bug 先报。

---

## 当前任务(2026-08-05 晚更新)

### 先说三个结论(回你上一轮的 npy bug 报告)

1. **npy bug 抓得极漂亮,但 RL 结果是干净的。** RL 的 godeye 路径读的是 **Java 仿真器的真未来**(`global_obs_java.getDcFutureShortMean()` 等,本机 grep 确认 env 里 `.npy`/`np.load`/`green_ts` 引用 **0 次**)。所以 5080 上 dc8_light 的 **−12% 结果不受那个 npy bug 影响,成立。** 那个坏 npy 只喂给规则测试探针。
2. **规则测试的 npy 不用修。** 那探针我们已经不信了(它拿 drain 当基线,系统性高估预报价值)。花 1 小时重建一份更准的 npy = 得到一个更准但我们不用的探针,低价值,**跳过**。
3. **你的去相关度分析方向对,但有个关键修正:你是在 npy 上算的,而 npy ≠ 仿真器绿电(你自己发现只有 DC5 对齐)。RL 用的是仿真器绿电,所以真正该算去相关度的对象是"仿真器录出来的绿电",不是那份坏 npy。roll_900 的排序建在坏 npy 上,可能不直接迁移到 RL。**

### dc8_light 为什么只有 12%(物理原因已找到)

它的绿电 DC 相位几乎对齐(`time_zone_offset_rows` = Nordic 0 / Germany 18 / Nordic2 36 / US_East 54,4 个绿电 DC 挤在 0–54 行内)→ 去相关度低 → "等当前最绿 DC"已接近最优 → 预报没多少可赢。**要更大的预报价值,得把绿电 DC 的相位拉开做成反相。**

关键机制(本机已核实):`time_zone_offset_rows` 是 **Java 侧 GreenEnergyProvider 的参数**,它移的是**仿真器实际用的绿电**(RL godeye 读得到)。所以**反相可以纯靠 config 改这几个 offset 来做,RL 会正确看到**——不需要碰任何 npy。

### 你这轮的任务(纯分析 + 录仿真绿电,不训练)

**任务 A — 在"仿真器录出的绿电"上重算去相关度(不是 npy!)。**
- 跑一个 episode(规则测试路径就行,你这台验证过能跑),逐步录每个 DC 的 `dc_current_green_power_w`(或从仿真器日志/obs 里导出),得到"仿真器真实绿电时序"。
- 对当前 dc8_light 的 4 个绿电 DC,算 `corr(green_dc[t], green_dc[t+H])`,H=300/600/900/1200;并算"当前最绿 DC" vs "H 步后最绿 DC"换手比例。这给出**当前场景**的去相关度基线(应该很低,解释 12%)。

**任务 B — 扫 `time_zone_offset_rows`,找最大化仿真绿电去相关度的反相组合。**
- 对 4 个绿电 DC(Nordic/Germany/Nordic2/US_East)尝试把 offset 拉开到跨越绿电主周期(比如均匀铺开 0 / T/4 / T/2 / 3T/4,T = 你从任务 A 录到的绿电主周期步数),让"当前最绿 DC"频繁换手。
- 对每组候选 offset,**录一次仿真绿电**、算跨 DC 换手比例 + corr,选换手率最高 / corr 最低那组。
- **报给本机:那组具体的 4 个 `time_zone_offset_rows` 数值 + 它的去相关度指标 vs 当前 dc8_light。** 本机据此建反相训练配置,在 5080 上 RL 测 Oracle vs NoForecast。

**回报**:①当前 dc8_light 的仿真绿电去相关度(解释 12%) ②最佳反相 offset 组合(4 个数)+ 它的去相关度 ③一句话:反相相对当前场景去相关度提升多少、值不值得 RL 一试。**不要训练,不改 config/仓库逻辑,只报 offset 数值 + 指标。**

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
