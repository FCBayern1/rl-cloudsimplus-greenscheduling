# GPU 机器任务简报

## ⭐⭐⭐⭐⭐ 当前任务(2026-08-07 深夜):auditor × anti 最小实验(纯 eval,不训练)

**目的**:论文的 runtime auditor 在摘要/方法里是卖点,但零实验支撑(评审必杀点)。它的设计靶点是 anti(反转)污染,主表却只测了 Blend/Shuffle。补一个 6 格对照,今晚就能出:

**{Vanilla, EU-CRD} × anti 污染 × {auditor 关, gate, repair}**,argmax,`--episodes 10`。

**判据**:gate 能否拉回 anti 下的完成率/碳损伤;repair 能否更进一步。正面 → 摘要说法落地;中性/负面 → 论文按降规格处理(也要如实报)。

### 步骤

0. **接线探针(先做,10 分钟)**:trust_sentinel.py 里有两个类——`TrustSentinel`(σ² 门控)和 `ForecastResidualMonitor`(χ 相关性审计器,257 行起,gate 阈值默认 0.2、repair −0.5、W=600)。都用 `TRUST_GATE_MODE` 环境变量,经 `global_scheduler._sentinel` 进 evaluate.py(808 行注释、954 行取用)。**先跑 1 个短 eval(TRUST_GATE_MODE=log)确认实际实例化的是哪个类、χ 有没有被记录**——如果接进来的是 σ² 版而不是 χ 版,立刻报回,别硬跑。
1. **拿 checkpoint(从 5080 scp,两个目录)**:
   - `drl-manager/logs/creg_van_local_s3/multidc_gtrxl_training/PPO_multidc_env_861eb_00000_0_2026-07-15_20-51-08/checkpoint_000010`
   - `drl-manager/logs/creg_eucrd_local_s3/multidc_gtrxl_training/PPO_multidc_env_014b9_00000_0_2026-07-15_14-42-21/checkpoint_000010`
   实验名从各自 `logs/<dir>/experiment_config.yml` 的 experiment_name 读,别猜。
2. **6 格 eval**(两臂 × {unset, TRUST_GATE_MODE=gate, TRUST_GATE_MODE=repair}),统一 `FORECAST_PERTURB_MODE=anti DECODE_TOPK=0 --episodes 10 --seed 3`,阈值用默认(0.2 / −0.5),别调。
3. **顺带补 2 格 clean 对照**(两臂 × auditor 关,clean)——同 ckpt 的基线锚点。
4. **回报**:8 行表(臂 × 条件 × carbon/completion/green)+ gate/repair 各自触发了多少步(log 里有计数)。

**纪律**:不训练、不改阈值、不改仓库逻辑;eval 你这台验证过能跑(反相那次)。

---

# (旧)找一个"预报真正有用"的调度场景

给在 **GPU 机器**(RTX 3060 那台,tailscale)上运行的 Claude Code agent。本机(RTX 5080)跑训练,你和本机**分工不重叠**。读完这份再动手。

---

## ⭐⭐⭐⭐ 开跑任务(2026-08-07):精修碳-headroom 靶子(纯 numpy,拉下来就跑)

**本地已用解析扫描找到正信号**:`drl-manager/scan_carbon_headroom.py`。均匀铺开相位 **spread3k [0,1000,2000,3000] + 长作业 L=700 + D/green=0.5 → oracle 比 greedy 少烧 38.8% 碳**。之前所有场景失败是三条件漏一(反相相位挤成堆 + 作业太短 + 负载太轻)。现在要你把这个靶子**精修 + 压力测试**,确认它不是简化模型的假象。

**启动(两行):**
```bash
cd ~/rl-cloudsimplus-greenscheduling && git pull
cd drl-manager && .venv/bin/python scan_carbon_headroom.py     # 先复现那张表(spread3k L700 ~39%)
```

**你的任务(改 `scan_carbon_headroom.py`,都是 numpy,不训练、不碰仿真):**

1. **最关键的压力测试——作业长度用分布,不是固定 L。** 现在是"所有作业都 L=700"。真实负载是混合长度。把 `score()` 改成对每个到达抽一个 L ~ 比如 LogNormal 或 mixture(短作业占多数 + 少量长作业),看 **headroom 在真实长度分布下还剩多少**。如果只有"全是长作业"才有 35%、一混短作业就塌 → 这个靶子脆,报回来。**这是决定靶子成不成立的头号问题。**
2. **finer 相位网格**:在 spread2k–spread3k 之间(offset 间距 700–1200)细扫,找 headroom 最高的那组 offsets。
3. **D/green 细扫** 0.3/0.4/0.5/0.6/0.7,找"headroom 高 + greedyC 明显>0(有棕烧)+ 不过载"的最佳负载点。
4. **greedy 会不会被写傻了的对照**:现在 greedy=选当前碳最低 DC。加一个"稍聪明的 reactive"(比如看当前 + 前几步平滑的绿电)当对照,确认 35% 不是靠打败一个傻 baseline。

**回报**:①作业长度混合分布下的 headroom(头号)②最佳 (offsets, D/green, L分布) 组合 + 其碳 headroom ③一句话:这个靶子在真实长度混合下还成不成立。**报碳/棕,不报绿电捕获。**

（下一步本机会据此建 RL config + 长 cloudlet trace 实测 godeye vs noforecast;GPU 若还有余力,可用第 1 节等式建正确 green_ts npy + rule-gate 在真仿真器上验证——但先把上面的解析压力测试做完。）

下面是历史任务,留档。

---

## 历史任务(2026-08-07):量"碳 headroom",不是"绿电捕获"

**背景(为什么之前都白找)**:反相 8DC 的 RL 判死了——episodes=10 下 **godeye(完美预报)0.0227 比 noforecast 0.0200 还高 +13.7%**(预报成了纯优化税)。dc8_light 也 ≈0。根因不是去相关度不够,是**绿电太富余、没烧棕**:你上一轮量的"贪心捕获 0.23"是**绿电缺口**,但绿电够喂轻负载时,少抓绿电也不烧棕 → 碳不动。**去相关度是必要非充分,每次都漏了"稀缺"这一条。**

**这一轮换个指标:直接算"碳/棕电 headroom",纯 numpy,用你验过的 CSV↔仿真绿电等式,不碰 rule-gate/npy。**

对每个候选 `(负载 D, 绿电缩放 divisor, 各绿电DC相位 offsets, 作业时长 L)`:
1. 用等式 `green_i(t)=ΣCSV[WARM+off_i+t]/DIV` 得每个绿电 DC 的绿电时序。
2. 设一个总需求 D(t)(先用常数或按 trace 的到达率)。
3. 两种路由,逐步算烧多少棕:
   - **greedy(=noforecast 上界)**:把 D(t) 派给**当前**最绿 DC → `brown=max(0, D − green_argmax_now)`。
   - **oracle(=godeye 上界)**:派给**作业运行窗 [t,t+L] 内平均最绿**的 DC → `brown=max(0, D − green_there)`。
4. `carbon = Σ brown × 碳强度`(碳强度用 config 里各 DC 的 emission factor)。
5. **carbon headroom = (carbon_greedy − carbon_oracle) / carbon_greedy**。

**目标:找到 headroom 大(>20%)、且 greedy 仍能基本完成(D ≤ 总绿容量的大部分时间,不是靠过载压completion)的 `(D, divisor, offsets, L)`。** 关键是同时满足:
- **稀缺**:D 大到 greedy 会烧棕(carbon_greedy 明显 >0);
- **去相关**:oracle 能路到"未来更绿"的 DC,把棕砍下来(需要相位错开 + 作业够长跨越切换);
- **不过载**:总绿容量 ≥ D 的大部分,否则是 completion 崩,不是碳。

**扫描轴**:divisor(绿电稀缺度)、D(负载)、L(作业时长,让"当前绿电"失效)、offsets(相位)。**报 top 5 组合 + 各自的 carbon headroom + greedy 的棕电占比 + 一句话哪个最可能让 RL 也吃到。**

**判读提醒**:这算的是 **oracle 上界**(解码无关)。本地 5080 会再验 RL 在 argmax 下能不能吃到——但**先得有一个 oracle 碳 headroom 就很大的场景**,否则 RL 更没戏。别报绿电捕获,报**碳/棕**。不训练。

---

## 🛑 撤销通知(2026-08-06):反相 RL 训练已移回本地 5080

**GPU 别再跑反相训练/smoke 了。** 本地 5080 训练更快也已验证能跑,反相 RL 对照(oracle vs noforecast,含 shuffle 污染 eval)整组在本地跑,`run_dc8_antiphase_smoke.sh` / `run_dc8_antiphase_fv.sh` 你这边**不用执行**了(留着无妨,别启动,免得和本地重复/拆对比组)。

**你这台改回纯分析**(不训练):如果还有余力,继续找**下一个候选反相/载重场景**——用你那套"仿真绿电 + 去相关度"的方法(第 1 节等式),扫别的负载档 / 别的绿电相位组合,看有没有比"贪心 0.23"更强、或能量隔离更干净的组合。只报 offset + 去相关指标,不训练。没思路就待命,等本地反相结果出来再定下一步。

下面是历史任务(反相 offset 已采纳,训练已移走),留档参考。

---

## ⭐⭐ 历史任务(2026-08-05 晚):smoke 闸门 + 反相 RL 实验(已移回本地)

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
