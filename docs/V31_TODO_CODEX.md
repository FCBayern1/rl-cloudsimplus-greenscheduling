# V3.1 任务书 — Codex(工单 3 → 工单 2,按此顺序)

**规格的唯一权威来源是 `docs/V31_WORK_ORDERS.md` 当前版本**(已含第四轮复审修订:
生命周期六规则、边界规格、fail-fast、评测一致性)。本文件只定义归属、顺序、约束和
交付口径,**不复制规格**——两处版本漂移是这个项目吃过的亏。

背景(改动动机)读 `docs/V3_FORECAST_DIAGNOSIS.md` §2b 与 §6c;赶时间读 §1 的三行表。

## 你的任务

1. **工单 3:训练侧固定 local drain**(先做,完成后先报告一次再继续)
   - 含 fail-fast(`local_dispatch_mode != dispatch_rate` 即抛错);
   - 含评测一致性:evaluator 读到 checkpoint 为 fixed-drain 时**拒绝** `--local rllib`。
2. **工单 2:观测补齐 + 仪表**
   - 批槽位四个新键(含 `is_deferred` / `defer_count`,不预合成 slack);
   - deferred 生命周期六条规则逐条实现,**不确定的 requeue 路径先枚举再归类**,
     写进交付说明(至少含 `MultiDatacenterSimulationCore` L603 forced-route-失败路径);
   - 边界:time_to_deadline 会负、wait_age 可达 ~7200s,归一化+显式截断+deadline mask。

## 硬性约束

1. **机器上有自动实验队列在跑**(P3 → track0 → drainfix,至明早)。禁止启动任何
   训练/评测/Java 网关;**禁止任何 pkill**(训练 JVM 同样匹配 `MainMultiDC`,误杀会
   冻死训练);禁止编辑 `local_eval_rt/run_v3_*.sh`。`gradle test` 和 python 单测可以跑。
2. 所有行为改动**配置门控、默认关**,必须有"默认值下输出与改动前逐位一致"的回归测试
   ——P3/主表的预注册口径正依赖旧行为。
3. 每项代码配测试(项目 CLAUDE.md 硬性要求);确实无法测的写明原因,不许默默跳过。
4. **在 git worktree/分支里工作**。你独占:`GlobalBroker.java`、
   `GlobalObservationState.java`、`hierarchical_multidc_env.py`、
   `hierarchical_multidc_pettingzoo.py`、`train_rlmodule_gtrxl.py`、`evaluate.py`、
   以及 `MultiDatacenterSimulationCore.java` 的**观测组装段**。
   **不许碰**:Core 的 L590-770 奖励段、`SimulationSettings.java`、`config_C.yml`、
   `preflight_scenario.py`(另一人在改)。合并顺序:对方先合,你 rebase;
   import/字段声明区的冲突由后合并者(你)解决。

## 交付口径

- 分支/worktree 名;
- 测试清单及运行结果(含回归测试的"默认值逐位一致"证明);
- 每个新配置键:键名、默认值、开启后的行为一句话;
- requeue 路径枚举表(哪条打 deferred 标、哪条不打、为什么);
- 无法测试项的原因说明。

完成工单 3 后停下报告一次,确认后再进工单 2。
