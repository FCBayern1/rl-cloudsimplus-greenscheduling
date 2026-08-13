# V3.1 任务书 — Claude(工单 1 → 工单 4 → 工单 5)

规格权威来源:`docs/V31_WORK_ORDERS.md` 当前版本。本文件只定归属/顺序/交付。
接手的会话先读 `docs/V3_FORECAST_DIAGNOSIS.md` §1 三行表 + §6c/§6d/§6e。

## 任务序列

1. **工单 1:奖励手术(Java)**,内部顺序:
   a. `per_action_completion_mode: bonus|no_offset`(正名"移除 route offset",直接可写);
   b. `defer_cost_mode: flat|incremental_urgency`(**修订规格**:U(s)=clip(1−s/W,0,1)²,
      每次重现——route 或 defer——先结清 U(now)−lastCharged,防逃单;不声称 Ng 不变性);
   c. `per_action_carbon_norm: fixed|scale_only|centered_zscore` + raw/normalized/clip-rate
      仪表(centered 的 +wμ/σ 是显式的碳阈值设计决定,注释里写明);
   d. **四行奖励真值表 JUnit**(高绿电 route>defer;棕电+绿将至+slack 足 defer>route;
      deadline 紧 route>defer;拥塞 DC 让位)——不过真值表的开关组合禁止进训练。
      ⚠️ 已知结构事实:`scale_only + no_offset` 下第①行**必然不过**(绿路由小负 vs
      defer≈0),route/defer 之间必须有正向差——centered_zscore 大概率是必选不是可选。
   e. 回归锁:全部新键默认值下输出与改动前逐位一致。
2. **工单 4**:配置模板(**开关全关**,认证配置等判决后生成)、preflight 四道新门
   (两臂 window_carbon_source 相同 / diff 白名单={forecast_mode} / obs bound ≥ trace p99 /
   认证臂 fixed drain)、`calibrate_reward_norm.py`(artifact 八项元数据,采样口径=
   所有候选 (task,DC) 对,defer 不混入)、`docs/V31_PREREG.md` 骨架。
3. **工单 5(填缝)**:探针扫 v3_oracle_s1/s2 的 ck1…ck10,画反号习得时点表。

## 我的文件所有权

`MultiDatacenterSimulationCore.java` **L590-770 奖励段**、`SimulationSettings.java`、
`config_C.yml`、`preflight_scenario.py`、新脚本/新文档。**不碰** evaluate.py /
GlobalBroker / GlobalObservationState / env.py / pettingzoo / train 脚本(Codex 在改)。
合并顺序:我先合,Codex rebase。

## 明早的开关决定(判决 → 开关)

| 判决 | 决定 |
|---|---|
| track0 上限 ≥15% / <10% | v3 考场续用 / 换场景参数(代码照用) |
| drainfix | 只复核旧判决;**drain 固定无条件进 V3.1,不由它决定** |
| sp_s2 配对 + P3 | 只判跨种子复现;**不决定 z-score 变体**(定位不了方差 vs 干扰项) |
| 工单 1 真值表 | 机械判定哪些开关组合可进训练 |

约束同 `V31_TODO_CODEX.md` 的硬性约束节(机器队列/门控默认关/测试/所有权)。
