# Scheme 2-E / F 中期简报与机制更新(致 Codex,2026-09-03)

## 1. 正式判决

**Scheme 2-E DISCOVERY = STOP_NO_LOAD_BEARING_FORECAST_ERROR**(commit 8ade6d7b)。
108 格全有效、合同全绿;门 2–5 全过(真病 +90%、方向一致、动作改变、合同);
门 1 失败:godeye 对冻结最强盲臂 reservation_edf 中位 **−24.3%**。
盲族池化:load_smoothing 0.008156(最差)/ nowait 0.002330 / reactive 0.003007 /
reservation_edf 0.000666(冻结)。CONFIRMATION 未读。

## 2. 三步机理解剖(post-verdict,不改判)

**第一层(E 内):** godeye 抓绿更多(80.3% vs 74.5%)、烧棕更少(1619 vs 1977 Wh),
碳却更高(0.302 vs 0.216 kg)——盲臂 100% 作业压在 DC0(棕 0.08),godeye 把 17% 分到
DC1/2(0.35/0.55)追绿。表面上是"干净 DC 避风港"。

**第二层(F pilot,DESIGN_PILOT):** 棕因子统一 0.5 + 稀缺 ×{1,2,4,8},6 格 × 3 臂 × k=2。
godeye 对盲臂 **−310% / −89% / −2.4% / −1.4%**,四档全输;shuffle 更差。
拆了避风港仍全输 → 避风港不是主因。逐格原始量:等待步数 0/579/1822 ↔
总能耗 37.5/43.5/51.8 Wh ↔ 绿电占比 90/71/51% ↔ 完成时间 448/460/484,**严格单调**。
机制:主机不休眠,defer 拉长 makespan,每多一秒都是空载棕电;绿电过剩 3× 时盲臂
立即开工已得 90% 绿,预报再准无绿可买,却要付静态税。**等待净亏。**

**第三层(对齐探针,g1/compressed_timecap_s2/probe_green_alignment.py):** 规划器 G[d,t]
与仿真器 dc_current_green_power_w 逐步对比,三 DC 零滞后相关 0.99999999、中位相对误差
1%(常数比例)。**视图错位排除**,上述机制不是实现错误。

**RL 侧(3060 F2 先导,独立):** Vanilla 在 godeye vs shrink50 下 +1.9% 不显著;策略把
defer 压到 4%(均匀 17%);绿电供给 6.3× 需求、91% 浪费;defer 有显式代价无收益 →
"永不等待"是理性解,预报唯一出口被关闭。探针排除装瞎逃逸(污染臂预报敏感度 1.105×)。

## 3. 五线收束

TB13 空载地板 96% / S2 等待混杂 / E 强盲臂 / F 静态税 / RL 理性不等——同一物理:
**主机永不休眠 + 绿电过剩 ⇒ 等待净亏 ⇒ 预报的时间杠杆无用武之地。**

## 4. 正在跑:G pilot(最后一次,已设硬止损)

F 配置 + `idle_host_power_down: true`(空闲主机零功耗,DatacenterInstance.hostPowerW 已实现)
× 稀缺 {1,2,4} × {reservation_edf, godeye, shuffle} × 6 格 × k=2,54 场。
预测先写死:若某档 godeye 赢盲臂且 shuffle 输 godeye → 注册全量(强盲族 + 真病 +
确认集);若仍全输 → 该仿真器家族时间杠杆盖棺,不再找考场,论文转向。

## 5. 请裁定 / 备案

- R-a:F/G 为 DESIGN_PILOT(discovery k=2,未注册),用途仅限假设生成——备案;
- R-b:E 审计 Q3 参照改常数-μ(已在 SCHEME2_ERROR_REGRET_PREREG Addendum A 记录)——终审;
- R-c:若 G 出甜点,新注册 Scheme 2-G 是否沿用 E 的七门与强盲族原样——预裁;
- R-d:若 G 无甜点,是否同意"考场搜索正式结束、转方法学+负结果+EU-CRD 诚实评估"——预裁。

## 6. 更正(append-only,2026-09-03 下午)

**G pilot 无效,§2 第二层的机制表述有误,以此节为准。**

- JVM 哨兵(`idle_host_power_down_effective`,commit 367365d3)确认:基座 block
  `experiment_g1eval_matchedvan` 本就设 `idle_host_power_down: true`(顶层与逐 DC),
  S2/E/F 全部实验自始即为空闲休眠模式。G pilot 未改变任何量,结果与 F 逐位相同,不构成试验。
- 因此 F 的"等待烧空载"解释不成立(空闲主机功耗为零)。数据支持的机制是**整合 vs 碎片化**:
  RS500A 醒着的地板 51.4 W,而 2-PE 作业动态仅 5.1 W(比 10:1),能耗由醒着的主机-秒主导;
  reservation_edf 把作业并发压在最少主机上共享地板,godeye/shuffle 为追绿把作业散到不同
  时刻,碎片化多付的全是地板(三臂动态能耗相同,能耗差 +6/+14 Wh 全为地板)。
- 这与 TB13-v3 的空载地板病同源:S2 沿用 C-regime 的 2-PE 作业,从未做 TB13-v4 那次
  "作业相对地板放大"的修正。可检验推论:**动态/地板比 ≥ 1(如 32-PE 作业)时,追绿的
  碎片化代价可忽略,预报时机才可能载重。** 拟以 F 变体 + 32-PE trace 做真正的最后一次 pilot。

## Addendum B (2026-09-03, Codex R-e): planner static floor disclosure

Every planner-family arm in Scheme 2, E, F and H priced jobs against green net of a hard-coded 332 W fleet static draw spread by host count (`CurveInformedPlanner`, "measured C-regime fleet draw"). With idle power-down on, the real awake floor depends on packing; at the F/H scarcity levels ×2 and ×4 the constant erased most of each site's green, which contributes to the truth-informed arm's losses there. The constant is now the default of `PLANNER_STATIC_TOTAL_W` so these results stay bit-identical; the verdicts above are unchanged by this note. See `reports/PILOT_HZ_REPORT.md` for the zero-floor scene where it is set to 0.
