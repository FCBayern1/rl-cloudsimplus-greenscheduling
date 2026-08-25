# P1 奖励分母标定 + P2 真值表(零训练) — 奖励一致性通过;clair 上界判据不通过

日期:2026-08-25。脚本 `tb12_reward_calib.py`(+3 条纯函数测试,含封顶套利
最小复现回归锚);判决 harness `tb12_run.py` 字节未动。
环境 = experiment_tb12_rl_fc(未来修复训练同一物理),gradlew 路径,接线哨兵在环。
数据:`local_eval_rt/audit/tb12_reward_calib.json`。

## 一个先被哨兵抓出的事实

rl block 实际是 `csv_year: 2021` —— **训练分布是 T100+101/2021**,不是 2020。
第一次启动即被接线哨兵拦下(sim 54.19 W vs 2020 离线 0.0),标定已对齐 2021。
此项必须写进新预注册(训练风 = T100+101/2021,green_episode_offset_range 52262)。

## per-step 碳分布(4 轨 × 6 偏移,T100+101/2021)

| p50 | p90 | p99 | max |
|---|---|---|---|
| 0(全绿步) | 2.635e-3 | 7.906e-3 | 9.956e-3 |

- 现行 fixed_max=2e-05 → **max 比值 497.8**(vs 封顶 3.0)——比 P0 用均值
  估的 ~27 倍还严重一个数量级,封顶饱和板上钉钉。
- **建议 fixed_max = 6.637e-3**(= max/1.5):注册范围内 max 比值 1.50,
  对 3.0 封顶留 2 倍余量,封顶零触发。

## 真值表(建议分母,无折扣 ΣĈ)

| 轨 | 物理 kg | ΣĈ | cap 命中 | ontime |
|---|---|---|---|---|
| greenfollow | 0.5193 | 78.25 | 0 | 1.0 |
| clair | 0.5637 | 84.94 | 0 | 1.0 |
| nowait | 1.1195 | 168.67 | 0 | 1.0 |
| always_defer | 1.4215 | 214.18 | 0 | **0.0** |

- **同序 PASS**:奖励排序 == 物理排序(无封顶时严格等比,恒等式保证)。
- **cap rate = 0**:四轨全部零命中。
- **always_defer 奖励垫底**:RL 坍缩解在新奖励下是最差解(旧奖励下它套利)。
  其确定性指纹(240 步/ontime 0.0/finished 5)与 rl_nofc eval 逐位吻合,
  "坍缩解替身"验证成立。
- **ontime 分离**:always_defer ontime=0.0 → sla_mode 改 on-time MI 后它
  另挨 SLA 惩罚,双保险。

## 判据结果的机械表述(Codex 更正 2026-08-25)

**奖励一致性真值表通过;预测价值/clair 上界判据在该校准格不通过**
(clair 0.5637 > greenfollow 0.5193,由单偏移 off=4000 驱动:0.195 vs 0.088,
其余 4 胜 1 平)。此失败项不阻塞奖励修复,但影响 capture 的解释(见下)。

**原因更正**:本文初版把 off=4000 的反例解释为"2020 拟合 DP 跨年损耗"——
**错误**。`tb12_run.py` 的 clair 臂调用的是 `coordinated_clair_contig`
(在线读取当前 episode 完整未来风的启发式;2020 冻结 DP 表对应的是 dpcont 臂)。
正确表述:**当前命名为 clair 的在线未来启发式并非全局最优调度器,其连续
插入/协调规则在该窗口被 greenfollow 打败**。clair 不是理论上界。

**capture 判据降格改名**(Codex 裁定):原"capture rate"更名
**frozen-reference gap closure** = (盲−RL_fc)/(盲−clair_ref)。锁定:
仅在池化 盲>clair_ref 时有定义;分母≤0 报 undefined 不反转解释;
**RL_fc vs RL_nofc 才是 RL 使用预测的直接主判据**;gap closure 是相对
冻结参考策略的次级强度指标,不得称"理论上限捕获率"。

## 复现口径(Codex 补全要求)

- 完整命令:`EVAL_CONFIG_PATH=$R/config_C.yml TB12_REQUIRE_FROZEN_JAR=0
  .venv/bin/python tb12_reward_calib.py --year 2021
  --json-out ../local_eval_rt/audit/tb12_reward_calib.json`(GATEWAY_LIBS unset)
- 脚本 --year 默认已改 2021,新增 `assert_year_consistency`
  (experiment csv_year == CLI year,启动前锁死,不依赖运行期哨兵)+ 单测。
- artifact sha256 见 git 提交与下方哈希记录。

## 后续(见 PREREG_RL_REPAIR_DRAFT.md)

v2 配置块已 append-only 生成;ontime_mi SLA、720s backstop、cap 监控已实现
并测试;**backstop 又揪出一个 2x runtime 低估 bug(per-PE 语义),已修**——
详见预注册草案的实现清单。真值表在 v2 块上重跑,结果另记。
