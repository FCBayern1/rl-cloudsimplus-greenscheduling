# V3.2B 退火 imitation 规格(草案,交 Codex 复审后实施)

日期:2026-08-17。地位:**学习信号修改,实施前必须过复审**;依据 = 双种子 600k
侵蚀证据(delta +0.119→+0.058/+0.088,基线 0.645→0.80/0.82)+ Codex 08-17 分析
(三项正确性主张已全部独立查证,见 §0)。

## 0. 已查证的三个正确性缺陷(退火之前必须先修的两个 + 一个背景)

| # | 缺陷 | 证据 | 处置 |
|---|---|---|---|
| P0-1 | **扁平 argmax 解码错位**:factorized gate 输出 9 路归一化 log-prob,`explore=False` 对 9 路直接 argmax → defer 生效阈值从 0.5 塌到 ~1/9(路由质量越分散越严重) | `rlmodule_gtrxl_models.py` factorized 分支 + `evaluate_model.py` `explore=False`;**盲臂 s2 评测 21,593 步全 action=8 的现场** | **修评测:分层确定性解码**(先 p_hold>0.5 判 defer/route,再对 8 DC argmax);flat-argmax 与 stochastic 保留为对照格;**全部既有 checkpoint 重评,不重训** |
| P0-2 | **训练 SLA 与判决合同不一致**:`c_ep=max(0,miss−0.005)` 再叠 `tolerance 0.05` → 实际容许 ~5.5% miss;判决合同却是 `completion_rate_mi ≥ 99.5%`。FT 两种子 λ 全程 0.0(约束从未介入),PPO 用完成率换碳完全"合法" | `lagrangian_callback.py` L399;`v32b_ft600_s1/lagrangian.csv` 末行 λ=0 | episode 约束改 `1−completion_rate_mi ≤ 0.005`(deadline_miss 降为第二约束);`c_ep_tolerance` 不再额外放宽;**preflight 增加"训练合同=评测合同"一致性检查** |
| 背景 | per_slot_credit 只是"剔 padding + 槽位 logp 求和 × 同一 advantage",不是逐作业信用;γλ=0.979 直接信号 300 步衰减到 0.0017,等待兑现全靠 critic bootstrap 而 critic 只预测整步回报 | `per_slot_credit_loss.py` L86 | 不在本轮修(Codex 第 5 步保底);退火若失败才进入逐作业 settlement/per-job critic 设计 |

**顺序锁定:P0-1 重评 → P0-2 对齐 → 退火臂。**P0-1 出来之前,现有 FT/盲臂物理数字
一律视为"解码污染待复核",不作正式对照。

## 1. 退火锚设计(temporal-gate-only)

### 1.1 结构

```
L_total = L_PPO + β(i) · L_anchor
L_anchor = mean_{真实槽位} BCE( p_hold^θ(s) , p_hold^BC(s) )
```

- **锚的目标**:冻结的认证 BC 模块(`v32b_cert_ck`)整体前向得到的 `p_hold^BC(s)`,
  在**当前 on-policy 批**的观测上计算——锚住的是 PPO 自己会到达的状态分布
  (BC 蒸馏失败的根因正是离开老师分布后的漂移,离线数据锚覆盖不到这里);
- **为什么冻结整个模块而不只 gate**:gate 输入是 trunk 输出 q,trunk 也在漂移;
  只冻 gate 权重、喂当前 trunk 的 q,得到的不是 BC 策略的 p_hold;
- **梯度限定**:锚损失只更新 **temporal gate 参数**(锚路径上对 q detach)——
  PPO 完整保留 route head 与 trunk 的优化自由(Codex:"PPO 继续优化 route head");
- **掩码**:真实槽位(`batch_cloudlet_mi > 0`),padding 不进锚。

### 1.2 退火日程(预注册,跑前冻结)

```
β(i) = max(β_min, β0 · exp(−i/τ))
β0 = 0.5,β_min = 0.05(非零下限,永不退到 0),τ = 25 iterations(≈200k 步到底)
```

非零下限是 Codex 建议的核心修订:600k 证据显示纯 PPO(β=0 的极限)必然侵蚀,
锚不是脚手架而是常驻项。数值可在复审中调整,跑后不动。

### 1.3 监控与中止(逐 checkpoint)

- `delta`(job_temporal)、`P(defer|not-worth)`、`completion_rate_mi`、carbon 四联板;
- **中止条件**:delta < +0.05(锚失效)或 completion 后段中位 < 99%(纪律崩)——
  中止即停臂,记录,不调参续跑。

### 1.4 对称安全约束(独立旋钮,默认关,一并复审)

老师的 horizon guard/backlog cap 转成 **action mask**:`budget ≤ 0` 的槽位屏蔽
defer 选项(等价于给策略与老师同一张安全网,同时封死"把活推到时域外换碳"的通道)。
若开启,两臂(FT/盲)必须同开。

## 2. 判决协议

- 探针:既有六条件门(`v32b_gates.py`)不变;
- 物理:**分层解码** + 双臂 `completion_rate_mi ≥ 99.5%` 合同 + 3 offset 配对;
- **对照臂的正名**(Codex §六):FT vs 盲臂是**方法级**比较(蒸馏+PPO vs 纯 PPO),
  不是预报单变量消融;**预报因果证据**改由同一 FT 策略的 clean/anti/shuffle
  条件差给出(Gate-5 形制)。两种证据分开表述,不得互相顶替。

## 3. 实施排期(预估)

| 步 | 内容 | 代价 |
|---|---|---|
| 1 | 分层解码器 + 全 checkpoint 重评(FT s1/s2、盲 s1/s2、BC) | 半天机器,零训练 |
| 2 | SLA 对齐(config + callback + preflight 检查 + 测试) | 2–3 小时人力 |
| 3 | 锚 learner 实现 + 单测(锚只动 gate 参数、β 日程、掩码) | 半天人力 |
| 4 | 退火臂 600k × 2 种子(P0 修复后的配方) | ~5 小时机器/种子对 |
| 5 | 判决:六条件门 + 分层解码物理 + clean/anti/shuffle | 半天机器 |


---

## 修订 R1(08-17,Codex 复审五条边界,全部采纳)

1. **P0-1 无条件批准**,实现约束:分层判据 `p_hold>0.5 → defer,else argmax(route[0:8])`,
   tie(=0.5)走 route;必测 p_hold=0.2+均匀路由(flat 会 defer、分层必须 route)、
   0.49/0.51 两侧、padding 隔离、**recurrent state 与原 evaluator 完全一致**
   (不许绕 connector/GTrXL state 取 logits);stochastic 不改(与分层采样数学等价)。
2. **P0-2 收敛为最小版单约束**:`c_ep = max(0, 0.995 − completion_rate_mi)`,
   `c_ep_tolerance = 0`;deadline miss 仅记录不参与 dual(双乘子版本留待以后,
   本轮不引 vector-constrained PPO);λ 由终局 completion 更新,每步稠密代理沿用
   pending-risk;preflight 检查训练 0.995 与评测合同一致;训练监控不替代确定性评测。
3. **教师语义锁定为选项 1:stateless BC teacher**(逐步零状态前向)——与认证探针、
   BC 训练语义一致;**显式声明:锚住的是 stateless temporal mapping,不是完整
   BC rollout policy**。损失精确式:
   `BCEWithLogits(current_gate_logit(current_q.detach()), teacher_p_hold.detach())`;
   frozen teacher `eval()+no_grad()`;掩码 = real_slot_mask × RLlib LOSS_MASK;
   单测:anchor 对 trunk/route head/critic 梯度**严格为零**,PPO 对它们梯度仍在。
4. **β 日程算术更正**:τ=25 时到底需 57.6 iter(≈460k 步),原文 200k 是算错。
   且 β 绝对值无跨 loss 尺度意义——实施前先在冻结 rollout 批上做**梯度标定**:
   报告 `||∇gate L_anchor|| / ||∇gate L_PPO||`,再定 β0/βmin;0.5/0.05 只是占位。
5. **安全网需要新观测通道**:现 action_mask 是 [128] 槽位掩码,不能表达 per-choice;
   需新增 `batch_cloudlet_defer_allowed[128]`,factorized 分支把不允许槽位的 defer
   log-prob 置不可选并重归一化 route 分支;两个条件都要(作业级 budget≤0 +
   全局级 backlog≥cap),公式与老师逐字复用;训练与评测同时生效。
6. 措辞更正:completion 后段中位 <99% 是**灾难性中止线**,不是合同门;正式合同
   = 分层确定性评测 ≥99.5%。
7. **扰动完备性**:clean/anti/shuffle 必须同步变换全部派生作业级特征
   (gain/time-to-best/best-future),只换 DC 级四通道会留干净预测泄漏。

**批准状态**:P0-1 立即实施;P0-2 按最小版实施;锚与安全网待梯度标定与
defer_allowed 通道实现后再放行 600k。


---

## P0-1 重评结果(08-17 10:36–11:5x,分层解码,3 offset,clean argmax)

| 臂 | flat argmax(解码污染) | **HIER argmax(修正)** |
|---|---|---|
| FT s1 | 0.451 @ 97.9% | 0.448 @ 98.6% |
| FT s2 | 0.450 @ 98.3% | 0.455 @ 98.3% |
| **盲臂 s1** | 0.381 @ 88.7% | **0.284 @ 99.25%** |
| **盲臂 s2** | 0.383 @ 88.8% | **0.284 @ 99.25%** |
| (参照)老师 | — | 0.268 @ 100% |
| (参照)no-defer 启发式 | — | 0.343 @ ~99.6% |

**三个判决级发现**:
1. **盲臂被解码假象整体埋没**:修正后完成率 88.7→99.25%、碳 0.38→0.284,双种子
   逐位级一致——盲 PPO 其实是**接近合同的强策略**,此前一切"盲臂卡死在掉活模式"
   的解读全部作废(Codex 的 P0-1 优先级判断完全正确);
2. **FT 几乎不动**(defer 基线高,p_hold>0.5 的槽位分层前后同判)——FT 的物理短板
   是真实的,不是解码假象;
3. **考场预报增量被重新标定**:最强无预报策略不再是 no-defer 启发式(0.343)而是
   盲 PPO(0.284);老师(0.268)对它的优势只剩 **5.5%,落在噪声底(10-13%)之内**。
   盲 PPO 只用当前绿电(forecast_mode=none 不遮 dc_current_green_power_w)就拿到了
   R0 里归给"预报"的大部分收益——与 project_forecast_ablation 的旧结论
   (spatial 主导、时移边际小)在新配方上再次成立。

**含义(待复审)**:V3.2B 的教学链路(可教、可保)结论不变,但**本考场的时移增量
可能只有 ~5%**——低于判决噪声底。继续退火工程之前,必须先回答:
(a) 盲 PPO 0.284 的多 offset/多局稳健性;(b) 老师 vs 盲 PPO 的 iso 配对
(盲臂 99.25% 尚差合同 0.25pp);(c) 若增量确认 <噪声底,V3.2 考场对
"预报载重"的认证价值需要重新评估——这是考场问题,不是学习问题。


---

## H0 判决(08-17 12:54,第三格待出)+ 残差时间门方案(Codex,条件批准待 H1)

| route-only 格 | 碳 | 完成率 | 判读 |
|---|---|---|---|
| 盲 PPO | 0.2844 | 99.25% | 与其正常分层评测逐位同 → **纯空间路由器,零时移** |
| FT | **0.4477** | 98.58% | 比盲臂高碳 **58% >> 10% 判决线** |

**H0 判决(三格全齐 13:11):全模块蒸馏破坏了空间路由,且破坏始于 BC 阶段。**
认证 BC route-only = **0.5048 @ 97.06%**(最烂)→ PPO 微调部分修复到 0.4477 →
仍远差于盲 PPO 0.2844。链条:BC 教坏 → 微调救不回。FT 的 route-only ≈ 其正常评测(0.448≈0.448),
物理短板几乎全部来自被教坏的路由头,与等待行为基本无关——此前"defer 水位过高"的
解读再次修正。机制:BC 全模块把路由头拽向贪心老师(路由质量 ≈0.343 档)且未恢复。

**残差时间门方案(H1 过门后的唯一配方)**:
1. 底座 = 强盲 PPO checkpoint(0.284),trunk/route head/critic **冻结**;
2. BC 只训 temporal gate;gate 输入限必要集(gain/time_to_best/best_now-future/
   slack/wait_age/defer_count/backlog/horizon_remaining);
3. **gain 基准已核实**:env 的 forecast_gain 即"可行集当前最小成本 vs slack 内未来
   最小预测成本"——残差语义现成(备注:理想化成本,不含排队,非盲路由器实现成本);
4. 老师标签 = gain 足够 ∧ slack 足够 ∧ backlog 未超 ∧ 时域允许 → defer,否则 route;
   route 目的地永远交给冻结的盲路由器;
5. PPO 微调两阶段(先 gate-only,过物理门后才许低学习率解冻少量共享层)+
   gate 常驻 imitation/KL 锚 + P0-2 合同 + safety mask;
6. 长延迟信用保底:若 gate-only PPO 仍学不动,改时间动作为半马尔可夫 option
   (到达时一次选:立即/等60s/等300s/等到预测最佳点),route 后逐作业结算——
   不再换 PPO 变体。

**H1 的判决含义不变**:oracle-gate ≥10% 且 ≥8/10 同号 → 按上述配方实施;
~5% 或符号混乱 → 时间门设计无罪,是 8DC 空间替代吃掉了等待价值,换预注册新场景。


---

## H1a 降级为诊断波(08-17 14:30,Codex 在跑复审,五条全采纳)

前两个 offset:ep0 oracle-informed 门 −17.8%(配对有效)/ ep1 +9.6% 且终局完成
99.27% <合同(**配对无效,非负例**)——1 有效胜 + 1 无效,混合。当前 10-offset 波
继续跑完但**只作诊断(H1a),禁止自动终判**。已修五项并落码+测试:
①配对有效性门(双臂终局 ≥99.5% 才计入,否则 invalid);②双时域拆分(teacher 预算按
决策时域 7200,排尽窗 10000 只管完成记账——原实现允许把活推过生产合同,真 bug);
③四快照(carbon/completion × 7200/terminal;H1a 缺 carbon@7200,重算时如实标注);
④更名 **oracle-informed slack-aware heuristic gate**(它成功→存在可利用时间收益;
失败→不可推"完美预测无价值");⑤"FT gate 死了"限定为"在 p_hold>0.5 确定性解码下
行为几乎不触发"(探针 delta 仍正,是输出未跨执行阈值)。
**流程**:H1a 跑完→按有效性门重算→有效配对 <8 或仍混合→跑修正版 H1b
(teacher 预算 7200)。判决门(≥10% ∧ ≥8/10 有效同号)不动。
