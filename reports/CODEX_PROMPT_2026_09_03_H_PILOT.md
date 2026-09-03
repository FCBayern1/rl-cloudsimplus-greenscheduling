# 致 Codex:Scheme 2-H pilot 结果与下一步裁定请求(2026-09-03)

## 0. 目标不变

论文主张三步:(1) 找到一个预测误差让调度变差的场景;(2) vanilla RL 在该场景同样受害;(3) EU-CRD 抵御该负影响。所有实验按预注册纪律执行:pilot 结果不进主表,确认集一次性不预览,冻结文件不改。

## 1. 上次工单以来的判决链(时间序)

| 步骤 | 结果 | 依据文件 |
|---|---|---|
| Scheme 2-E discovery(T123/10/51/53/112,k=2/10/18) | STOP,门 1 失败:godeye 输给盲臂 reservation_edf(清洁 DC 避风港) | `reports/S2_E_F_INTERIM_FOR_CODEX.md` §1–2 |
| 对齐探针(规划器 G vs 仿真器绿电) | corr 0.99999999,相对误差 1%,排除"规划器看错绿电" | 同上 §3,`probe_green_alignment.py` |
| F pilot(棕电碳因子统一 0.5 + 容量收紧 divisor ×1/2/4/8) | 三档全输,godeye 未在任一档打败盲臂 | 同上 §4 |
| G pilot(空闲主机休眠) | 作废:JVM 哨兵 `idle_host_power_down_effective` 证实基座本就开着休眠,G 与 F 逐位相同 | 同上 §5,commit 367365d3 |
| 机制修正 | "等待烧空载"解释撤回;数据支持的机制是整合 vs 碎片化:地板 51.4 W,2-PE 作业动态 5.1 W(10:1),盲臂把作业压在最少主机上共享地板,追绿把作业散开多付地板 | 同上 §6,commit 28484116 |
| H pilot(32-PE 作业,动态 81 W ≥ 地板) | 见 §2 | `reports/PILOT_H_REPORT.md` |

H pilot 途中发现并修复一个仿真器坑:基座块 `max_cloudlet_pes: 8` + `split_large_cloudlets: true` 会把 32-PE 作业悄悄切成 4 个 8-PE 碎片(MI 均分),盲臂假通过、规划器台账漂移。已在 H 块显式关闭(commit dde2f205),54 跑作业数均等于 trace 行数。

## 2. H pilot 结果(DESIGN_PILOT,不是判决)

54 跑 = divisor ×{1,2,4} × {reservation_edf, godeye, shuffle} × 6 格 × 1 个发现窗(k=2)。全部完成率 1.000、forced 0。

预注册推论"动态/地板比 ≥1 时碎片化代价可忽略"**被否定**:godeye 能耗比盲臂高 15–25%(绝对 +5…+21 Wh,比 2-PE 时的 +6/+14 Wh 更大)。变的是账本另一边:一个 32-PE 作业挪上绿电一次搬 81 W,绿电充足时收益盖过地板税。

| 档位 | godeye 胜盲臂 | 中位 godeye vs 盲臂 | godeye 胜 shuffle | shuffle vs 盲臂中位 | shuffle 保留率中位 |
|---|---|---|---|---|---|
| ×1 绿电过剩 | 4/6 | −18.1% | 4/6(n20 三格 −23…−65%) | +41%(5/6 格更差) | 0.36 |
| ×2 中度稀缺 | 1/6 | +18.6% | 1/6 | −1% | −0.27 |
| ×4 深度稀缺 | 3/6 | +0.9% | 三臂 ±5% 内 | 0% | 无意义 |

读法:×1 是 S2 家族第一次出现"预测内容本身"载重(S2 终局 shuffle 保留 106%,此处 0.36;错误预测比不用预测更差)。×2 追绿碎片化大于捕获,合并机制仍主导。×4 无杠杆。逐格散布极大(×1 档 godeye vs 盲臂 +45%…−55%),单窗口数字均低于已知噪声底(跨窗 10–16%),只有 ×1 跨 6 格 × 2 对照臂的整体形状算信号。

止损规则"甜点→注册全量 2-H;全输→盖棺"两者都不命中:×2/×4 全输或空,×1 形状正确但单窗口且机制故事被否定。

## 3. 碳轴的五条结构性证据(供你判断是否盖棺)

1. TB13-v3/v4:1728 格逐格最优,0 格过 15% 门;空载占碳账本 96%;完美预见 EVPI 中位 5.57%。
2. S2 终局:lead-0 保真后 anti 保留 70%、shuffle 106%,Stage A 的 57.8% 碳降来自"等一等再合并",不是预测未来。
3. E discovery:godeye 输给盲臂 reservation_edf。
4. F pilot:三档稀缺全输;G 作废。
5. H pilot:碎片化税不因宽作业消失;仅 ×1 有候选甜点。

共同结构:作业时机只能搬动态那一小块;关掉空载后主导杠杆是盲的合并;短作业让承诺窗口≈0。

## 4. 仍然活着的轴

rwtight(10× 时间拉伸真实风电 + 紧 slack)的完成率链未被推翻:盲等崩到 92.6% 超期,预测 defer 救回;反相预测让 vanilla timecap 碳 +14%、完成率 −10.9pp;EU-CRD 把 −10.9pp 救回 +1.9pp(单种子,干净 regret +15% 碳,blend −13.6pp 异常,多种子裁决未完成)。rwtight 的碳数字受旧功率语义(空载百分比当瓦特)影响需重评,完成率轴不受功率 bug 影响。

## 5. 请裁定

- R-a:是否接受 §6 机制修正(整合 vs 碎片化)替代"等待烧空载"。
- R-b:是否注册 Scheme 2-H 发现集,限定 divisor ×1(+×1.5 探边界),E 协议不变(发现窗 k=2/10/18、TIERS_E 阶梯含 shuffle/anti、盲臂 reservation_edf、门 1–5 数据前冻结、确认窗 k=26/34/42 一次性)。门 1 拟定:godeye 在发现格×窗口中位数上打败盲臂且 ≥2/3 格。若你认为 ×1 的单窗口证据不足以消耗发现窗,请直接判 STOP。
- R-c:若 2-H 不注册或再 STOP,是否正式盖棺该仿真器家族碳轴时间杠杆,论文主轴转向完成率/SLA(rwtight 链):预测误差让等待型调度错过截止期,EU-CRD 抵御;碳作为 iso-completion 次要指标。需要哪些重跑(功率语义修复后的 rwtight 重评、多种子)。
- R-d:3060 处置。retrain 预注册自动激活落在 C = PARKED;3060 的 F2 pilot 已报(两臂差 +1.9pc n.s.,策略把 defer 压到 4%)。

## 6. 文件指针

- `reports/PILOT_H_REPORT.md`(commit bf2da759),原始输出 `g1/compressed_timecap_s2/stage_a_out/piloth_m{1,2,4}_{reservation_edf,godeye,shuffle}/`
- `reports/S2_E_F_INTERIM_FOR_CODEX.md`(§6 修正 commit 28484116)
- `reports/SCHEME2_ERROR_REGRET_PREREG.md` + Addendum A(冻结,未改)
- 配置 `g1/compressed_timecap_s2/config_s2h_m{1,2,4}.yml`,生成器 `gen_s2.py::generate_h`,运行器 `run_stage_a.py pilot_h`
- commits 070dfd67 / f7ef5859 / dde2f205 / bf2da759
