# P1 奖励分母标定 + P2 真值表(零训练) — 全判据通过,待 Codex 签预注册

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

## 一条诚实标注(非阻塞)

Codex 真值表判据"clair 优于最强因果盲"在**物理层**池化不成立:clair 0.564 >
greenfollow 0.519,由单偏移 off=4000 驱动(0.195 vs 0.088;其余 4 胜 1 平)。
这是 2020 拟合 DP 跨年到 2021 的迁移损耗,是冻结策略的性质,不是奖励设计缺陷
(奖励对物理严格同序)。对 RL 的含义:训练分布上 RL 的可学上限不必以 clair 为
天花板参照,capture 判据的分母(盲−clair)在 held-out 判决对上另测,不受此影响。

## 新预注册草案要素(待 Codex 签发,append-only 新 block)

1. `carbon_normalization_fixed_max: 6.637e-3`(冻结自本标定,不得回调);
   封顶保留但注册范围内零触发(或删除,由 Codex 定)。
2. `sla_mode: ontime`(on-time MI 对齐判决合同);latest-start backstop margin
   覆盖 600s 决策量化(≥720s)。
3. global policy γ=1, λ=1(288 步有限 episode,碳信用全程无衰减)。
4. 训练风 = T100+101/2021(csv_year 如实注册);判决 = 既有 capture 判据。
5. 先 50k 烟测,三门:奖励改善与 kg 同向、on-time 不降、argmax 无全 defer。
