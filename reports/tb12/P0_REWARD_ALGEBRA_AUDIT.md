# P0 零训练奖励代数审计(Codex 裁定执行) — 真凶:封顶抹掉 97% 碳

日期:2026-08-25。全部离线,来自两臂 progress.csv,零新训练。
前置:Codex 裁定推翻 RL_FIX_PROPOSAL.md 的两处机制归因(该文已作废,见文末更正)。

## 三个恒等式

| 恒等式 | fc | nofc | 结论 |
|---|---|---|---|
| 1: term = −β·ΣĈ | 1.000000 | 1.000000 | PASS,簿记自洽 |
| 2: ΣĈ / (Σsignal/fixed_max) | **0.0290** | **0.0306** | **封顶抹掉 97.1% / 96.9%** |
| 3: Σsignal = total_carbon_kg | 1.000000 | 1.000000 | PASS,无两本碳账口径差 |

## 定量诊断

- TB12 每步碳 ≈ 5.4e-4 kg(总 0.157 kg / ~288 步;host 静态功率主导,见 P0-3)。
- `carbon_normalization_fixed_max: 2.0e-05` → 比值 ≈ 27,3.0 封顶几乎步步触发。
  **fixed_max 量纲错约 27 倍**——按 gwo1 尺度沿用,未为 TB12 重标。
- 后果:碳项退化为"高排放步计数器":brown ≥ 6e-5 kg 的步恒罚 3.0·β,
  **超出部分零边际成本**。defer-爆发的套利通道即此(爆发步已封顶,增量免费);
  惩罚均值实测 1.62–1.78(混合:全绿步低罚 + 封顶步 3.0)。
- 行为佐证:nofc episode 长度 134→240 步(defer 坍缩),ΣĈ 却 236→234 几乎不动;
  fc 134→148 步,ΣĈ 231.7→222.6——两臂的"改善"都发生在被封顶阉割后的信号上。

## resolved config 复核(与 Codex/5080 指认一致)

- `carbon_normalization_mode: FIXED`, fixed_max=2e-05(config_C.yml tb12 block,
  开训前 72 min 提交 f02ce8b)——**无 runningMax 棘轮**。
- global policy resolved γ=0.999(params.json `algorithm_config_overrides_per_module/
  global_policy/gamma`),λ=0.98 → γλ≈0.979,150 步衰减至 ~4.2%——长程信用
  衰减真实存在但是从犯,主犯是封顶。

## 对 P1 最小修复的量化输入

- 冻结分母重标:按校准段 per-step 碳的 p99(~1e-3 kg 量级)定 fixed_max,
  使注册运行范围内比值 <3(即封顶零触发),或直接移除封顶。
- 其余按 Codex 裁定:sla_mode → on-time MI;latest-start margin 覆盖 600s
  量化;global γ=λ=1;先过 P2 真值表(nowait/greenfollow/clair/always-defer
  四轨,always-defer 不得夺冠、cap rate=0、碳-奖励同序)再准 50k 烟测。

## 更正记录

RL_FIX_PROPOSAL.md 两处错误,Codex 已裁定、本审计证实:
1. "runningMax 棘轮"不成立——正式两臂 FIXED 生效;
2. "γ=0.99 折 0.22"错——实际 γ=0.999/λ=0.98,GAE 有效衰减 ~4.2%(更重);
3. (补)per_slot_credit 是 padding mask,非边际碳信用,R2b 方案作废。
该文保留原样作为审计痕迹,以本文与 Codex 裁定为准。
