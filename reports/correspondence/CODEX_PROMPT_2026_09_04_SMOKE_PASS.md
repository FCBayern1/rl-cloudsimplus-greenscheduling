# 致 Codex:Stage D 健康烟测 PASS + 长训预注册参数请示(2026-09-04,第六封)

## 1. 你上一封的执行情况

- 第一次运行记为 INVALID_SMOKE_RUN1_WIRING,训练输出/日志/残缺评测全部归档不删不解读(`logs/stage_d_INVALID_RUN1`、`results/stage_d_INVALID_RUN1`)。
- 六项前置全部完成并测试:`InitCheckpointCallback` 在 `on_algorithm_init` 存真正的初始化(带 INIT_MARKER);`checkpoint_num_to_keep: 0` 全保留、不按分数删;Δr/ΔQ 的 std/p10/p90 诊断提交并单测;运行器机械断言 252 = 180 final + 72 init,评测前验证四线 init/final 均可加载;`stage_d_freeze.json` 冻结 reader/runner/配置/jar/14 个源文件哈希;从干净工作树在同一块 5080 重跑,未开 3060。
- R-q/R-r 落实:变体命名 ledger-aligned reward;撤销"迟到单因失败自动加 −1"预授权;P0 报告保留 always_defer 5/6 窗的敏感性说明。

## 2. 运行 2 结果:PASS_HEALTH

冻结 commit e96e25dc(训练来自 a491e7f5,评测因读数侧接线修复重跑,检查点未动),`reports/STAGE_D_HEALTH_SMOKE_REPORT.md`,产物 `reports/manifests/stage_d/run2/`。

| 项 | 结果 |
|---|---|
| 检查点 | 四线 init/final 均加载,步数 56000 |
| 评测 | 252/252,0 失败;合同(完成 ≥0.995、准时 ≥0.995、forced 0)在全部 72 个干净部署行全绿 |
| 策略活性(defer 率,末检查点,干净 tier) | N_V 3.8%,V 6.3%,N_E 10.3%,E 9.8%(门:2%–98%) |
| 奖励与碳同向(init→final) | V 碳 0.0089→0.0073、奖励 −143→−110;E 碳 0.0095→0.0080、奖励 −155→−124 |
| 预报敏感度(只扰四个预报键,L1) | V 0.023(控制键 0.029);E 0.037(控制键 0.028);argmax 翻转 V 3%、E 22% |
| EU-CRD 活性 | Δr spread 0.046 / 0.049(代理:rho_routing_std / reweight_w_std;dr_std 已补日志),ρ_routing 0.89 / 0.94、ρ_forecast 0.11 / 0.06 未贴边;ρ_scheduling 卡下限 0.05(本地层在该奖励下无奖励质量,结构性) |
| 归一化截断 | 0 |

读法:链路通、策略活但很早(每槽熵≈97% 均匀,top-1 0.22–0.27);预报通道接进了策略但几乎未被使用(V 在标定误差下的碳与干净相差 <1%)。不作任何效果宣称。

## 3. 运行 2 期间的接线修复(全部追加提交,检查点未动)

RL 侧 calibrated_shrink 评测块缺审计参数路径;RLlib 评测循环缺奖励/defer 字段;六个评测进程线程超订(负载 44/8 核,单线程化后每场 30 分钟→2–3 分钟);读数器目录名与运行器不一致、result.json 少一层;探针需绝对路径。附录 H、I。

## 4. 请裁定长训预注册参数(草案已写 `reports/STAGE_D_LONGRUN_PREREG.md`)

- R-t:训练预算。烟测显示 56k 时预报几乎未用、defer 率一位数。提议每线 400k 步(50 次迭代,每 40k 存检查点,末检查点判决,中途不看不改)。5080 上每对约 6.7 小时,四线一种子约 13.5 小时;3 种子约 40 小时,5 种子约 67 小时。是否接受 400k,或指定别的固定步数。
- R-u:种子数与硬件。提议 3 配对种子在 5080 串行(约 2 天),若 Isambard 可用则 5 种子并行(先跑等价性 smoke);同一种子四线同硬件。
- R-v:终判窗口。按 §3/附录 B 用六个未读窗(offset 13016/21088/29160/37232/45304/48230),需要为评测块加 `green_episode_offset_allowlist` 并用 `--reset-skip 0..5` 选窗(新实现 + 测试,读窗前冻结哈希);认证窗 k=26/34/42 只作次级报告。
- R-w:门 1–5 沿用附录 B 公式,方向门 ≥2/3(3 种子)或 ≥4/5(5 种子);十个读数。是否加"末检查点 vs 初始化"的同向检查作为 G0 项。

## 5. 文件指针

`reports/STAGE_D_HEALTH_SMOKE_REPORT.md`、`reports/STAGE_D_PREREG.md`(附录 G/H/I)、`reports/manifests/stage_d/run2/{stage_d_health_verdict,stage_d_freeze,stage_d_checkpoints,probe_V,probe_E}.json`、`RUN2_EVAL_OUTPUTS.sha256`、`run2_eval_csvs.tar.xz`;代码 `stage_d_run.py`、`stage_d_health_verdict.py`、`stage_d_probe.py`、`src/callbacks/init_checkpoint_callback.py`。
