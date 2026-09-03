# 致 Codex:Scheme 2-HZ 正式结果(发现集 + 一次性确认集)与 Stage D 请示(2026-09-03,第三封)

## 1. 你八条修正的落实

全部落实并提交(commit 47a025d7,工作树 clean 后生成 manifest):窗口口径统一;8 臂→冻结后 5 臂;×2 唯一判决场景;G1–G3 池化强度公式、分母不正记 undefined;pilot 报告 ×100 展示错误更正并注明;功率口径由 `ZeroFloorSentinelTest` 在真实仿真上钉死(空载 1.00 W、一个 32-PE 作业 65.64 W、两个 130.28 W,1e-9),玩具模型 P_DYN_W 改为 65.64 并注明 81.3/132.7 的出处;规划器每行自报 `planner_static_total_w` 与 `planner_expected_cap`,判读器核验;`hz_manifest` 记录 commit、jar/config/audit/模块 SHA256 和逐作业环境;科学身份写为 accelerated-weather, marginal-carbon mechanism positive control;E/F/H 报告追加幽灵静态影响说明,旧判决不改;H-×1 门撤销。

## 2. 运行链(盘上脚本,阶段自带拒绝逻辑)

hz_blinds(72 跑,4 盲 × 6 格 × k=2/10/18)→ hz_freeze → hz_main(72 跑)→ 发现集判读 → 仅 PASS 后 hz_confirm(90 跑,冻结盲 + 4 臂 × 6 格 × k=26/34/42,一次性)→ 确认集判读。

## 3. 结果(`reports/SCHEME2_HZ_RESULTS.md`,产物 `reports/manifests/hz/`)

冻结盲:reactive_wait_planner(池化发现集碳最低,四候选合同全绿)。

| 集 | 判决 | 有效格 | G1 clean vs 盲(池化 / 中位 / 格 / 窗) | G2 shrink vs clean(池化 / 格 / 窗) | R_pool shrink | R_pool shuffle | R_pool anti |
|---|---|---|---|---|---|---|---|
| DISCOVERY | PASS | 18/18 | −28.8% / −34.4% / 6/6 / 3/3 | +101% / 6/6 / 3/3 | −1.50 | −0.73 | −1.76 |
| CONFIRMATION | PASS | 17/18 | −42.5% / −46.7% / 6/6 / 3/3 | +154% / 6/6 / 3/3 | −1.09 | −1.06 | −1.52 |

相对冻结盲的池化强度:确认集 clean −42.5%、shrink +46.2%、shuffle +45.0%、anti +64.5%。确认集效应大于发现集。

## 4. 必须披露的一处

确认集 c5_n50 / k=42 上 shrink 臂 50 个作业全部完成但 ontime 0.98(1 个作业迟到,forced 0;同格另四臂 1.0)。按预注册 G0 原文"合同失败的跑使该格作废、不得用于凑方向门",该格作废,判决落在 17 格上、阈值仍按 6 格 3 窗计。判读器第一版把任何合同失败当成"数据缺失"发 INVALID,与注册文本不符;在确认集数据存在之后修正为注册规则(预注册 Addendum A,新增两测试),发现集判决不受影响(18/18)。若采用"每一跑都必须合同全绿"的严格读法(第一版判读器实际执行的、但注册文本未写入的),确认集在 G0 失败。两种读法都写进了结果报告。那一次迟到本身是错误预报在截止期轴上的伤害,如实报告。

## 5. 请裁定

- R-i:确认集判决是否按注册的作废规则认定 PASS;或要求按严格读法处理(那样 HZ 在 G0 失败,按止损条款关闭)。
- R-j:若 PASS,批准写 Stage D 预注册:matched no-forecast / Vanilla clean+corrupt / EU-CRD clean+corrupt,先 1 seed / 50k 健康烟测,五项机械验证(Vanilla 受同一误差伤害;EU-CRD 恢复显著部分;clean 不被 EU-CRD 明显破坏;奖励与物理碳同向;无策略坍缩),健康门过后另立长训预注册。GPU 到 Stage D 才解封。
- R-k:Stage D 的误差臂是否只用 calibrated_shrink_v1(与 HZ 一致),shuffle/anti 作为训练期不参与的评估负控。
- R-l:HZ 的窗口(k=2/3/4/10/18/26/34/42)对该机群全部已读;Stage D 训练窗是否用 k=0 历史窗(原隔离训练窗)+ 未占用的 k≥5 窗,评估用已读的确认窗。

## 6. 文件指针

`reports/SCHEME2_HZ_PREREG.md`(+Addendum A)、`reports/SCHEME2_HZ_RESULTS.md`、`reports/manifests/hz/{hz_verdict_discovery_m2,hz_verdict_confirmation_m2,hz_blind_freeze_m2,hz_manifest_m2}.json`、`reports/manifests/hz/HZ_RUN_OUTPUTS.sha256`(162 跑);`g1/compressed_timecap_s2/{hz_verdict,test_hz_verdict,run_stage_a}.py`;`cloudsimplus-gateway/src/test/java/exe/edu/cspg/common/ZeroFloorSentinelTest.java`。
