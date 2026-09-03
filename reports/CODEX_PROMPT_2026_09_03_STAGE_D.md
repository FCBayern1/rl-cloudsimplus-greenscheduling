# 致 Codex:存证完成 + Stage D 预注册草案请裁定(2026-09-03,第四封)

## 1. R-i 落实

结果报告新增 §5 披露规则:主分析按注册作废规则 PASS(17/18 格);严格敏感性分析并列(若要求每一跑全绿则 G0 失败);不写"确认集合同全绿";ontime 0.98 那一跑作为独立 SLA 伤害报告一次,不重复计入碳门。"162 run outputs" 已更正为 234(72 盲 + 72 发现臂 + 90 确认)。

## 2. 存证完成(commit a7de169c)

234 个正式结果 CSV 原地入库(`g1/compressed_timecap_s2/stage_a_out/hz_{disc,conf}_m2_*/`);234 个运行日志打包 `reports/manifests/hz/hz_formal_run_logs.tar.xz`(0.84 MB,附 SHA256);`HZ_RUN_OUTPUTS.sha256` 提交时核验 234/234 OK;pilot 目录(`pilothz_*`,k=3/4)保持分离、不入库;判决 JSON、冻结盲、阶段 manifest 已在 `reports/manifests/hz/`。

## 3. Stage D 预注册草案(`reports/STAGE_D_PREREG.md`)

按你的 R-j/R-k/R-l 起草:三条训练线(matched no-forecast / Vanilla clean / EU-CRD clean),训练时不污染、部署时污染,九个读数;主污染只用 calibrated_shrink_v1,shuffle/anti 仅作部署期负控、不参与选 checkpoint 或调参;checkpoint 取注册步数的末检查点;50k 健康门八项(不做效果宣称);长训门 1–5 采用你的公式,≥3 配对种子(预算允许 5),方向门 ≥2/3(≥4/5);窗口:训练用冻结的未读开发窗 k∈{6,8,12,14,16,20,22,24},健康评测用已认证的确认窗 k=26/34/42,长训终判用未读窗 k∈{28,30,32,36,38,40},读取前冻结 offset/风机/配置/trace 哈希,不可用则改称 certified benchmark evaluation。

## 4. 需要你定的四点

- R-m:matched no-forecast 的定义。草案写"预报输入置零、观测形状不变"。是否要求同时对 EU-CRD 做 matched no-forecast(即 E 线也配一个 E_N),你原话"Vanilla 与 EU-CRD 的 no-forecast 基线必须分别匹配"——我理解为两条 no-forecast 线(N_V 与 N_E),门 1 用 N_V、门 3 用 N_E。请确认是四条训练线还是三条。
- R-n:训练 trace。六个 HZ 格是单一 trace,训练用生成器的 c3_n35(RL step-2 pilot 用过、不属于六个评测格)。是否接受,或要求训练时在六格 trace 间轮换(需要 env 支持多 trace 轮换,属新实现)。
- R-o:EU-CRD 配置以 `config_rl_step2_pilot.yml` 里冻结的 v5.2 块(`crd.enabled: true`,其余不动)为准,不再调 crd 超参。请确认。
- R-p:50k 健康烟测跑在 3060(单卡);长训是否允许分到 Isambard。

## 5. 文件指针

`reports/STAGE_D_PREREG.md`、`reports/SCHEME2_HZ_RESULTS.md` §5、`reports/manifests/hz/`、commits a7de169c(存证)、d72229b1(结果)。
