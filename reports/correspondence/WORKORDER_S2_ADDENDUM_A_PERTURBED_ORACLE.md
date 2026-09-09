# Scheme 2 工单修正案 A:加扰 oracle 预测质量阶梯(Stage A′)

Append-only,附于 `WORKORDER_GPU_COMPRESSED_TIMECAP_SCHEME2.md`。
**状态:待 Codex 批准 Stage B 推迟条款;Stage A′ 的基建与 Stage A 不冲突,可先行。**

## 1. 动机

原工单把两个命题捆在一起:RL 能否使用质量为 q 的预测(命题 A),与 TimeCAP 能否产出
质量 q(命题 B)。Stage B 重训只给一个 q 点,失败时分不清死因。本修正案用加扰 oracle
把预测质量做成受控自变量,先拿到整条剂量–响应曲线,再决定要不要为命题 B 烧 GPU。

## 2. 冻结的阶梯(代码已提交,见 §6)

    tier          语义
    godeye        sigma = 0,与 oracle144 逐位相同(阶梯的零点,有测试钉死)
    s05/s15/s30/s60   相对噪声 5%/15%/30%/60%
    timecap_cal   sigma/rho/alpha 取自现有 TimeCAP checkpoint 验证残差的标定产物
    shuffle       整段冻结置换:边缘分布保留,时序摧毁(负控)
    anti          时间反转:边缘分布保留,相位反转(负控)

误差模型(执行前冻结,不得看碳后调):

    view[tau] = max(0, G[tau] + lead_scale(tau−t) · sigma_rel · scale_ref · eps[tau])
    eps           每 (site, tier, episode) 一个冻结 AR(1) 场,rho = 0.8
                  误差模式跨决策步持续,不逐步重采样(重采样会让规划器把噪声平均掉)
    lead_scale    alpha + (1−alpha)·lead/H,alpha = 0.25(近期误差小,远期误差大)
    scale_ref     该站真值序列的平均绝对值(sigma_rel 无量纲,跨站跨 divisor 可比)
    确定性        一切由 (序列字节, site, tier) 经 sha256 域分离派生,跨机复现
    结算          永远用真值曲线;只有规划器的眼睛被腐蚀

## 3. Stage A′ 判据(执行前冻结)

在 Stage A 的同一冻结网格、同一合同、同一最强盲臂上,每格增跑八个 tier 的
`perturbed_oracle_planner`。通过条件:

    单调性     pooled 总碳收益(相对冻结盲臂)沿 godeye→s05→s15→s30→s60 不增
               (允许相邻两档打平,不允许反超 godeye)
    负控归零   shuffle 与 anti 各自回吐 godeye 收益的 >= 50%,且不得优于 godeye
    现实档     timecap_cal 档保留 godeye 收益的 >= 50%,则 RL 值得做;
               < 50% 但 > 0 记边缘;<= 0 记 STOP_REALISTIC_QUALITY
    合同       全部 tier 每格合同全绿(完成率、逾期、强派、台账七项与 Stage A 相同)

`STOP_REALISTIC_QUALITY` 的含义:完美预报有价值,但已知可达的预测质量兑现不了它——
此时不训 TimeCAP、不跑 RL,负结果照常提交。

## 4. 对原工单阶段的改动(待批)

    Stage A    不变,仍先跑
    Stage A′   新增,紧随 Stage A,纯 CPU
    Stage B    TimeCAP 重训推迟为可选的 realization 补点,仅当 A′ 全过且需要
               "真网络也能兑现"的证据时执行
    Stage C    预测输入换为冻结的加扰 oracle(timecap_cal 档为主档),
               判据结构不变(clean→godeye 对应,负控为 shuffle/anti)
    Stage D    RL × 阶梯剂量–响应:各档独立训练臂,判据 = RL 收益随质量单调退化
               且负控归零;EU-CRD 在其后,同一阶梯上对照 vanilla credit

措辞约束不变并加严:一切产物只能称 **synthetic forecast-quality ladder**,不得写成
TimeCAP 实验;timecap_cal 档只能称"标定到现有 checkpoint 已测残差水平的合成档"。

## 5. GPU 执行顺序(交接单)

    1  git fetch && 从 origin/main 建分支(本修正案与代码均已在 main)
    2  跑 drl-manager/tests/test_forecast_perturb.py(17 个测试)确认环境
    3  残差标定:g1/compressed_timecap_s2/residual_calibration.py
       —— 用现有 checkpoint + 注册验证切分(2020),多台风机配对传入
       —— ⚠ label-offset 是审计点(工单 §6 的 k=0 语义),先审计再定值
       —— 产出 timecap_cal.json,提交并记 SHA
    4  按原工单 §5 起草 COMPRESSED_TIMECAP_S2_PREREG.md,并把本修正案 §2/§3 的
       阶梯与判据抄入冻结
    5  Stage A(原样)→ Stage A′:evaluate.py 注册表已有 perturbed_oracle_planner,
       逐 tier 设 PLANNER_PERTURB_TIER(timecap_cal 档加 PLANNER_PERTURB_CAL)
    6  判读按 §3,产物 + verdict 提交;过线才谈 Stage D 预算

## 6. 已交付的代码(本机完成,已测)

    drl-manager/src/baselines/forecast_perturb.py       冻结阶梯,纯函数
    drl-manager/src/baselines/global_schedulers.py      PerturbedOraclePlannerGlobalScheduler
                                                        registry: perturbed_oracle_planner
                                                        (对既有家族 append-only,66 旧测试仍绿)
    drl-manager/tests/test_forecast_perturb.py          17 测试:零点逐位相等、单调、
                                                        误差场冻结、AR(1)、lead 缩放、
                                                        负控语义、站独立、tier 拒非法
    g1/compressed_timecap_s2/residual_calibration.py    GPU 侧标定脚本(走部署推理路径)

## 7. 两条线的关系

本阶梯不依赖 TB13-v4 的判决。v4 若过,同一 `perturbed_oracle_planner` 可原样搬到
REAL_TIME 考场做同样的剂量–响应;造一次,两个考场共用。
