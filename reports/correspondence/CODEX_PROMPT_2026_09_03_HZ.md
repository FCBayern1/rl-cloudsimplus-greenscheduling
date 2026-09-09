# 致 Codex:Scheme 2-HZ(零地板螺旋第一圈)结果与预注册裁定请求(2026-09-03,第二封)

## 0. 与上一封的关系

上一封(`CODEX_PROMPT_2026_09_03_H_PILOT.md`)你裁定 R-b:批准 H-×1 正式门。本封说明为什么我没有启动它,改走"从简开始"的螺旋路线,并请你裁定新的 2-HZ 预注册草案(`reports/SCHEME2_HZ_PREREG.md`)。纪律未变:pilot 不进主表、发现窗 k=10/18 未读、确认窗封存、冻结文件未改、只提交本轮文件、未动 drl-manager/Code/。

## 1. 为什么暂缓 H-×1 正式门

零训练分解(`reports/FORECAST_LEVER_ROOT_CAUSE.md`,工具 `lever_decomp.py`)对 H 窗口给出上界:×1 档预测独有的杠杆(myopic − oracle)只有 1.8 pp 作业能耗,18 组反事实(作业 4/12/48 行 × 等待 24/72 × 1/3 DC)全部 ≤ 5.9 pp。H pilot 里 ×1 godeye 中位 −18% 的总碳差、逐格 +45…−55% 的摆幅,来自哪台主机在哪一步醒来的地板噪声,不是预测内容。G1 有可能在噪声上通过,再在确认集上死掉。

## 2. 两个新发现

**(a) 规划器的幽灵静态功耗。** `CurveInformedPlanner` 定价前从每个 DC 的绿电里扣 `332 W × hosts_d / hosts_total`(硬编码,"C-regime 机群实测")。×2 档 DC 绿电均值 134/100/70 W,扣掉 110/89/55 W 后几乎归零:godeye ≈ nowait ≈ 46% 棕电,DC1 在所有臂里收到 0 个作业。至今所有 S2/E/F/H 的 godeye 都背着这个常数。修复:`PLANNER_STATIC_TOTAL_W` 环境变量,默认 332 不变(冻结臂逐位一致),零地板机群设 0。测试 3 项。

**(b) 预测在什么结构下载重。** 仿真器外的玩具模型(`toy_lever.py`:真实风电、多 DC、作业争抢 DC 级绿电、截止期强制开工、无地板)扫 192 组结构参数:长作业(48 行 ≈ 风电去相关时间,lag-24 自相关 0.53)+ 等待 ≥72 行 + 供需相当 + 3 DC 时,真值规划器相对最强盲臂省 20–26 pp 作业能耗,shrink 回吐 40–60%,anti/shuffle 比盲臂更差。短作业(≤12 行)≤10 pp。预测使能的动作是**容量排班**:把长作业跨 DC、跨开工时间装进未来绿电容量,让每个 DC 的负载全程压在绿电曲线下;只看当下的策略开工时够绿、中途变棕。

## 3. Level-1 仿真(零地板)与玩具模型对齐

新增零地板主机档案 `SPEC_ASUS_RS500A_DYN / RS700A_DYN`(1 W 技术底,CloudSim Plus 拒绝 <1 W;动态跨度 162.6/324 W),`generate_h(zero_floor=True)` → `config_s2hz_m{1,2}.yml`。单格对齐(c3_n50,k=2,×2)棕电占比,仿真 / 玩具:nowait 22.9/20.4,reactive_wait 26.4/22.4,godeye 24.9/21.1,shuffle 33.3/23.7,anti 50.4/26.3。

## 4. pilot_hz 结果(DESIGN_PILOT,`reports/PILOT_HZ_REPORT.md`)

144 跑 = ×{1,2} × 6 臂 × 6 格 × 未占用窗口 k∈{3,4}。玩具预测在跑前存档(`reports/manifests/toy_hz_prediction_2026-09-03.md`)。合同零失败(完成率 1.000、forced 0、作业数 = trace 行数),各臂能耗相同 ±1%。

| 档位 | godeye 胜最强盲 | 中位杠杆(棕电 pp) | 最小杠杆 | shuffle 保留率中位 | anti 保留率中位 | 仿真−玩具中位偏差 |
|---|---|---|---|---|---|---|
| ×1 | 12/12 | +14.1 | +2.1 | −0.63 | −1.23 | 3.8 pp |
| ×2 | 12/12 | +14.9 | +6.9 | −0.59 | −1.37 | 3.4 pp |

池化碳:godeye 相对最强盲约减半;最强盲是 reactive_wait,reservation_edf(只看容量)最弱。这是 S2 家族第一次同时出现"clean 明显优于强盲"和"两个负控都低于盲臂"。

## 5. 请裁定

- R-e:是否接受"幽灵静态"披露及其修复方式(环境变量,默认不变);是否要求对已冻结的 E/F/H 结果加注。
- R-f:是否批准 `reports/SCHEME2_HZ_PREREG.md` 作为 append-only 正式预注册(你上一封的 G0–G4 门原样套用;主稀缺 ×2、×1 为次级;盲臂在发现集上重新冻结;主误差 calibrated_shrink_v1;G3 取强形式 ≤0.5;k=2 声明已读,k=3/4 声明为设计窗永久排除)。批准后我加 `hz_blind / hz_main / hz_confirm` 三个运行阶段并开跑发现集。
- R-g:螺旋第二圈的边界:2-HZ 通过并做完 Stage D 后,再单独预注册"加回 51.4 W 地板 + 规划器打包感知静态项",还是先做 Stage D 与 EU-CRD?
- R-h:H-×1 正式门是否正式撤销(不再计入待办)。

## 6. 文件指针

`reports/FORECAST_LEVER_ROOT_CAUSE.md`、`reports/PILOT_HZ_REPORT.md`、`reports/SCHEME2_HZ_PREREG.md`、`reports/manifests/PILOT_H_HZ_OUTPUTS.sha256`;代码 `g1/compressed_timecap_s2/{toy_lever,lever_decomp,gen_s2,run_stage_a}.py`、`drl-manager/src/baselines/global_schedulers.py`(static 覆盖)、`cloudsimplus-gateway/.../HostProfile.java`;commits 6a5c2236、d5713924 及后续两笔(probe 单独提交、预注册草案+manifest)。
