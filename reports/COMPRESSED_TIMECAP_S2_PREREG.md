# Scheme 2 预注册:COMPRESSED 短视界合成正控

执行位置经用户指令改为本机(8 核,GPU 3060 过慢),工单其余纪律不变。本文件在任何
S2 碳数字产生之前提交;判据一经写入不改。身份逐字沿用工单:

> accelerated-weather synthetic mechanism positive control

COMPRESSED 结果只用"行 / epoch"描述,不得解释为现实时间尺度。

## 1. 基座与功率语义

    基座 block            config_C.yml 的 experiment_g1eval_matchedvan(逐键继承)
    拓扑                  五 DC;DC0 {12,36} DC1 {95,91} DC2 {96} 有风机,DC3/DC4 无
    碳因子                brown 0.08/0.35/0.55/0.75/0.92,green 0.01
    风年                  2021(默认),divisor 1500
    jar                   必须为修复 profile 功率接线后的新 build(功率硬门 commit 2d0f8dd
                          之后);RS500A 空载 51.4 W 精确派生;RS700A 仍为 24.7% 舍入派生
                          (106.21 W,与 SPEC 106 W 差 0.21 W,已登记待审计,本轮如实声明)
    与 TB13 数据关系      C-regime 风机 {12,36,91,95,96} 恰为 TB13 涡轮分割排除项,两线数据不相交

## 2. 场景轴(工单 §4 冻结值,一格一 trace)

    runtime_rows          {24, 48, 72}(格内统一)
    wait_cap_rows         {24, 48, 72, 96, 120}
    相容对                runtime + wait_cap <= 144   → 12 对
    concurrency           {1, 3, 5}
    n_jobs                {20, 35, 50}
    格数                  12 x 3 x 3 = 108
    作业形状              2 PE,MI = runtime_rows x 40000 x 1.0
    cloudlet utilization  1.0(场景假设:计算密集批作业;不得称实测分布)
    backstop              defer_deadline_force_mode: latest_start(runtime-aware),
                          测试钉死;基座的 legacy 600 s 不得继承
    到达                  服务跨度 S = ceil(n x r / c) 分区,每区一抽,域分离 seed
                          (sha256(cell + ":arrival"), namespace s2v1),禁止 clip
    deadline              arrival + runtime + wait_cap(绝对 sim 秒)
    闭合条件              (s − a) + r <= wait_cap + r <= 144,逐行断言
    逐格记录              arrival span / offered concurrency / runtime 分布 /
                          deadline 可达性 / trace SHA256(s2_manifest.json)

派生纪律:derived block 与基座的差异**只允许** {experiment_name, simulation_name,
cloudlet_trace_file, defer_deadline_force_mode, cloudlet_cpu_utilization} 五键,
由精确差分测试断言;max_episode_length 7200 与 green_episode_offset_range 44950 原样继承。

## 3. 六个窗口(仿真器自身 offset 调度,机械选取)

窗口 = reset 序号 k,offset = (1009 x k) mod 44950。冻结规则:k 从 1 升序扫描
(k = 0 为历史训练窗,隔离不用),保留与已留窗口 offset 两两相距 >= 7300
(> max_episode_length 7200,最坏情形也不重叠)且 offset + 7200 <= 52559 者,取前六:

    DISCOVERY      k=1  offset 1009      k=9  offset 9081      k=17 offset 17153
    CONFIRMATION   k=25 offset 25225     k=33 offset 33297     k=41 offset 41369

CONFIRMATION 三窗零触碰,直至 DISCOVERY 主门全过且臂与场景冻结后一次性读取。
TimeCAP 侧(仅 timecap_cal 标定与可选的 Stage B)只准用 2020 数据。

## 4. Stage A(零训练场景门,判据抄自工单 §7,不改)

每格 x 三个 DISCOVERY 窗,同一合同运行:最强因果盲(先按 DISCOVERY 池化总碳冻结单一臂,
不逐格选)、full curve oracle、oracle144。逐格合同:completion_rate_mi >= 0.995、
ontime_mi_share >= 0.995、deadline_forced_count == 0、planner_n_stale_dropped == 0、
n_unplanned_start == 0、n_wrong_dc == 0、n_dispatched_never_started == 0、
running_pes_over_cap == 0、同窗各臂 workload/weather/power/row signature 一致。

oracle144 主门(全部满足才算该格通过):相对冻结盲臂 pooled 总碳下降 >= 5%;
三窗至少 2/3 方向有利;capture >= 50%(相对 full-oracle 天花板);full oracle 不劣于盲臂。
最终须有 >= 3 个参数邻接的通过格;稳定区域多个时按 canonical cell JSON 的 SHA256
最小值冻结中心格。无稳定区域 → STOP_ORACLE144_GATE,不训练、不跑 RL。

## 5. Stage A′(加扰阶梯,判据抄自修正案 A §3,不改)

八档 `perturbed_oracle_planner`(godeye/s05/s15/s30/s60/timecap_cal/shuffle/anti)跑
Stage A 冻结的稳定区域。判据:收益沿质量单调不增(容平不容反超)、shuffle 与 anti 各
回吐 godeye 收益 >= 50% 且不优于 godeye、timecap_cal 保留 >= 50% 才准 Stage D、
全档合同全绿。timecap_cal 参数由现有 checkpoint 的 2020 验证残差标定(残差脚本已交付,
label-offset 先过 k=0 语义审计)。

## 6. 执行阶梯与提交纪律

    生成器 + 测试提交(本文件同 commit)
    → 生成 108 trace + config_s2.yml + manifest,提交
    → 烘焙 trace 进 resources,重建 installDist 一次,冻结 jar SHA
    → 单格单窗 smoke(合同通道对齐,不读主门)
    → Stage A 全网格(盲臂先行并冻结,再解 oracle)
    → Stage A′ 阶梯
    → 判读与 verdict 提交;Stage D(EU-CRD)另立预注册

产物目录 g1/compressed_timecap_s2/,不触碰 TB13 的任何目录;不编辑 drl-manager/Code/;
只提交本任务产生的文件。
