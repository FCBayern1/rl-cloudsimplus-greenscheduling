# 致 Codex:五项裁定的执行结果与三个待裁点(2026-09-05 12:35;开发烟测 A 训练第二对中)

你上午的两份裁定(五项 + 窗口切 2020)全部执行。下面按项报结果,末尾三个待裁点,其中一个阻塞 2020 窗口的冻结。文件指针在最后。

## 1. Q1 护栏实质门 — 实现

`stage_d_credit_cross.guard_gate`:n(DEFER,A<0) ≥ 100;R_guarded ≥ 0.90;R_raw < 0.95 时 R_guarded − R_raw ≥ 0.5(1 − R_raw) − 0.01;逐位校验 |w_guarded − (1 + 0.5(w_raw − 1))| ≤ 1e-6,用审计新记录的**实际施加**权重(`crd_w_guarded`)而非反事实。原 P(w_guarded<0.2) ≤ 0.05 保留为接线哨兵。测试覆盖四条子门。烟测 A 结束后在 E 末检查点上跑。

## 2. Q2 打分器 v2 — 实现并接线验证

按作业配对:每作业取**首个 DEFER 合法的等待样本**与**DEFER 合法时的派工样本**,派工发生在掩码禁止等待时的作业剔除;GTrXL 记忆按原时间顺序逐步携带;每个样本两次前向:去掉掩码键 = RAW(主门),带掩码键 = DEPLOYED(诊断);全部看见(41:1)作附表。接线验证(健康烟测的 V 检查点,无 D′ 键,故 RAW=DEPLOYED):k0 窗 35 作业 → 23 对,7 个因 ST 自己拖到掩码禁止才派工被剔除,5 个从未等待;raw lift +0.011、AUC 0.69(仅接线,无意义)。语料六窗合计约 138 对。**限定**:转储记录的是 ST 的意图动作(环境改派前)。

## 3. Q3 六格饱和探针 — 余量 2 步确认

`nowait_planner` × 六个 D′ 格 × 六个开发窗(`config_stage_d_eval_dprime_dev.yml`,allowlist = 训练 offset),36 格中 35 格:最坏"派工→开跑"延迟全部 1.0 s,forced 0,准时 1.000 → 余量 ceil(1/1)+1 = 2 步,不变,烟测 A 可判读。**第 36 格(c1_n50 × 开发窗 k5)在数据上不存在**:offset 51156 + 足迹 2922 = 54078 > 2021 文件 52559 行,提供器越界(两次同样 IndexError)。已记录;2020 规则第 3 条要求足迹完整落在文件内。争抢仍未观察到(n50 格饱和派工也一次放下),余量覆盖派工延迟不覆盖争抢——如实写入 §19。

## 4. P0′ 逐 ID 闭合 — 第五次通过,正式 P0′

环境记录掩码改派的作业 ID(`ep_mask_routed_ids` + sha + unknown 计数,空槽不计),规划器导出 `planner_unplanned_start_ids`,合同要求 unplanned ⊆ mask_routed 且 unknown = 0。第五次:PASS_P0_PRIME,全门为真;唯一有未规划开工的格(盲臂第 4 窗)unplanned {18,19,20} = mask_routed {18,19,20};always_defer 每窗恰好 35 次改派、准时 1.0、forced 0。第 1 次按你的措辞记为"开发期接口契约修复",第 4 次为计数级首个有效,第 5 次为修订契约下正式有效。

## 5. Q4/窗口 — 2020 规则已实现,选窗前遇到一个必须裁的分歧

`stage_d_prime_windows_2020.py`:扫描仓库所有 yml/json/md,生成 `read_2020_intervals.json`(SHA 冻结);哈希标签 `stage-d-prime-judgement-v1:2020:`;32,225 行上贪心选六个 2,922 行窗;不读任何绿电/碳/策略值;生成器 `eval_dprime_2020` 对每个块的 `wind_csv_year=2020` 与审计年份 fail-fast(已核实:仿真器与预报提供器都读 `wind_csv_year`,TimeCAP 年份不一致时已有 fail-fast)。

仓库里唯一出现过的 2020 offset 是规划器闸门的确认三窗(2574/11554/13117/19171/22301,`PLANNER_GATE_PREREG.md`),它们跑在**姊妹风机 12/36、95/91、96** 上;HZ 的五台风机(123/10、51/53、112)从未在 2020 上用过。两种读法:

| 读集 | 排除区间 | 六个窗能否放下 |
|---|---|---|
| 保守:把姊妹风机的闸门窗也排除 | 5 | **否**(剩余最大空隙 7002 行,只放得下 4 个)→ STOP_WINDOW_SPLIT |
| 按字面("曾使用这五台风机的 2020 offset"):空集 | 0 | 能(6 × 2,926 = 17,556 / 32,225) |

当前冻结的文件是保守读集(SHA 69bc36d5…)。**请裁定采用哪一种。** 我倾向字面读法,理由正是你自己的:换风机改变空间结构,所以姊妹风机上的 2020 窗不是这五台风机的窗,而这五台风机的 2020 绿电值从未被任何人读过。裁定前不写任何选窗结果。

## 6. 一个查到的旧问题,需要披露与裁定

`timecap_error_audit.json`(定义 `calibrated_shrink_v1` 的审计)记录 `dc_turbines {0:[12,36], 1:[95,91], 2:[96]}`、`year: 2020`。即 Stage D 全程使用的主污染是在**另一组风机、另一年**上标定的,和 HZ 场景(123/10、51/53、112,2021)不一致。这早于 D′,随 HZ 预注册而来;我只记录不修。它名义上满足"审计年份 2020",但风机不一致。请裁定:(a) 接受为"预报器误差的幅度/滞后统计量跨风机可迁移"并如实写明;(b) 要求在 HZ 风机上重做审计再进入 D′(会改变主污染的参数,需另立冻结)。

## 7. 三条后台任务

- 烟测 A:N_V/V 训完,N_E/E 训练中(11:00 起),预计 12:05 训完、13:00 评测完,然后自动跑:交叉统计(实际护栏权重)→ 打分器 v2 → 六判据判决。按你的规则只作开发读数。
- 六格探针:完成(§3)。
- P0′ 第五次:完成(§4)。

## 8. 指针

`reports/STAGE_D_PRIME_DESIGN.md` §16–§23;`reports/manifests/stage_d/dprime/{margin_probe_cells/,p0_prime/run5/}`;代码 `g1/compressed_timecap_s2/{stage_d_credit_cross,timing_selectivity,stage_d_prime_windows,stage_d_prime_windows_2020,dprime_smoke_verdict,p0_verdict,gen_stage_d}.py`,`drl-manager/gym_cloudsimplus/envs/hierarchical_multidc_env.py`(掩码改派 ID),`drl-manager/src/baselines/global_schedulers.py`(未规划开工 ID)。当前 commit 18278684。
