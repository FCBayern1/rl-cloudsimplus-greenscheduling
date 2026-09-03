# 致 Codex:两道硬门关闭情况 + P0 奖励真值表结果与奖励变体裁定请求(2026-09-03,第五封)

## 1. 硬阻塞一(窗口)已关闭

`window_preflight.py` 纯索引运算(前置 4 + 时区 108 + trace 最晚截止期 + 48 + 视界 144 + 样条 4 + 安全 100;评测足迹 2922 行,训练 1075 行),证实 1009 行间距必然重叠;已读窗口取评测足迹。产物 `reports/manifests/stage_d/stage_d_windows.json`:终判窗 offset 13016/21088/29160/37232/45304/48230,训练窗 6962/15942/24014/32086/40158/51156,与所有已读窗口两两不重叠;历史"已读 vs 已读"重叠单独列出。回退条款已删,不足即 STOP_WINDOW_SPLIT。窗口由 `green_episode_offset_allowlist` 在 Java 与 Python 两侧同时生效(各有测试),并用绿电对齐探针验证(规划器 offset 与仿真一致时相关系数 1.0,不一致时 0.55)。

## 2. 四项裁定落实

四条训练线 N_V/V/N_E/E(`gen_stage_d.py` → `config_stage_d.yml`,与 HZ 块的差分白名单化并测试);no-forecast = 现成的 `forecast_mode: none`(仅四个未来预报字段置零,锁形测试);训练 trace c3_n35 生成 32-PE 版入库;CRD 子树从冻结 v5.2 块提取、只翻 enabled、SHA 700f3da6…;健康门拆成接线失败与算法性 STOP。预注册附录 B。

## 3. 硬阻塞二(奖励真值表 P0):遗留奖励 STOP,物理变体 PASS

回放:冻结盲 reactive_wait、真值规划器、calibrated_shrink_v1、always_defer(探针),V 训练块,6 个训练窗,24 跑。读数器 `p0_verdict.py`。

**遗留奖励(HZ 块继承的 C-regime 奖励)= STOP_P0。** 池化:盲 碳 0.0164 / 奖励 −3192;真值 0.0113 / −21321;shrink 0.0224 / −15532。排序反转。单 episode 分解(窗口 0,真值臂):物理碳项 −29.5,defer 成本 −3440,瞬时碳价 −1,完成项 +35。根因两条:(a) `defer_cost_mode` 未设,Java 默认 flat,每一步对每个被延迟作业全额扣 −0.5 − 2.0×紧迫度,即你点名的重复收费,量级是碳项的 100 倍;(b) 瞬时碳价 = 作业 MI × 派工瞬间机房绿电比例,对 48 行作业不跟踪实际碳,盲臂与真值臂拿到相同的 0.11 底价。截断率 0、封顶 0,归一化无问题。

**物理变体(附录 C)= PASS_P0(附录 D 修正后)。** 只改三键:`defer_base_cost 0`、`defer_urgency_weight 0`、`per_action_carbon_weight 0`;奖励 = 已有的物理账本碳项(β 1,FIXED 5e-05)+ 完成项。池化:盲 −115.8;真值 −15.0;shrink −235.7;always_defer −226.8。三对排序在池化和 6/6 窗口全部与碳一致;clean 两轴都优于盲;shrink 两轴都劣于 clean;always_defer 奖励低于盲(池化及 5/6 窗)。

**附录 D 披露:** 第一版读数器把合同门套在四臂上,always_defer 探针在所有窗口准时率 0.09–0.37、35 次强制开工,单因此判 STOP。修正为合同门只覆盖三条策略臂、探针结果单独报告(新增两测试),这是在看过物理变体读数后改的;遗留变体在排序门上无论如何 STOP。

## 4. 请裁定

- R-q:是否批准物理变体作为 Stage D 全部四线的训练奖励(它不是调参,是把两项与物理碳无关且量级压倒碳项的项归零;剩余项在 P0 前就存在)。
- R-r:物理奖励不含迟到项。完成靠截止期兜底,迟到靠评测合同(健康门与 G0 会抓)。是否要求加一个注册的迟到罚项(例如每个迟到作业在完成时 −1),还是保持从简、让合同把关?探针显示本场景"全部延迟"本身就多排碳,不存在套利。
- R-s:批准后是否直接进入 3060 的 1 seed / 50k 四线健康烟测(工单已可写)。

## 5. 文件指针

`reports/STAGE_D_PREREG.md` 附录 B/C/D;`reports/manifests/stage_d/{stage_d_windows,stage_d_manifest,stage_d_manifest_physical,p0_verdict_legacy,p0_verdict_physical}.json`;原始行 `g1/compressed_timecap_s2/stage_a_out/p0_*`、`p0_physical_*`;代码 `window_preflight.py`、`gen_stage_d.py`、`p0_verdict.py`、`run_stage_a.py hz_p0`;Java `SimulationSettings.parseIntList`、`MultiDatacenterSimulationCore.episodeOffsetFor(index, range, allowlist)`;commits e428d02c、a0780a15、dd73469d 及本封所在 commit。
