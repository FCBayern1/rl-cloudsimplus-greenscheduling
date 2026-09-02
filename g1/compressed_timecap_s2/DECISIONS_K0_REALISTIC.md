# 决定记录:k=0 修法与"现实档"判读位置(2026-09-02)

针对 3060 审计报告(5ed29a6..b741b4b)提出的两项待决事项。

## 1. k=0 修法:采纳甲案,连同 peak_timing 分母,统一落在重训/部署预注册里

    甲案生效     将来重训的标签构造改为 pred[0] 对应当前行(r_begin = s_end - 1),
                 改动只落在 clean 数据包装器,不碰 drl-manager/Code/,
                 用测试钉死标签行对齐
    乙案否决     不改 Java;Java godeye 是在位的冻结观测通道,动它会改变
                 既有通道数值,代价与风险都更大
    peak_timing  N vs N-1 与行偏斜一并修,修在 Python provider 侧向 Java 看齐
                 (Java 为基准语义),配 Java/Python 逐位对拍测试
    时点         两处修复均写进 Stage D / 重训预注册后执行,现在不动任何代码;
                 在此之前 provider 的 "drop-in replacement" 表述一律不得使用

既有标定(timecap_cal.json / dc_residual_cal.json)不作废,理由沿用审计 §"不作废"三条:
它们测的是部署态,偏斜已计入残差,且网络无系统性位移。

## 2. "现实档"的判读位置:今晚不动任何阈值与表述

审计发现现有 checkpoint 在 lead 1-23 多数劣于零成本的持续性基线。处置:

    今晚的 ladder-v2 判读     surrogate 档的 ≥50% 门原样执行,不因新信息调整
                              (读数前改判据 = 污染一次性确认集,禁止)
    若 surrogate 档失败       结论措辞为"现有 checkpoint 已测质量兑现不了预报价值",
                              并附持续性基线对比作为"这不是天花板"的证据——
                              重训路线(干净装载 + 甲案标签 + 19x GPU)即为下一步,
                              重训出的新 checkpoint 需新标定、新档、新 addendum
    若 surrogate 档通过       持续性发现仅入讨论,不改变任何结论
    可选的 post-verdict 诊断  "persistence 质量参照档"可作诊断跑,不入判决

审计报告的三处警告标记(汇总 RMSE 非对齐统计量、AMP 差异不可逐位比、
clean smoke 与 stock 不可比)与"判别规则先于读数写定"的纪律照单采纳并存档。
