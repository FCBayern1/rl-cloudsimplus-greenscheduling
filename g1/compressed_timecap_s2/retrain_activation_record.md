# 重训预注册的自动触发判定（3060 侧机械执行）

时间：2026-09-02。判定依据：`reports/TIMECAP_RETRAIN_PREREG_DRAFT.md` §9.4。
判决输入：`g1/compressed_timecap_s2/ladder_v2_verdict.json`（commit `9eb9273`）。

本文件只记录**机械判定**，不改动任何已裁定文件，不新增任何判据。

## 1. 判决读数

    verdict                              STOP_LADDER_V2
    frozen_blind                         nowait_planner
    windows                              [25, 33, 41]（CONFIRMATION 三窗）
    region_cells                         97
    excluded_fraction                    0.0     排除率零，未触及 20% 上限

    gates.complete_and_contract_green    True
    gates.exclusions_within_cap          True
    gates.surrogate_retains_half         True
    gates.monotone_noise_axis            False   <- 失败
    gates.shuffle_destroys_half          False   <- 失败
    gates.anti_destroys_half             False   <- 失败

    median_retention
        godeye                           1.000000
        s05                              1.000586
        s15                              1.000014
        s30                              1.015992
        s60                              1.021707
        shuffle                          1.062859
        anti                             0.705573
        checkpoint_residual_surrogate_v2 0.938847

## 2. §9.4 情形匹配

    情形 A  四门全过（含 surrogate ≥50%）              不匹配：单调与两个负控均失败
    情形 B  单调/shuffle/anti/合同/排除率全过，
            仅 surrogate <50%                          不匹配：恰好相反，
                                                       surrogate 通过而其余三门失败
    情形 C  负控或单调或合同任一门失败                  **匹配**（三门失败）

## 3. 判定

> **情形 C。Scheme 2 按裁定 STOP。重训预注册保持 PARKED。**
>
> 3060 侧不启动重训，不产出 v3 checkpoint，不做 DC 级 v3 标定，不执行 G1′ 验收。
> `--label-start-offset 1` 的实现、T1–T5、预检 `--dry-run` 全部保留在库中备用，
> 但在新的授权到来之前不得运行 §4 的训练命令。

理由是裁定本身的措辞：**不为死考场重训。** 本轮失败的不是预测器质量这一环——
surrogate 档以 0.9388 的中位保留率通过了 ≥50% 的门。失败的是考场本身能否分辨预测价值：
噪声不伤（s05→s60 保留率不降反升到 1.0217）、时序被摧毁的 shuffle 反而拿到 1.0629、
连相位反转的 anti 都保留 0.7056。在一个连负控都拿得到收益的考场里，
把预测器换得更准不会改变任何结论。

## 4. 与本轮 3060 侧既有产物的关系

- **E1 持续性基线**（`persistence_baseline_cal.json`）不受影响，仍然有效：它是对
  2020 残差水平的直接测量，不依赖 ladder 的任何判决。按重训预注册 §6.3，
  它正是"这不是天花板"论证的量化版——**但请注意本轮的失败模式与 §6.3 设想的不同**：
  §6.3 设想的是"现实档失败"，而实际失败的是负控与单调，两者的论证含义不一样，
  §6.3 那段话在本轮不可直接套用。
- **k=0 审计**（`k0_semantics_audit.md`）的结论与本判决无关，继续有效：
  接线偏斜与 `peak_timing` 分母两处不一致仍然存在，仍然应该修，
  只是不再由重训这条路径驱动。
- **甲案实现与 T1–T5** 保留备用；若将来另立预注册要重训，它们不需要重做。

## 5. 3060 侧的后续动作

无。等待新的授权。不得以"顺手"为由运行 §4 命令或任何变体。
