# 工单 3060:EU-CRD 消融族(6 臂,自成一体)

2026-08-27。作者决定把 3060 的算力从 TB12 V4 转到这里。**TB12 V4 不撤销,降为次优先**,
消融跑完再回去。协议约束沿用 Codex 已有裁定,未作任何放宽。

---

## 一、为什么是这个

G1 已 STOP,论文走 B 路线,主张是:

> EU-CRD 在注册的污染条件下没有降低脆弱性,反而**一致放大**了策略对 forecast 通道的响应。

支撑它的是四个污染条件下的放大倍数(blend 1.13× / anti 1.19× / noise 1.25× / shuffle 1.37×),
四条件同向。但 Codex 对措辞加了一条硬约束:

> 措辞不要直接写"信用重加权**必然**放大依赖",**除非消融独立证明因果**。

**所以消融是把"观察到放大"升级为"放大由某部件造成"的唯一途径。** 现在论文只能说
"observed 1.13–1.37× larger shifts",消融过了才能说是哪个部件干的。

## 二、跑什么

六个臂,种子 **101 / 102**,每臂 600k 步:

| 臂 | 配置键 | 相对 knSV3b 的唯一改动 |
|---|---|---|
| 基准 ×2 | `experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b` | — |
| ablG ×2 | `experiment_ctrl_ablG` | `crd.blender.fixed_c = 1.0`(门固定打开) |
| ablW ×2 | `experiment_ctrl_ablW` | `crd.responsibility.reweight_advantages = false` |

两个消融块在 `config_controls.yml`,**已从 knSV3b 程序化派生**(`drl-manager/make_control_arms.py`),
33 个守卫测试断言差异集恰好是一个叶子键 + 身份字段,且 `per_action_carbon_weight`
必须等于父臂的 0.25。开跑前跑一遍 `pytest drl-manager/tests/test_control_arms.py`。

约 **6 × 7.2h ≈ 43 小时**。

## 三、必须自成一体 —— 这条是有效性前提

**基准臂必须在 3060 上重训,不要拿 5080 的 knSV3b 来比。** 跨机比较会把机器效应混进去,
而配对估计量只在配对内同机时才消得掉。

好在论文本来就是这么设计的,附录 H 写着:

> Each variant differs from the base in exactly one configuration key, the base arm is
> evaluated through the same protocol as the variants, and **no cell is compared across
> training campaigns**.

所以整族搬到 3060 完全合规。三条要求:

1. **六个臂同一个 jar**
2. **开跑前记下该 jar 的 SHA256**,写进消融的 manifest
3. **不与主表任何数字交叉比较**,论文里注明消融族与主表来自不同构建

我们这边的冻结 jar 在 `/home/joshua/frozen/g1_gateway/lib`(SHA `aba6f0ed…`,1.1 GB)。
你们重新 build 未必逐字节相同,**这不影响族内有效性**,只要六个臂一致即可。

## 四、评测协议(与主战役逐字相同)

```
6 臂 × 3 条件(none / blend eps=1.0 / shuffle) × 3 注册窗口(low k=19 / mid k=56 / high k=34)
每格一局      ← 不是三局。--episodes 3 会跑到 k+1/k+2,是已证实的坑
argmax,eval seed 20260823
checkpoint 一律 600k 最终那个,不按结果挑
```

= 54 格。评测块要从各自的训练块**派生**并加 `green_episode_offset_range: 44950`,
差异集必须恰好是这一个键(见 `drl-manager/tests/test_g1_eval_blocks.py` 的写法)。

## 五、要报的量 —— 不只是碳

主结论是"放大",所以**必须报路由位移倍数**,这是 B 路线的核心量:

对每个臂,算它在每个污染条件下相对自己 Clean 的**路由分布 L1 位移**
(用 eval CSV 里的 `received_dc_*` 归一后算),然后报

```
放大倍数 = shift(消融臂) / shift(基准臂)
```

同时报碳(三窗池化 Σcarbon/Σcompleted_MI,种子级配对 log-ratio)和
加权棕电强度(用 `received_dc_i` 份额乘各 DC 的 `brown_carbon_factor`)——
后者在主战役里 5/5 同号地决定了碳的输赢。

## 六、预注册的判决规则(先冻)

**问题**:1.13–1.37× 的放大,由 gate 造成、由重加权造成、还是都不是?

| 结果 | 结论 |
|---|---|
| ablG 的放大倍数显著低于基准,ablW 不低 | 放大由 **gate** 造成 |
| ablW 低,ablG 不低 | 放大由**重加权**造成 |
| 两者都低 | 两个部件共同造成 |
| **两者都不低** | 放大不来自这两个部件,B 路线的因果措辞**不解锁**,只能停留在"observed" |

判据:2 种子,**只作描述性判断,不做显著性声称**。方向一致性(2/2)与倍数一并报。

**我的预测(低置信度,明确标注)**:两者都不低。理由是我在主战役上提出的"排序"机制假设
刚被自己的预注册判据推翻(`reports/G1_RANKING_PROBE_VERDICT.md`),我对自己当前的
机制直觉不再有信心。**预测错了如实记,不改判据。**

## 七、边界

- 这不是 G1 扩种子,**不改变 G1 的 STOP**
- 不寻找"能让 EU-CRD 翻正"的条件
- 消融跑完回 TB12 V4
- 期间 5080 在跑 T1(特征冗余度)和 T2(No-Forecast),两边不共享机器

## 八、给 Codex 的备案

作者已决定这个优先级调整。Codex 裁定第 5 条原本把 3060 分配给 TB12 V4,
本工单把消融提到其前,理由是他自己那条措辞约束把消融变成了 B 路线的前提。
协议层面未作任何放宽,请他在看到时确认或否决排序。
