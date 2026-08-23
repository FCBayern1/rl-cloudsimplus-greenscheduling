# 给 3060:`experiment_tb12_rl_fc` 的 crd 子树是前 v5 的

一句话:现在无害,**你们把 `crd.enabled` 打开的那一刻会变成灾难**,而且不报错。

顺带附一个三层自查脚本,和一条对你们有用的 Java 发现。

---

## 一、真问题:派生 EU-CRD 臂会继承 v4 的坏机制

`experiment_tb12_rl_fc` 的 `crd` 子树缺全部十个 v5 标记键:

```
forecast.scale_fix        forecast.carbon_norm       forecast.magnitude
responsibility.anomaly_gate  responsibility.normalize_shares
responsibility.share_scale_decay   ensemble.stable_bootstrap
delta_r.mode              baseline.kind              blender.tau_mode
```

现在没事,因为 `crd.enabled: False` —— fc/nofc 是纯 PPO,这些键读都不读。

但你们下一步要从 `tb12_rl_fc` 派生 EU-CRD 臂。开关一开,继承的就是 **v4 那套**:

- `R_forecast` 因 `predicted_wind_w` 差 1500 倍成了**有偏常数**;
- quarantine 因份额量纲病(`ρ_routing → 0.99`)**基本惰性**;
- 路由器 `Δr` **恒零**(gate 无兜底)。

这三条让我们整批 EU-CRD 数据作废,论文里每个 EU-CRD 数字都得重生成。
它会在 TB12 上原样重演,而且**不会有任何报错或告警** —— 通道名字还在,里面装的不是那个量。

正确做法:EU-CRD 臂的 `crd` 子树从 **C-regime 的 `..._eucrd_knSV3b`** 取(那是 v5 全量),
而不是从 `tb12_rl_fc` 继承。TB12 特有的部分(`green_interpolation_mode: STEP`、
风机、offset 轮换等)保留,`crd` 整个换掉。

## 二、三层自查脚本

`drl-manager/check_algo_provenance.py`(已推)。三层各自会独立变旧,而且都是静默失败:

| 层 | 旧了会怎样 | 查法 |
|---|---|---|
| Java 源 | 派发器 bug:`vm.getFreePesNumber()` 不更新,每步"挑最空 VM"起点就错 | `git merge-base --is-ancestor 61043cf HEAD`,查内容 |
| jar | 你以为在跑新代码,其实加载的是旧 jar | 从 `GATEWAY_LIBS` 取实际文件,打 sha256 |
| config | 新代码 + 旧 block = 还是旧算法 | 十个 v5 标记键的存在性 |

```
GATEWAY_LIBS=<你们的路径> python drl-manager/check_algo_provenance.py \
    --experiment experiment_tb12_rl_fc
```

**时间戳不是证据。** mtime 能被 touch,先 build 后 pull 会留下一个"看起来很新、实际由旧源编译"的 jar。
所以脚本查的全是内容:git 祖先关系、sha256、键存在性。

派发器 bug 对固定策略的实测幅度:碳 −0.7%,但 **episode 长度 −14.7%、能耗 −9.9%**。
头条数字上很小,**策略训练时面对的时序结构上很大** —— 这是它危险的地方,你不会从碳看出来。

## 三、唯一测行为而不是元数据的检查:接线哨兵

跑一局,从 `LoadBalancingBroker` 的 INFO 行统计**每 VM 的分派直方图**。
有 bug 时负载堆在前几个 VM,修好后摊开。我们 5-DC 上"承载 90% 负载的 VM 数"是 **65 → 155**。

绝对数依机群规模而定,**别抄我们的数** —— 拿你们自己的新旧 jar 各跑一局对比。

## 四、一条对你们有用的 Java 发现

`GreenEnergyProvider.java` 在 `5097899` 里加的那个守卫
("future bins 必须在 spline 构造失败时也存活"),走的是
`if (interpolationMode == SPLINE) {...} else {新守卫}` 的 **else 分支**。

- C-regime 两个主臂:`green_interpolation_mode` 未设 → Java 默认 **SPLINE** → 那段是死代码,
  所以我们的冻结 jar(早于该 commit 11 分钟)对 G1 无影响;
- **`tb12_rl_fc` 是 `STEP`** → 走 else 分支 → **这个修复是给你们的**。

确认一下你们训练用的 jar 是否已经包含 `5097899`。如果 fc_s1 是用更早的 jar 起的,
那它跑的是没有该守卫的版本。
