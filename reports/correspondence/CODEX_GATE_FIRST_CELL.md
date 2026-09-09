# 2021 校准第一格失败 —— 漂移哨兵咬中我自己的 drain 实现

裁定全部落实(2020 range 24669、2022 撤下、容量三项锁、reactive_wait),
预注册与 Addendum A 已冻结。随后跑第一格,**四项合同同时不过**,
我据此**停住了 margin 网格**,没有按计划烧那 9 小时。

---

## 1. 第一格结果(curve_planner / low k=19 / margin 2 / terminal 12000)

    carbon_per_completion_mi   0.086168
    completion_rate_mi         1.0000     ok
    ontime_mi_share            0.9312     FAIL   需 >= 0.995
    deadline_forced_count      512        FAIL   需 0
    planner_n_stale_dropped    512        FAIL   需 0
    planner_occ_max_over_cap   0.0        ok
    planner_drift_abs_max      6855       FAIL   需 0
    workload/created/finished  8000/8000/8000  ok
    episode_length             9474       自然收尾,未撞 12000
    planner_n_drain_pulled     4542
    planner_cap_observed       480;384;296;240;144

该格已移入 `g1/gate2021/invalid_m2_predrainfix/` 标记无效,不进任何池化。

## 2. 两条已确认的事实

**`forced == stale == 512`,数字完全相同。** 一条自洽因果链:512 个作业被 Java backstop 强派,
规划器不知情,其预约随后作废。这与 margin 无关 —— margin 只改抢跑时机,
不改变「规划器看不见强派」这一事实。

**我先提出又自己推翻的一个假设。** 我一度怀疑本地 `drain` 不看 PE 容量,导致仿真器报负的可用 PE。
用 `nowait_planner` 跑 1500 步取证,结论相反:

    avail min per DC          [199, 232, 137, 240, 144]
    implied in-use max        [281, 152, 159,   0,   0]    cap [480, 384, 296, 240, 144]
    steps with any avail < 0  0 / 1500

`dc_available_pes` 确实跟踪运行负载,且从未变负,在用量始终在容量内。**该假设撤回。**

## 3. 当前主假设:漂移源自我边界 drain 的实现缺陷

`n_drain_pulled = 4542` —— 边界一到,规划器把 4542 个预约**同时**拉起,
一次性灌进容量 480/384/296 的站点,约十倍超订。而我的 drain 分支是

    self._release(d, s, e, pp)
    self._hold(d, self.t, self.t + (e - s), pp)

**没有过可行性检查**。作业实际在本地排队,规划器却把它们记作正在运行,
预测占用与实测负载就此分叉。这与 `occ_max_over_cap = 0` 不矛盾:
后者只约束我自己的账本,漂移量来自实测侧。

正是你预先点名的那一类 ——「active 是规划器推算的运行账,不是 Java 实际开始事件……
若发生队列或启动漂移,本轮无效,不能靠账本自洽掩盖」。

**取证状态**:正在用 1500 步边界的短跑直接观察 `avail` 在边界处的行为,
以确认超订时刻与 drain 时刻重合。结论未落定前不改代码。

## 4. 我倾向的修法(请裁)

**drain 受容量限速**:边界之后按可行性逐步派出,摊到剩余步里,而不是一次性倾倒。
仍在你「drain 到自然完成」的语义内,且让规划器的信念与现实对齐。
**我不打算改哨兵口径去迁就实现** —— 哨兵这次是对的。

---

# 请裁定四件事

## (一)drain 限速是否即为正解?

若同意,细则请定:边界后每步派出量的上界取什么?候选:
(a) 当步可行容量(`cap - occ[:, t]`)允许多少派多少,按预约的原承诺站点排队;
(b) 固定速率(例如每步不超过 N 个作业);
(c) 保持一次性倾倒,但把 `active` 改成由实测负载驱动而非由派工时刻推算。

(c) 会让规划器的容量账依赖观测反馈,与你此前否决的「反推」相近,所以我倾向 (a),但不自决。

## (二)`forced == 512` 怎么归零?

即使 drain 修好,backstop 强派仍会发生:被强派的作业规划器看不见,预约必然变陈旧。
margin 网格能压低强派频率,但**能否压到恰好 0** 我没有把握。
若跑满 {2,4,8,16,32,64} 仍有 forced,按冻结判据这是 STOP,还是你要另设处置?

## (三)是否需要「已强派作业」的观测通道?

根治办法是让规划器知道哪些作业被 backstop 拿走了。
现有观测无此信息(`is_deferred` 只标「这个槽是重新出现的」)。
Java 侧加一个 evaluator-only 的「上一步被强派的 cloudlet id 列表」是可行的,与稳定 ID 同一条路子,
不进策略观测。**要不要做?** 做的话本轮 jar 需再构建一次并更新冻结哈希。

## (四)时间预算

单格实测 22 分钟,8 核允许 4 路并行,一个 margin 档 15 格约 1.5 小时。
若 (三) 要做,还要加一次 Java 改动 + 重建 + 回归。
请确认是先修 drain 跑一档看合同,还是把 (三) 一起做完再进网格。

---

## 附:本轮已交付且已验证的东西(供你判断底座是否可信)

    Java  batchCloudletIds 稳定 ID,空槽 -1,108 个 Java 测试 0 失败
    P0-5  evaluate 不转发 time_to_deadline,旧规划器每槽拿到 1e9,deadline 从未生效
    容量  真值 min(VM PE, 主机 PE) = [480,384,296,240,144];配置算术与 host*64 都错
          三项锁已实现:仅零负载时读、reset 后重冻、不等于登记向量即停
    账本  scratch / reservations / active 三层,occ 唯一真值,27 项测试全过 + 变异校验
    恒等  逐 DC 能量-碳审计在 blind trace 上逐位闭合(carbon/green/brown/energy rel=0.000%)
    数据  preflight 拒绝占位数据;2022 五涡轮全 REJECT,2021/2020 全 ok
    回归  全量 Python 套件 1218 passed,18 failed 与干净树同集,已逐条隔离
