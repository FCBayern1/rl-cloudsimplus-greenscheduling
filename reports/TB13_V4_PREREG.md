# TB13-v4 预注册:物理拓扑重标定

独立注册。v1、v2、v3 的三个 STOP 永久保留,本轮数字不得回溯解释前三轮,前三轮数字亦不得
替代本轮判决。本轮不是放大功率让数字过线,而是把 64 核主机、可调度容量与峰值功率第一次
统一到同一个物理模型里。

## 1. v3 为什么必须重标定

v3 的 STOP 不是"预报没有价值",是容量与功率曲线没有对齐:

    Java HostProfile.SPEC_ASUS_RS500A   64 核, idle 51.4 W, peak 214 W
    v3 每站暴露                          16 PE,却支付整台 64 核主机的静态功耗
    v3 cloudlet utilization              0.5,16 PE 全占只增加 20.3 W
    动态能量上限                         182.9 Wh  对固定静态 3,700.8 Wh

即一台 64 核主机被长期限制在最多 12.5% CPU 利用率,注册的总碳 15% 门在算术上几乎不可达。
这不是门限问题,改分母也不作为主判据。

## 2. v4 唯一允许的物理映射

不扫描常数,只注册一个由 HostProfile 推出的配置。

    每站主机              1 x RS500A
    主机物理容量          64 PE
    VM                    2 x 32 PE
    可调度容量            64 PE
    作业 PE               {8, 16, 32}
    流体负控              8 PE  = 12.5%
    正式作业              16 PE = 25%,  32 PE = 50%
    cloudlet utilization  1.0
    idle_host_power_down  false
    idle / peak           51.4 / 214 W
    dynamic per PE        (214 - 51.4) / 64 = 2.540625 W

物理闭合,三点精确:

    0  个繁忙 PE    51.4000 W
    32 个繁忙 PE   132.7000 W
    64 个繁忙 PE   214.0000 W

两个 32-PE 计算密集作业恰好填满一台 64 核主机并达到注册峰值。

## 3. 必须写明的场景假设

`cloudlet utilization = 1.0` 的语义是"TB13 模拟计算密集型批作业",这是**场景假设**,
不是实测负载分布,任何产物与论文文字都不得把它写成实测。

离线模型把每站的 64 PE 当作单一容量池,不建模 2 个 VM 的分配粒度。一个 32-PE 作业恰好
占满一个 VM,两个 16-PE 作业在池化口径下可共处一站。这是**离线模型的作用域声明**,
真实仿真器门(§8)是检验它的地方,不在本轮判决内消化。

## 4. v4 沿用 v3、不得改动的部分

    时间轴与十分钟物理行距
    89 组相容轴规则与三条逐格断言(arrival_span > 1、deadline <= horizon、S == ceil(sum r / c))
    六个确定性季节窗口与零时区语义(base offset 4307 / 13067 / 21827 / 30587 / 39347 / 48107)
    DISCOVERY / CONFIRMATION 涡轮分割,CONFIRMATION 零触碰
    随机数域分离(":arrival:" 与 ":runtime:")、冻结 seed 的字节语义、内容哈希
    最严 budget_fraction = 0.10 的纯可行性验收与 MAX_RETRIES = 64 的生成器 STOP 条款
    reservation_edf_blind 的精确语义(v2 Addendum A.4)与全格履约门
    installed_divisor 轴、concurrency 轴、n_jobs 轴、wait_cap 轴、budget_fraction 轴
    block cohort 规则:1 anchor x 1 (n_jobs, wait_cap) x 3 divisor 邻域 x 4 budget = 12 cells,
      最多 144 block 共 1,728 cells,层轮转后按 block SHA,不拆邻域、不拆 budget、不读绿电与碳
    Round 0 物理门的五个量与阈值(正相关 0.70-0.95、同时贫风非退化、最优 DC 变化 >= 10%、无 rho 截断)
    Phase A 冻结单一盲臂之后才准运行碳 oracle

## 5. v4 相对 v3 改变的部分

    CAP_PES_PER_SITE      16 -> 64
    VMS_PER_SITE x PES_PER_VM   2 x 8 -> 2 x 32
    PES_PER_JOB           {2, 4, 8} -> {8, 16, 32}
    FLUID_CONTROL_PES     {2} -> {8}
    cloudlet utilization  0.5 -> 1.0
    DYN_W_PER_PE          1.2703 -> 2.540625

`MIN_PES_SHARE = 0.25` 不变:8/64 = 12.5% 仍是流体负控,16/64 = 25% 与 32/64 = 50% 是正式作业。

因功率与 PES 轴改变,**Round 0 与 cohort 必须整体重跑**,v3 的 Round 0 产物与 cohort
保留但不用于 v4。相容轴规则不依赖功率,89 组与 267 个 workload key 上限不变。

## 6. 判据

    主门        exact 模型总碳 EVPI >= 15%,分母是总碳,不改
    诊断        可移动碳口径的 EVPI 只作诊断并列报告,不参与任何判决
    其余门      proven OPTIMAL、立即启动比例落在 20-80%、pes_share >= 25%,沿用 v3
    通过标准    需要**稳定邻域**达标,而不是零星单格:以 block 为单位统计,
                至少一个完整 block(12 cells)全部 advancing,并报告 advancing block 数

## 7. 开跑前功率哨兵(先于一切)

在**真实 Java 仿真器**上做 0 / 32 / 64 个繁忙 PE 的三点微测,要求全部成立:

    实测主机功率      51.4 / 132.7 / 214.0 W
    VM 总 PE          64
    cloudlet utilization  确为 1.0
    idle power-down   确为 false

任一不符即 **STOP**,不得用离线近似替代,不得先跑阶梯再补测。哨兵的原始输出、配置与
commit 一并作为产物存档。

## 8. 终止条款

若 v4 的 exact 总碳 EVPI 仍无稳定邻域达到 15%,TB13 这条场景搜索正式收兵,
**不再开 v5 调常数**。若通过,才进入真实仿真器 oracle >= 10% 与 TimeCAP 门。

## 9. 预期算术(预测,不是结果)

按 v4 常数直接算能量账本:

    T=144 n=12 pes=32   静态 3,700.8 Wh   可移动 1,463.4 Wh   份额 28.3%
    T=144 n=12 pes=16   静态 3,700.8 Wh   可移动   731.7 Wh   份额 16.5%
    T=72  n=12 pes=32   静态 1,850.4 Wh   可移动 1,463.4 Wh   份额 44.2%
    v3 同规模参照                                              份额  2.41%

v3 的 1,728 个已解格在可移动碳口径下 EVPI 中位 29.05%、p75 51.81%。若该分布在功率重标定
后近似不变,总碳口径的 EVPI 中位落在 7% 上下、上四分位一带接近或越过 15%。这是**预测**,
其前提(更大的作业改变容量约束与调度自由度)并不保证成立,v4 仍可能 STOP,§8 已写明后果。

## 10. 执行阶梯(冻结顺序)

    功率哨兵(真实仿真器,§7)
    → v4 轴 / 窗口门(89 组与六窗复验,常数换为 v4)
    → Round 0-v4 物理门(clean commit,记录源码 / 预注册 / 窗口 / data split 的 SHA)
    → 冻结最多 144 个完整 block 的 cohort-v4
    → 零碳 preflight(读 cohort-v4,全过之前不跑任何碳比较与 exact oracle)
    → Phase A 冻结单一盲臂
    → Phase B exact 模型与 EVPI

产物目录独立:`round0_v4_out/`、`zero_emission_v4_out/`、`round1_v4_out/`,不覆盖 v1/v2/v3。

## Addendum A:RS500A 空载功率常数的精确派生(append-only)

本 addendum 写在**任何 v4 碳结果产生之前**,commit 与哨兵产物可核。这不是按结果调功率常数。

官方 SPECpower_ssj2008 记录的原始测量值是

    idle  51.4 W
    peak  214 W
    来源  https://www.spec.org/power_ssj2008/results/res2020q1/power_ssj2008-20191125-01011.html
          (ASUSTeK RS500A-E10-PS4, AMD EPYC 7742, 提交号 20191125-01011)

`HostProfile.SPEC_ASUS_RS500A()` 存的 `24.0` 是 `51.4 / 214` 四舍五入后的**派生值**,
按它回算得 51.36 W,与原始记录差 0.04 W。修正为由原始值精确派生:

    staticPowerPercent = 100 * 51.4 / 214 = 24.018691588785046

修正后

    getIdlePowerW()        51.4 W
    50% 利用率             132.7 W
    满载                   214.0 W

判据不放宽。§7 的"实测与 51.4 / 132.7 / 214 W 一致"保持字面相等,代码测试里只允许
`1e-9 W` 的浮点表示容差,这是数值表示容差,不是 ±0.05 W 的判据容差。

只修 RS500A 一个常数。其余 profile 的同类舍入问题另开审计,不在本轮顺手改。

## Addendum B:功率硬门的两层结构(append-only)

§7 的哨兵必须同时通过两层,缺一即 STOP:

    第一层  构造路径测试
            经 DatacenterSetup 的建主机路径拿到的 PowerModelHostSimple
            在 0 / 32 / 64 个繁忙 PE 上给出 51.4 / 132.7 / 214.0 W

    第二层  真实仿真微测
            在真实 CloudSimPlus 仿真里实际跑出 0 / 32 / 64 个繁忙 PE
            由能量增量除以时间反算平均功率,而不是直接调用功率模型函数
            同时确认  2 x 32-PE VM,合计 64 PE
                      cloudlet utilization 确为 1.0
                      idle_host_power_down 确为 false
                      32 PE 与 64 PE 阶段确实形成稳定利用率平台

两层全过后冻结:source commit、jar SHA、配置 SHA、哨兵原始输出,然后才进入 Round 0-v4。
