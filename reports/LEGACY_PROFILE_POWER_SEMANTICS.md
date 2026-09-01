# legacy-profile-power-semantics:影响清单

**这是清单,不是改判。** 旧结果与旧 jar 永久保留并标记为 `legacy-profile-power-semantics`,
不得静默覆盖,也不得假设相对比较不受影响。

## 1. 缺陷

CloudSim Plus 8.5.5 的 `PowerModelHostSimple(maxPower, staticPower)` 第二个参数是**瓦特**:
构造器带 `maxPower has to be higher than staticPower` 校验,`getStaticPower()` 原样返回该参数。
`DatacenterSetup.createHost(HostProfile)` 传入的是 `profile.getStaticPowerPercent()`,即百分比。

因此每台 profile 主机的空载功率等于"空载百分比的数值",而不是空载瓦特。已在 commit
`137d90c` 修复为 `profile.getIdlePowerW()`,复现测试为 `HostPowerWiringTest`。

## 2. 受影响的空载功率

    profile                peak W   修前 idle W   修后 idle W   倍数
    SPEC_ASUS_RS500A          214          24.0        51.40    2.14x
    SPEC_ASUS_RS700A          430          24.7       106.21    4.30x
    SPEC_ASUS_RS720_E9        385          12.5        48.12    3.85x
    SPEC_ACER_R520            269          57.6       154.94    2.69x
    SPEC_ACER_AR360           315          22.0        69.30    3.15x
    LOW_POWER                 208          27.9        58.03    2.08x
    MEDIUM                    345          28.4        97.98    3.45x
    HIGH_PERFORMANCE          476          29.8       141.85    4.76x
    ULTRA_HIGH                634          30.6       194.00    6.34x

RS500A 的 51.40 已含 Addendum A 的精确派生;其余 profile 的同类舍入问题**另开审计**,
本轮未改动。

## 3. 受影响的范围

使用 profile 模式(`host_count_spec_*` 或 `hostProfiles`)的配置,已知包括

    drl-manager/configs/config_C.yml
    drl-manager/configs/config_controls.yml
    g1/config_C_phys.yml
    g1/config_C_2020.yml
    g1/ab/implicit.yml

这些实验**不能再声称使用了正确的 SPEC 空载功率**。空载被低估意味着静态地板偏小,
调度可移动份额相应偏大,方向上有利于"调度带来碳节省"的读数。

## 4. 明确不受影响的部分

TB13 v1、v2、v3 全部是离线模型,`instance_gen.HOST_IDLE_W = 51.4` 一直是瓦特,
从未走 Java 接线。三个 STOP 保持不变。

## 5. 未关闭的同类缺陷(独立登记)

非 profile 的默认主机路径仍有

    DatacenterSetup.java:159   new PowerModelHostSimple(250, 70)
    DatacenterSetup.java:337   maxPower = 250.0, staticPowerPercent = 70

其中 `70` 从命名与注释看同样是"把百分比当瓦特"(注释写的是 70% static power,
而 70 W 只占 250 W 的 28%)。它不阻塞使用 RS500A profile 的 v4,但**不得据此宣称
仓库的功率单位问题已全部关闭**。此项单独登记,待另行裁定。

## 6. 后续判定原则

    先做清单,不立即改判全部结果
    不得假设相对比较不受影响
    若训练奖励使用了旧功率账,关键实验可能需要重训,而不是用新 jar 重评一遍
    旧结果与旧 jar 永久保留,标记 legacy-profile-power-semantics
