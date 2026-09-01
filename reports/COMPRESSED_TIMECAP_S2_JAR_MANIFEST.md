# 方案二 jar / 源码 / 配置 manifest

日期：2026-09-01。分支：`gpu/compressed-timecap-s2`。
配套预注册：`reports/COMPRESSED_TIMECAP_S2_PREREG.md`。

**本 manifest 只记录构建，未运行任何碳评测。**

## 1. gateway jar

    构建命令        cd cloudsimplus-gateway && ./gradlew -q installDist
    产物路径        cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib/cloudsimplus-gateway.jar
    大小            1089374305 bytes
    构建时间        2026-09-01 22:46:07 +0100
    SHA256          cdb17b5f8cee82e617d5a9db4d2dc246948cf2fd2465ba8c7dce0d3bf1a578e3

jar 本身不入 Git（1.0 GB）。上表的路径、大小与 SHA256 就是它在本仓库中的登记方式。
运行时必须显式设置

    GATEWAY_LIBS=<repo>/cloudsimplus-gateway/build/install/cloudsimplus-gateway/lib

并在每份产物中回写该 jar 的 SHA256；不一致的运行作废。

## 2. 源码来源

    仓库 HEAD                 d91d39ce6c5592b3ce224e6d0ff669a99692cae5
    最后触碰 Java 的提交      2d0f8dd44c645d7e14e50f5034204574a1efbc1b
                              "tb13 v4: derive RS500A idle from the SPEC watts and add
                               the two-layer power gate"  (2026-09-01 22:07:27 +0100)

profile 功率接线修复已包含在本次构建中（两条提交均为 HEAD 的祖先，已核验）：

    137d90c  profile hosts: pass idle watts to PowerModelHostSimple instead of the idle percentage
    2d0f8dd  tb13 v4: derive RS500A idle from the SPEC watts and add the two-layer power gate

因此工单第 3 节的要求满足：本轮使用**修复 profile 功率接线之后**的 jar。
被替换掉的旧 jar 构建于 2026-08-26 00:39，早于这两条提交，其产生的一切数值
（含旧的 `-68.36%`）不得作为本轮输入或通过证据。

**本轮未改动任何 Java 源码。** jar 需要重建，只是因为磁盘上那份早于 profile 功率修复，
且新增的 108 条 trace 必须进入 classpath。

## 3. 构建与测试环境

    Java        openjdk 21.0.12 2026-07-21 (OpenJDK 21.0.12+8-1-24.04-Ubuntu)
    Gradle      8.14
    平台        Linux 7.0.0-30-generic

Java 测试（本次实跑，非缓存；结果时间戳 2026-09-01 22:46:47）：

    ./gradlew test     BUILD SUCCESSFUL
    21 个测试类，124 个测试，0 failures / 0 errors / 0 skipped

Python 测试：

    drl-manager/.venv/bin/python -m pytest g1/compressed_timecap_s2/ -q
    65 passed

## 4. classpath 资源核验

新增资源必须真的进入 jar，否则会**静默**变成"trace 不存在 / 零绿电"而不是崩溃：

    traces/cts2_*.csv                                108 / 108 在 jar 内
    windProduction/simplified/Turbine_12_2021.csv    在 jar 内（1433699 bytes）

（后续每次向 `src/main/resources` 增删文件之后，都必须重跑 `installDist` 并重做这一节的核验。）

## 5. 配置与场景哈希

    base config                       g1/config_C_2020.yml
    base block                        experiment_g1eval_matchedvan
    windows.json selection_hash       e216d4d35f0a7320e523335ae42bbed27138b451b51daa3bbc9f6c8e67bb793f
    windows.json 文件 SHA256           9f646c4621246ad497a7eedacbbc32ccdeb8365a242222c89839cf157e7db10c
    config_cts2_stage_a.yml SHA256    a5b924dde7f2c876bbb5afb6a995f9380135e18783c031e9d536a8d5afc05502
    cells                             108
    traces                            108，逐条 SHA256 见 g1/compressed_timecap_s2/workloads.json

## 6. 尚未完成、Stage A 开跑前必须补上的

1. **clock zero 复核**：`CLOCK0_SEC = 13.0` 继承自 C-regime 实测，尚未在本配置下重测。
   见预注册第 3.5 节，偏差超过 `guard_rows = 64` 即 `STOP_CLOCK_ZERO`。
2. **冻结盲臂**：四个候选盲臂的 pooled 结果必须先跑完并写入
   `g1/compressed_timecap_s2/frozen_blind.json`，之后才允许运行 oracle 臂（预注册第 7.4 节）。
3. **Stage C 的 `timecap.forecast_perturbation`**：gateway 尚未实现 shuffle / anti 开关。
   Stage C 不得在实现并有测试证明它确实改变了送达调度器的预测之前启动（预注册第 6.2 节）。
