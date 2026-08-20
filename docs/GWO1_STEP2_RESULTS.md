# 价值检查结论 + gwo1 场景建成认证 —— 3060 回报

`joshua-MS-7C09`（RTX 3060）→ 5080。生成于 **2026-08-20 08:30**。

对应你 08-19 那条指令的 ①②③④ 四项。

---

## 0. 三句话

1. **价值检查的答案与假设方向相反：放宽决策域不是放大优势，是摧毁优势**（窄门 −17.99%，宽门 **+25.65%**，三个锚点全部同号）。结论：**保持窄门，不要放宽**。
2. **gwo1 与 gwo1ho 已建成并双双 `AUDIT PASSED`**（各 30 项），暴露门 0.270 / 0.225 落在注册带宽 [0.20, 0.50] 内，cashability 1300 ≤ 1500 保留余量。
3. **本轮抓到 4 个会静默产生错误结果的问题**，其中 2 个已经在仓库里活了一段时间了（详见 §3）。

---

## 1. ④ 价值检查（你要的那个 30 分钟检查）

### 1.1 口径先说清楚

| 项 | 实际 |
|---|---|
| 场景 | **SQT2-cal**（`experiment_sqt2_noforecast`）+ t60 trace |
| 底座 | **greedy** |
| 锚点 | 0 / 79 / 158（`GWO1_ANCHORS=0,79,158`） |
| 臂 | nowait vs clairvoyant |
| 耗时 | 窄门 2h05m + 宽门 2h20m = **4h25m**（约 22 分钟/episode，远超"30 分钟"的估计，排产请按这个数） |

prescreen 当时只支持 `--schedule {cal,ho}`，所以跑的是 SQT2 而非 gwo1 —— 这正好是回答你问题的对照面，因为 −8.8% 参考值就在这个场景上。

### 1.2 结果

clairvoyant vs nowait 的碳差（负 = 更好）：

| 偏移 | nowait | 窄门 clair | Δ | 宽门 clair | Δ |
|---|---:|---:|---:|---:|---:|
| 0 | 0.1810 | 0.1502 | **−17.02%** | 0.2862 | **+58.12%** |
| 79711 | 0.2860 | 0.2248 | **−21.40%** | 0.3177 | **+11.08%** |
| 159422 | 0.2678 | 0.2276 | **−15.01%** | 0.3194 | **+19.27%** |
| **池化** | 0.7348 | 0.6026 | **−17.99%** | 0.9233 | **+25.65%** |

### 1.3 三点读数

**① 放宽决策域严格有害，且有机制、不是噪声。** 延迟量（MI·s）从窄门均值 12.4 万涨到宽门 63.5 万，**5.1 倍**。宽门允许在绿窗外延迟，于是 greedy 底座把本来已经落在绿窗里的作业推出了绿窗。`c@7200` 也从窄门的全 1.0000 掉到 0.9964 / 0.9966。

**② 但宽门这一列有一个混淆，必须标出来。** `blmax` 在宽门三个 episode **全部 = 200**（窄门是 182/200/182），backstop 上限被顶死。宽门测到的一部分是"200 这个 cap 变成了实际策略"，不是纯粹的决策域放宽。**+25.65% 是"放宽 + 撞 cap"的合成效应，不能单独归因给放宽。** 如果你要把"绿窗外延迟为什么有害"写成论文里的消融，我需要补一组 `blmax` 上限扫描才能解开这个混淆。

**③ 窄门在真仿真里有实打实的余量：greedy 底座 −17.99%，三锚全负。**

### 1.4 一处我先前口径错误的更正

我一开始把 −17.99% 说成"是 −8.8% 参考的两倍"。**这不成立，已收回** —— prescreen 文档里 −8% 那条线明写在 layer 4 上、绑定 **ppo 底座**（"the FORMAL base — conclusions must hold here"），而我跑的是 greedy。不同底座，不可比。

要拿到同底座可比的数，需要补一列 `--bases ppo` 的窄门跑（3 锚约 1 小时）。**这一列是否必须，取决于你要不要用它来判 DP 值不值得做。** 如果"窄门有余量 + 放宽有害"这两条已经够，就不用补。

### 1.5 对照臂自检

nowait 三行在 `GWO1_WIDE_DOMAIN=0` 与 `=1` 下**逐字节相同** —— 开关是外科式的，没碰对照面。这是我给这个开关设计的自证。

---

## 2. ② + 第 2 步：gwo1 场景建成并认证

|  | gwo1 (cal) | gwo1ho (held-out) |
|---|---|---|
| 认证 | **AUDIT PASSED**（30 项） | **AUDIT PASSED**（30 项） |
| 暴露门（带宽 [0.20,0.50]） | **0.270** | **0.225** |
| 锚点范围 | 0.106–0.323 | 0.165–0.342 |
| 绿窗到达 MI | 0.717 | 0.604 |
| cashability | 1300 ≤ ON_min 1500 ✅ | 1300 ≤ 1500 ✅ |
| 源 trace | `sqt2_n1200_t60` | `sqt2ho_n1200_t60` |
| 绿电序列 | `gwo1_schedule.json`（种子 20260818） | `gwo1ho_schedule.json`（种子 20260819） |
| 涡轮块 | 97xx | 98xx |

### 2.1 held-out 是怎么隔离的

`gen_gwo1_trace.py` 是确定性变换，`--seed` 只记进 artifact、不参与计算。**held-out 的隔离来自源 trace（sqt2ho）和绿电序列（gwo1ho，种子 20260819），不是来自 seed 参数。** 请按这个理解审我的做法。

### 2.2 config 是克隆的，不是手抄的

新写 `gen_gwo1_config.py`，从 SQT2 块克隆，**只改** trace / 涡轮块（+200）/ 名称 / preflight profile。理由：奖励、观测、Lagrangian、容量必须与 SQT2 逐字相同，否则 gwo1 的结论无法归因给"决策域 + trace"这两个唯一自变量。任何手抄都会引入不受控的第三个变量。

追加后 config_C.yml 原有 **40510 行逐字节未变**（已 diff 验证）。

### 2.3 ② `cloudlet_cpu_utilization: 1.0` —— 已是硬检查，且被继承

你要的那条已经在 preflight 里（`preflight_scenario.py`），文案带原因说明：

```
[PASS] gwo1: full CPU utilization (physics == registered maths)
       cloudlet_cpu_utilization=1.0 (0.5 stretches runtimes ~2.5x and voids every budget check)
```

克隆后 gwo1 自动继承，另外在生成期也加了断言（`test_full_cpu_utilization_is_inherited`），**两道**。

### 2.4 preflight 家族化

`sqt2_trough_*` 与 `gwo1_trough_*` 归为"绿电断续"族：共享几何/对称/预算类检查，暴露门各走各的（SQT2 量 tight/loose 类条件的"值不值得等"，gwo1 量"绿窗内到达、立刻释放会溢出到棕电"的 MI 占比）。

**SQT2 的两份认证输出与重构前基线逐字节相同** —— 我先存了基线才动手（`logs/baseline_*.txt` vs `logs/after_*.txt`），SQT2 已冻结，它的 30 项 PASS 是判决的前提。

新增 5 项 gwo1 专属门：`decision exposure (MAIN)` / `no dead anchor` / `registered runtime scale` / `source trace is t60` / `anchor modulus invariant`。

---

## 3. 本轮抓到的 4 个问题（全部会静默产生错误结果）

### 3.1 【已修复】config 指向的 gwo1 trace，仓库里是 t50 底座的

HEAD 里 4 个 gwo1 trace 的 calib artifact 写着 `source: "sqt2_n1200_t50.csv"`。t60 重生成版本一直只在工作区、从未提交。我第一个提交把 config 指向了 `gwo1_n1200_x130.csv`，而仓库里那份当时还是 t50 底座 —— **我认证的和仓库里的不是同一份**。

已用一个单独的 commit 把四个 scale 变体和 artifact 一起对齐。

### 3.2 【已修复】gwo1/gwo1ho 的 14 个涡轮 CSV 从未 git add

config 引用 97xx/98xx，但文件只在本地磁盘上。SQT2 的 95xx/96xx 是已跟踪的，所以这是遗漏而非 gitignore 规则。

**失效模式是最危险的那种：缺涡轮文件不会崩，数据中心只是报告零绿电。** clone 这个仓库跑 gwo1 会得到一个全棕电场景，所有碳数字静默无意义。

### 3.3 【已修复】`gen_gwo1_trace.py --src` 默认值还是 t50

重跑一次就会静默产出同名但底座不同的 trace。已改默认为 t60，并加了 `gwo1v1: source trace is t60` 这道门锁住。

### 3.4 【已修复】×1.30 把 max MI 顶到 52e6，超过观测上界 50e6

```
[**FAIL**] obs bound >= max MI    bound=50000000 vs max MI=52000000
```

缩放 runtime 同时缩放了 MI，`obs_cloudlet_mi_high` 会**静默截断**最大的那批作业。我按注册 scale 同比放大到 **65e6 = 50e6 × 1.30**，这样与 SQT2 的 1.25 倍余量比例**完全一致**，不引入新的自由参数。

**这一条需要你裁定，见 §4.1。**

### 3.5 【本地已排除，但是颗地雷】`build/install` 的 jar 是 08-11 的

价值检查第一次启动秒退，报 `NoSuchFileException: traces/sqt2_n1200_t60.csv`，但那文件在 `src/main/resources` 和 `build/resources/main` 里都在。根因是我在启动脚本里设了 `GATEWAY_LIBS`，它让 env 走 `java -cp build/install/.../lib/*` 直连 JVM，而那个 jar 打于 **08-11 16:19**。

**波及面已查清：没污染任何本地结果。** `GATEWAY_LIBS` 全仓只出现在 `isambard/*.sbatch`（HPC，那边每次自己 build）和 `~/rw_smoke.sh`。本地训练/评测/审计全走 gradlew 路径。是我自己加的，我的错。

但那个 jar **落后 7 个 Java 提交**，恰好包含你 ②③ 两条问的东西：

| 提交 | 内容 | 在旧 jar 里 |
|---|---|---|
| `62b2283` | `cloudlet_cpu_utilization` 配置旋钮 | ❌ |
| `c8849fd` | `ontime_mi_share` + 三重完成率合同 + 统一 B≤30s 放行 | ❌ |
| `375d843` | K=round(178m/9) 锚点集、latest_start backstop | ❌ |

**任何人在本地手设 `GATEWAY_LIBS` 就会静默跑回 SQT2.3 之前的 Java。** 我建议 preflight 再加一条：校验 jar mtime ≥ 最后一次 Java 提交时间。重建 jar 我排在队列里，等一个空档做（遵守"一次只跑一件事"）。

---

## 4. 需要你决策的三件事

### 4.1 `obs_cloudlet_mi_high` 50e6 → 65e6 算不算动了冻结参数？

我按**场景自洽性修复**处理了 —— 它是 preflight 在任何仿真跑之前抓到的截断缺陷，不是看了结果回调，且缩放因子直接取注册的 1.30、不引入自由度。

**如果你认为 `obs_cloudlet_mi_high` 属于冻结集，我把它写进预注册表再重新认证一遍。** 现在改成本很低，进了训练就贵了。

### 4.2 要不要补 `--bases ppo` 的窄门那一列？

见 §1.4。3 锚约 1 小时。只有你需要与 −8.8% 同底座可比时才必要。

### 4.3 §1.3② 的 `blmax` 混淆要不要解开？

只有当你打算把"绿窗外延迟为什么有害"写成论文消融时才需要补 cap 扫描。作为"不要放宽"的决策依据，现有数据已经够。

---

## 5. 提交清单（全部本地，未 push）

| 提交 | 内容 |
|---|---|
| `ac9f7a3` | `GWO1_WIDE_DOMAIN` / `GWO1_ANCHORS` 开关 + 价值检查结果写进 commit message + 9 测试 |
| `746251b` | gwo1 场景建成、preflight 家族化、双臂 AUDIT PASSED + 27 测试 |
| `a256d5c` | 对齐 t60 trace（§3.1） |
| `2381ded` | 补提 14 个 gwo1/gwo1ho 涡轮 CSV（§3.2） |

**测试**：新增 36 个（9 + 13 config + 14 preflight/接线）。全量套件 **930 passed**（本轮前 903）。

**21 个失败是预存在的，与本轮改动无交集** —— 已用 stash 掉我的改动复现同样的失败证明。成因三类：`oracle_fdefer_gate` 缺 `_persist`/`_fmt_arm`/`--results-json`（测试文件存在但实现没落地）、`config_C.yml` 缺 `experiment_multi_dc_5` 等两个块、pettingzoo 测试需要活的 JVM。**这三类要不要修，请指示** —— 按项目规则套件应该全绿，但它们都不在 gwo1 路径上。

---

## 6. 纪律执行情况

1. **命令里没有 `pkill`** —— 清理孤儿网关时按 PID 杀（`kill 1932542 1932232`），全程未用 `pkill -f MainMultiDC`。
2. **一次只跑一件事** —— 价值检查两个门串行；跑仿真期间不跑 gradle。
3. **没碰论文文件**（`paper_materials/`、`.tex`、`references.bib`）。
4. **没 push**，全部本地提交在 main 上。
5. **参数冻结** —— 除 §4.1 那一条（已单列请你裁定），没有因任何结果回调过参数。

---

## 附：证据文件

全部随本文档提交在 `docs/data/gwo1_step2/`：

```
docs/data/gwo1_step2/
├── gwo1_valuecheck.log      ← 价值检查原始输出（12 episode，§1）
├── baseline_sqt2.txt        ← preflight 重构前基线
├── baseline_sqt2ho.txt
├── after_sqt2.txt           ← 重构后
├── after_sqt2ho.txt
├── cert_gwo1.txt            ← gwo1 认证全文（§2）
└── cert_gwo1ho.txt          ← gwo1ho 认证全文
```

`diff baseline_sqt2.txt after_sqt2.txt` 与 `diff baseline_sqt2ho.txt after_sqt2ho.txt`
**应当无输出** —— 这是 §2.4 那句"SQT2 认证未被家族化重构改动"的可复核证据，
不必信我的话，跑一下 diff 即可。

---

## 7. 下一步

默认往 **DP 策略表接入仿真器**推进（第 2 步的后半段）。队列里还有：`--bases ppo` 窄门列（§4.2）、重建 08-11 的 jar（§3.5）。

你有任何一项要插队，随时说。
