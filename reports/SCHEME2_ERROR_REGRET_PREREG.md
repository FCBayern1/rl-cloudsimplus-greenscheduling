# Scheme 2-E 预注册:真实预测误差的 regret 门

按 Codex 2026-09-02 工单(§2–§5)起草。身份逐字沿用:

> Scheme 2-E: realistic forecast-error regret successor
> accelerated-weather synthetic mechanism positive control

不进入 REAL_TIME 物理证据链;COMPRESSED 结果只以"行/epoch"描述。旧判决(TB13-v4、
Stage A PASS、A′ STOP、ladder-v2 STOP)永久保留,本注册不回写任何一项。

## 1. 研究问题(error-to-regret)

不再要求所有预测错误都伤害调度。问:真实 TimeCAP 会产生哪些系统性错误?其中是否存在
一种**预先注册、现实可观测、能够改变动作并显著增加总碳**的错误?EU-CRD 能否比 Vanilla
更好地抵抗它?

## 2. 场景与数据(冻结)

    基座            S2 的 108 格 workload 网格、trace、派生 block 七键纪律、
                    latest_start backstop、util 1.0、新 jar 功率语义,全部原样沿用
    风机与窗口      e_data_split.json(SHA e8ae414fcf1a1d96,commit fc3ebbd,机械发牌):
                    DISCOVERY  T123/10/51/53/112,k=2/10/18(offset 2018/10090/18162)
                    CONFIRMATION T52/71/47/54/3,k=26/34/42(offset 26234/34306/42378)
                    两集风机零交集、窗口零交集且均从未被任何臂运行;
                    与训练风机 1/15/30、S2 五机、TB13 四十八机、封存 116/117 零交集;2022 禁用
    评测格          全部 108 格(不复用旧 97 格区域——那是旧风机上的 Stage A 产物)
    lead-0          全档保真(ladder-v2 R3 语义,含全部盲臂、clean 与污染臂)

## 3. 盲族与冻结规则(§5.1)

四臂,均源码级禁读未来绿电(逐臂测试断言):

    nowait              现有 nowait_planner
    reactive_wait       现有 reactive_wait_planner
    reservation_edf     新臂:到达即在容量台账上取最早可行 (start, DC),EDF 平局序,
                        预约不可撤销;不读任何绿电(现在与未来都不读)
    load_smoothing      新臂:只按队列/容量/死线摊铺——slot 定价 = 预约台账上的
                        占用重叠,取重叠最小的 (start, DC),平局取更早;不读任何绿电

在任何 clean/corrupt 碳数字产生前,按 **DISCOVERY 三窗池化总碳**冻结单一最强盲臂,
写 freeze artifact;CONFIRMATION 不重选。

## 4. 误差臂(§5.2)

    clean           godeye(oracle144 视图,lead-0 保真)
    主误差          empirically calibrated shrink:参数(逐 DC 振幅增益 λ 的 lead 曲线、
                    加性偏置 b、荒时乐观分量、残差 AR 与跨 DC 相关)一律取自
                    timecap_error_audit.json 的 primary_error_params 字段,机械代入,
                    不得按调度碳挑选;审计产物落库后以 append-only addendum 冻结数值
    次级误差        optimistic hallucination(按审计的假峰率标定),只作次级,
                    主次顺序现在写死,不得按结果择优
    描述性对照      shuffle / anti / s30(不设"必须伤害"判据)

前提:审计三问中"λ<1 回归均值"为 yes 才启用该主误差;若审计推翻(λ≈1 且无系统偏置),
主误差另立 addendum 重报 Codex,不得自行改选。

## 5. 零训练主门(§5.3 逐字,全部满足才准 EU-CRD)

    1  clean 相对冻结最强盲臂,总碳降 ≥5%
    2  主误差相对 clean,总碳增 ≥5% 或回吐 clean 收益 ≥50%
    3  至少 2/3 DISCOVERY 窗方向一致
    4  主误差必须改变等待或路由动作(动作全等即失败)
    5  completion / SLA / cap / 容量 / 台账合同全绿
    6  在未读 CONFIRMATION 上按相同门重复通过
    7  CONFIRMATION 不重选误差、参数、场景或盲臂

统计口径:逐格三窗聚合碳强度(Σcarbon/Σcompleted_MI),格间取中位;门 1/2 以中位判,
门 3 以逐窗池化方向判。失败记 **STOP_NO_LOAD_BEARING_FORECAST_ERROR**,不跑 RL,
不继续搜索更疼的污染。

## 6. 通过后(§6 预告)

Stage D 五臂(no-forecast / Vanilla clean / EU-CRD clean / Vanilla corrupt / EU-CRD
corrupt)另立 50k 预注册;健康门与三配对种子长训条款按工单 §6/§7,措辞按 §7 冻结版。

## 7. 执行阶梯

    3060 误差审计(timecap_error_audit.json)落库
    → addendum 冻结主误差数值参数
    → 实现两新盲臂 + 主/次误差臂,测试全绿提交
    → DISCOVERY:盲族四臂 → 池化冻结最强盲 → clean + 主/次误差 + 对照
    → 门 1–5 判读(判读器先于数据冻结提交)
    → 全过 → 一次性 CONFIRMATION → 门 6/7
    → 双过 → Stage D 预注册

## Addendum A(append-only,主误差数值冻结,先于任何 E 碳数字)

参数来源:`timecap_error_audit.json`(file sha 475793a6c036c801,commit 480783f),整块机械消费。

**主误差 `calibrated_shrink_v1` 生成式**(全部逐 DC 逐 lead,lead-0 保真,截零):

    view_d[t+l] = max(0, mu'_d + lambda_d(l)·(truth_d[t+l] − mu'_d) + b'_d(l) + sigma'_d(l)·eps_d[t+l])
    view_d[t]   = truth_d[t]

    lambda_d(l)   审计的 144 点 lead 曲线(近场 ~0.88,lead 11 ~0.45,lead 47+ ~0.1),无量纲直用
    截距          b_ols(裁定采纳;mean-bias 在 lambda<1 时重复计入偏置,弃用)
    残差          eps 为冻结 AR(1) 场,rho_d = 审计逐 DC 值(~0.9986-0.9991,持久漂移),
                  跨 DC 单因子 c = 0.8646(审计中位非对角;与 ladder-v2 独立测得的 0.8601 互证)
    迁移归一      审计量纲量(b、sigma)按 mu 比例迁移到新风机:
                  scale_d = mean(新 DC 全序列真值) / mu_audit_d;lambda 无量纲不迁移;
                  DC 映射按序号(审计 DC0/1/2 → E DC0/1/2)

**按审计裁定不进入污染器的项**(记录,非按碳挑选):相位偏移(中位 +3~+5.5 行但 IQR ±35、
25-27% 锚点顶边界,系相关成分缺失下的噪声 argmax,不构成参数);假峰项(命中/漏报/假峰率
全部在随机基准 ±0.02 内,峰值信号缺失而非系统性造假,①的强收缩已内含此效果)。
**次级误差 optimistic hallucination 依据审计 Q2 撤销**——荒时乐观(P=81-86%)已由
lambda 曲线 + b_ols 生成式内生产生,单列假峰项无实证依据。

**工单判据偏离记录(待 Codex 终审)**:Q3 空间排名的参照由"均匀随机 1/3"改为
"常数-mu 预报器"(0.564 vs 0.714,判 yes 主动损坏);理由:三站 mu 差两倍时,
均匀随机参照会把静态知识误报为天气知识。
