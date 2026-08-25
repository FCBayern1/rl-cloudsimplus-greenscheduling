# TB12 50k smoke 四门判定 — G3 FAIL → STOP(按冻结纪律,不进 300k)

日期:2026-08-25 21:27。执行:tb12_smoke_run.sh(哨兵 33089d1d PASS)→
fc→nofc 串行 → tb12_smoke_gate.py 机械判定。
数据:`local_eval_rt/audit/tb12_smoke_gate_s1.json`。ck0 训练前固化成功
(注:algorithm.logdir 落在 ray working_dir,已复制持久化——接线笔记见文末)。
**按纪律:50k 不作任何效果声明。**

## 四门结果

| 门 | 结果 | 关键读数 |
|---|---|---|
| G1 奖励—物理 | PASS | 两臂 reward 均未改善(fc −208→−215,nofc −208→−208),蕴含式空真;**cap 命中全零(24 集)** |
| G2 SLA | PASS | fc/nofc 池化 ontime_mi_share = **1.0**(backstop per-PE 修复生效) |
| G3 坍缩 | **FAIL** | fc_ck50 全 6 偏移:forced=5、active=0、defer_frac=1.000 |
| G4 信息活性 | PASS | fc vs nofc 逐偏移 kg 差 5/6 非零 |
| **ALL** | **STOP** | |

## 机械事实(不作效果解读)

- **fc 的 argmax 在 50k 仍是全 defer**(全部作业由 backstop 强制释放);
  nofc 的 argmax 是全立即释放(defer_frac=0.000),与 ck0 初始化行为逐位相同
  (reward/kg 完全一致 −208.012/1.124)。
- 与 v1 事故的三点不同(仅记录,不解读):
  1. cap 命中 0/24 集——新分母下奖励通道诚实;
  2. ontime 全 1.0——坍缩不再违约(backstop 修复);
  3. fc 的全 defer 在新奖励下**被惩罚而非奖励**
     (reward −215 差于 ck0 的 −208,kg 1.233 高于 1.124)——
     v1 中同样行为是 reward 上升。
- ck0 行为基线:初始化 argmax = 全立即释放(两臂同一 ck0 指纹)。

## 处置

按 Codex 签发的阶梯纪律:四门任一失败 → **立即 STOP**。不启动 300k,
不调参重跑,T116+117 保持封存。是否修订 G3 在 50k 处的判定位置
(50k 全 defer 是终态还是暂态,是否以 300k 为坍缩判定点)属于阶梯定义
变更,**只能由 Codex 裁定**;在裁定前机器保持空闲。

## 接线笔记(下轮修正项,非本轮判定因素)

`on_algorithm_init` 的 ck0 经 `algorithm.logdir` 落在 `/tmp/ray/session_*/
.../working_dirs/<trial>/checkpoint_ck0`(volatile),本轮由 after 链复制到
trial 持久目录。下轮应把保存路径直接指向 storage_path(或训练后立即持久化),
避免依赖 /tmp/ray 存续。
