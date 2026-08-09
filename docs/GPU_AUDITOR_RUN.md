# GPU 简报:auditor 8 格实验(2026-08-09,加急)

## 一句话
跑 `local_eval_rt/run_auditor_gpu.sh`(串行 8 格,~7h),结果落在
`local_eval_rt/auditor_gpu.txt`,逐格追加,核心 2×2 前四格 ~3.5h 先出。

## 步骤
```bash
git pull                       # 拿到本简报 + run_auditor_gpu.sh
cd cloudsimplus-gateway && ./gradlew installDist -q && cd ..   # 保险起见重建
nohup bash local_eval_rt/run_auditor_gpu.sh >/dev/null 2>&1 &
```
脚本自己会从 `ckpt-transfer` 分支解包 checkpoint 对(首次 ~42MB),缺 ckpt 会
写明并退出,不会带病硬跑。

## 是什么实验
C-regime seed-3 匹配对(van ck10 / eucrd_v4 ck10),argmax,ep10,
反相污染(FORECAST_PERTURB_MODE=anti)下测运行时审计器:
{off, gate, repair} × {van, eucrd} + 两格 clean 对照。
审计器 = χ 滚动相关(**必须 TRUST_GATE_SOURCE=resid**,默认 qvar 需要
Q-ensemble,van 的 ckpt 没有——脚本已内置,不要改)。gate 阈值 0.2。

## 判读
- 主格子:anti 下 gate 相对 off 是否把碳/完成率拉回 clean 水平附近
  (audit 生效证据,论文附录 H 的表)。
- repair 是加分项,坏了不影响主结论。
- 8 格全在你机器上,内部自锚(有 clean 对照)——**不要**和我们本地机的
  历史数字对表,跨机噪声底 16%。
- auditor-lines 是日志里 gate/repair 触发行计数,>0 说明审计器真的动了;
  clean-off 与 anti-off 两格应为 0。

## 跑完
把 `auditor_gpu.txt` 全文发回来即可,per-cell 日志(audgpu_*.log)先留着别删。
