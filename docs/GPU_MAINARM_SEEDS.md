# GPU 简报:主臂补种子(2026-08-11)

## 一句话
跑 `local_eval_rt/run_mainarm_seeds_gpu.sh`,给论文主表补两对匹配种子(s5、s6),
结果落 `local_eval_rt/mainarm_seeds_gpu.txt`,逐格追加。
**时间(按 GPU 机实测):训练约 13h/次 × 4,eval 约 2.6h/格 × 12,合计约 3.5 天。**
只评最后一个 checkpoint——配对分析本就不做选点,评三个只服务于中位数。

## 步骤
```bash
git pull
cd cloudsimplus-gateway && ./gradlew installDist -q && cd ..
nohup bash local_eval_rt/run_mainarm_seeds_gpu.sh >/dev/null 2>&1 &
```

## 为什么这是当前第一优先
主表目前四个训练种子里只有 2-3 个产出满足完成率合同的 checkpoint,所以 clean 和
blend 两列的中位数只有 n=2-3,配对检验只有 n=2。补两对之后配对数到 4-5,
单侧符号检验才可能过 0.05(0.5^5 = 0.031)。这是外部预审和导师意见的第一条。

## 两个臂
- Vanilla:`experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecapV3`
- EU-CRD :`experiment_multi_5dc_carbon_v2_deferrable_gdpd_timecap_eucrd_knSV3b`

顺序是 van s5 → knSb s5 → van s6 → knSb s6,**第一对跑完(约 34 小时)就有一对
可用的配对数据**,不必等全部结束。

## 三条纪律(踩过的坑)
1. **配对必须同机**:一对种子的两个臂都要在你这台机器上训练+评测。我们报的是
   配对内差值,跨机噪声底 16%,混机会毁掉这个量。
2. **永远只跑一件事**:一次 eval 会拉起 8 个 JVM(checkpoint 带 6 个 worker 配置),
   两个 eval 并行会互相拖慢约 4 倍。脚本内部已串行,不要再手动并行别的任务。
3. **不要在训练进行时执行任何 pkill**:训练的 Java 网关同样匹配
   `MainMultiDC`,而 py4j 调用无超时,worker 假活时 Ray 不会自愈,训练会冻死。

## 判读与回传
输出行的格式和本地 `local_rt_summary.txt` 完全一致
(`[v3ht {arm}_s{seed} ck{N}@ARGMAX {cond}] cc=... completion=...`),
所以我这边的聚合脚本可以直接吃,不用转换。

跑完(或第一对跑完)把 `mainarm_seeds_gpu.txt` 发回来即可。

## 已知会看到的现象
个别 checkpoint 的完成率会低于 99.5%,这是这个配方已知的贪心解码过度延迟盆地
(论文 §4.2 有记录,约 1/4 种子)。不用处理,聚合时按 iso-eligibility 规则自动排除。
