# Project Rules

## Testing
- 任何新实现的代码必须配对应的测试(单元测试或集成测试)。
- 提交/收尾前必须跑测试套件,所有测试通过才算完成。
- 修 bug 时,先写一个能复现 bug 的失败测试,再改代码让它通过。
- 如果某段新代码确实无法测试(如纯 IO 入口、第三方框架钩子),在回复里明确说明原因,不要默默跳过。

## Scheduling Algorithm Evaluation
- 新实现的调度算法在做 evaluation 时,除了效果指标(carbon、SLA、utilization 等),必须配套做**效率/开销测试**,量化算法本身引入的 overhead。
- 至少覆盖:每次调度决策的耗时(平均/p95/p99)、整次仿真的 wall-clock 时间、相对 baseline(如 FCFS/Round-Robin)的相对开销。
- 涉及推理的算法(RL policy、ML 预测器等)还要单独报告推理延迟与内存占用。
- 结果要和效果指标一起出在同一份评测报告里,不能只报效果不报代价。
