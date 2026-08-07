# 算子开发任务目录

这份目录让开发者和 AI Coding 助手直接选择任务，减少从大量材料中自行拆题的时间。

| ID | 难度 | 任务 | 起点 | 完成证据 |
|---|---|---|---|---|
| T01 | 入门 | 补全 Vector Add Triton kernel | `triton-basics/day1/02_vector_add_triton.py` | 1、127、128、129、4097 长度全部 PASS |
| T02 | 入门 | 为 Softmax/LayerNorm 增加非整 tile 与极值测试 | `triton-basics/day1/03_*`、`04_*` | 稳定归约、NaN/Inf、误差表 |
| T03 | 进阶 | GEMM block sweep 与 autotune | `labs/gemm-tuning/` | 全候选 PASS/FAIL、winner 复核、报告 |
| T04 | 进阶 | Fused Add + RMSNorm 前向 | 模块 2 学生版 `fused_rms_norm.py` | fp32/bf16、多 shape 正确性 |
| T05 | 进阶 | RMSNorm backward、Dispatcher、Meta | 学生版 `grad_kernel.py`、`custom_op.py` | 梯度误差、eager、compile graph 结果 |
| T06 | 进阶 | 新建 Bias + Activation 融合算子 | `make new-task NAME=fused-bias-activation` | 独立合同、reference、kernel、测试、bench |
| T07 | 高阶 | 追踪一个 kernel 的 TTIR/TTGIR | 模块 3 编译链练习 | 四层 lowering 地图与三处 pass 解释 |
| T08 | 高阶 | 设计 AllReduce 成本计算器 | 模块 4 通信练习 | Ring/Tree 成本、参数假设、算法选择 |
| T09 | 高阶 | 对 GEMM 做 Benchmark→Roofline→Profiler | 模块 5 `performance-analysis/` | JSON、Roofline、trace、诊断报告 |
| T10 | 研究 | 为 FlagPrism 增加一种观测或后端适配 | `FlagPrism/Debugger/` 或 `Profiler/` | 最小设计、单测、artifact 与限制 |

## 最快上手路线

### 只有半天

完成 T01，并让 AI 解释每个 offset、mask 和边界 shape。随后运行 `make new-task`，观察同一开发合同如何迁移到独立任务。

### 两天

按 T01 → T03 → T09 推进。开发者会经历“写正确—调参数—用证据解释性能”的完整闭环。

### 一周

按 T01 → T03 → T04 → T05 → T07 → T09 推进。最后用 T06 独立实现一个新融合算子，作为结课任务。

## 分配任务给 AI

每次只选择一个 ID，并给出允许修改路径。例如：

```text
执行 T04。允许修改 modules/02-high-performance-operators/labs/
fused-rmsnorm-student/rmsnorm_fusion_lab/ 下的 TODO 文件。
先阅读 AGENTS.md 和模块 2 README；先给正确性矩阵，再实现前向。
当前没有 BI-V150，目标平台运行项保留为待验证。
```

对实际项目，建议用脚手架创建 `workspaces/<name>/`，再把任务合同和验收门槛写入该目录，避免修改课程样例。
