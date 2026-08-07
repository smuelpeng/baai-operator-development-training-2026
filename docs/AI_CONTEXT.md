# AI Coding 项目上下文

这是一份可直接提供给编码助手的紧凑背景。更详细的教学解释在各模块 README 和课件中。

## 项目目标

仓库覆盖算子开发的完整链路：

```text
算子语义 → Triton kernel → 框架注册/自动微分
        → 编译 lowering → 多卡执行 → 性能评测与诊断
```

目标使用者能够借助 AI Coding 快速完成新算子、融合算子、调优、编译链分析和性能诊断任务。

## 目标平台

- 硬件：天数智芯 BI-V150，架构标识 ivcore11；
- 软件：FlagOS / CoreX 适配的 PyTorch 与 Triton；
- 设备接口：`cuda:0`、`torch.cuda`；
- 课程环境口径：warp size 64；
- 性能工具：`ixsmi`、`ixsys`、`ixkn-cli`；
- 典型 GEMM：fp16 输入/输出、fp32 accumulator。

兼容接口不等于 NVIDIA 设备。平台相关配置和性能必须在 BI-V150 上验证。
FlagOS 实例配额、CoreX 环境指纹和低风险上机顺序见
[`FLAGOS_ONLINE_LAB.md`](FLAGOS_ONLINE_LAB.md)。CoreX 4.4 的版本字符串可能不含
`+corex`，核验时还要读取包路径、运行时路径、设备属性并执行最小 kernel。

## 最短代码地图

| 需要 | 文件 |
|---|---|
| 最小 Triton grid/offset/mask | `modules/01-ai-system-and-triton/labs/triton-basics/day1/02_vector_add_triton.py` |
| Softmax/LayerNorm/GEMM | 同目录 `03_*`、`04_*`、`05_*` |
| GEMM kernel、正确性、计时 helper | `modules/01-ai-system-and-triton/labs/gemm-tuning/common.py` |
| block sweep | `modules/01-ai-system-and-triton/labs/gemm-tuning/01_block_sweep.py` |
| Triton autotune | `modules/01-ai-system-and-triton/labs/gemm-tuning/02_autotune.py` |
| 融合算子学生任务 | `modules/02-high-performance-operators/labs/fused-rmsnorm-student/rmsnorm_fusion_lab/` |
| 融合算子参考 | `modules/02-high-performance-operators/labs/fused-rmsnorm-reference/` |
| Dispatcher / Meta / Autograd | 学生任务中的 `custom_op.py`、`grad_kernel.py` |
| 编译链练习 | `modules/03-ai-compiler/exercises/01-trace-a-kernel.md` |
| AllReduce 推演 | `modules/04-distributed-training-and-communication/exercises/01-allreduce-and-parallelism.md` |
| Benchmark/Roofline/Profiler | `modules/05-performance-engineering/labs/performance-analysis/` |
| 插桩、Debugger、Profiler | `modules/05-performance-engineering/tools/FlagPrism/` |

## 默认工程口径

- 参考函数优先使用直观 PyTorch 表达，归约精度写清楚；
- kernel wrapper 负责输入合同、contiguous、输出分配和 launch；
- 尾块必须使用 mask，`tl.load` 的 `other` 要符合归约单位元；
- 正确性检查先于 benchmark；
- benchmark 排除首次 JIT，使用同步和多样本统计；
- 优化结果保留失败候选，避免只记录 winner；
- 结论分为“本机实测”“由证据推断”“待目标平台验证”。

## 当前事实边界

- 课程源码、课件和来源已经归档并校验；
- 本仓库所在 Mac 不具备已确认的 BI-V150/CoreX 环境；
- 2026-08-08 已在 FlagOS 一卡 BI-V150 实例动态验证模块 1、2、5 的代表性链路，命令、固定提交和结果见 [`records/experiments/2026-08-08-flagos-bi-v150-validation.md`](../records/experiments/2026-08-08-flagos-bi-v150-validation.md)；
- 模块 3、4 和双卡路径仍未动态验收，课件中的性能目标不能当作现有复现结果；
- 新任务应在 `workspaces/` 开发，避免覆盖固定上游快照。

## 给 AI 的最小请求格式

```text
阅读 AGENTS.md、docs/AI_CONTEXT.md 和 <任务目录>/TASK.md。
请先复述算子合同和验收矩阵，再实现参考函数与正确性测试。
正确性通过后再写 Triton kernel；没有目标硬件时保留待运行命令，禁止虚构性能结果。
所有实验结果写入 <任务目录>/results/，结论写入 REPORT.md。
```
