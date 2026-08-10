# 算子开发知识索引

遇到问题时先走最短路径，再按需要展开课件。

## 建立课程全景

| 问题 | 先读 | 下一步 |
|---|---|---|
| 如何建立从 AI 芯片到 FlagOS 的全栈认识？ | [课程地图](COURSE_MAP.md) 的王晶课件页段表 | 按表进入模块 1–5 |
| 如何把实验组织成竞赛式课程项目？ | [教师授课指南](TEACHING_GUIDE.md) 的“全景导入与赛教融合” | 再读 [作业与验收](ASSESSMENT.md) |

## 写 kernel

| 问题 | 先读代码 | 再读知识 |
|---|---|---|
| 如何计算 program id 和 offsets | `triton-basics/day1/02_vector_add_triton.py` | 模块 1：grid/program/tile |
| 尾块 mask 与归约单位元 | `03_softmax_compare.py`、`04_layernorm_compare.py` | 模块 1：mask、稳定归约 |
| 二维 tile 与 stride | `05_matmul_compare.py`、`gemm-tuning/common.py` | 模块 1：tiled GEMM |
| fp16/bf16 的归约精度 | Fused RMSNorm 的 `fused_rms_norm.py` | 模块 2：RMSNorm 与融合 |
| backward kernel | Fused RMSNorm 的 `grad_kernel.py` | 模块 2：反向与原子累加 |

## 接入 PyTorch

| 问题 | 代码入口 | 验收 |
|---|---|---|
| 自定义算子 schema | `fused-rmsnorm-student/.../custom_op.py` | `torch.ops` 可调用 |
| CUDA/AutogradCUDA/Meta | 同上 | eager、backward、FakeTensor 均有实现 |
| `torch.compile` graph break | `compile_demo.py` | 图数量与 break 数明确 |
| 参考与融合实现对照 | `benchmark.py` | 相同 shape、dtype、计时边界 |

## 调优与性能

| 问题 | 代码入口 | 需要保存 |
|---|---|---|
| 可信计时 | `gemm-tuning/common.py::bench_ms` | warmup、samples、inner、同步 |
| 控制变量 sweep | `gemm-tuning/01_block_sweep.py` | 全部 PASS/FAIL 与错误摘要 |
| Triton autotune | `gemm-tuning/02_autotune.py` | 候选来源、winner、复核结果 |
| Offline/Server workload | `performance-analysis/01_benchmark.py` | median、P99、吞吐、TFLOP/s |
| Roofline | `performance-analysis/02_roofline.py` | FLOPs、Bytes、峰值、带宽、ridge point |
| 平台 profiling | `performance-analysis/03_profiling.py` | 命令、trace、occupancy/memory 证据 |

## 编译与调试

| 问题 | 入口 |
|---|---|
| Python → TTIR → TTGIR → LLVM/目标代码 | `modules/03-ai-compiler/README.md` |
| 如何做 lowering 对照 | `modules/03-ai-compiler/exercises/01-trace-a-kernel.md` |
| 数值摘要、地址、完整值、时间线插桩 | `FlagPrism/Debugger/` |
| timeline、Hatchet、vendor 数据 | `FlagPrism/Profiler/` |

## 分布式算子与通信

| 问题 | 入口 |
|---|---|
| DP/TP/PP/ZeRO/FSDP 的选择 | `modules/04-distributed-training-and-communication/README.md` |
| Ring/Tree/halving-doubling | 同模块课件与练习 |
| α-β 成本和拓扑假设 | `exercises/01-allreduce-and-parallelism.md` |

## 原始课件

需要引用教师原始定义、公式或架构图时，从 `materials/README.md` 进入课件。代码任务优先引用模块 README 和源码路径，避免让 AI 一次加载全部 PDF。
