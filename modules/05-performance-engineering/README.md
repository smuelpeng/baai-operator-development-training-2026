# 模块 5：性能评测与下一代内核生成

## 核心问题

如何证明一次优化真的有效，并判断瓶颈位于计算、内存、调度、通信还是端到端路径？

## 诊断闭环

```text
Benchmark：速度差多少？
       ↓
Roofline：计算上限还是带宽上限？
       ↓
Profiler：哪类设备事件支持或推翻这个判断？
       ↓
调优 / KernelGen / 融合：提出改动
       ↓
回到 correctness 与同口径 Benchmark
```

## 学习重点

- FLOPS、带宽、延迟、吞吐、利用率与 MLPerf 场景；
- Roofline：`P ≤ min(π, βI)`、ridge point 与模型边界；
- BI-V150 工具：`ixsmi`、`ixsys`、`ixkn-cli`；
- occupancy、warp state、instruction、memory 等 kernel 证据；
- tiling、unrolling、vectorization、binding 与自动搜索；
- Amdahl 定律和局部优化到端到端收益的转换；
- FlagPrism 如何连接源语句、IR、设备值、地址与时间线。

## 课件、实验与工具

- [模块五主课件](../../materials/courseware/module-05/0807-课件-预习版本-模块五：性能评测与下一代内核生成.pdf)
- [Profiler & Debugger 专题](../../materials/courseware/module-05/基于%20FlagTree%20生态的%20Profiler&Debugger%20初探%20-%20周炽金%20老师.pdf)
- [性能分析实验](labs/performance-analysis/)
- [FlagPrism 固定快照](tools/FlagPrism/)

实验以模块 1 的 Day 2 GEMM 作为 SUT。完成环境检查后，依次运行 Benchmark、Roofline、Profiling 和 summary；原始 JSON、trace 与报告必须同时保存。

从仓库根目录运行：

```bash
cd modules/05-performance-engineering/labs/performance-analysis
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py
export LAB_DAY2_ROOT="$PWD/../../../01-ai-system-and-triton/labs/gemm-tuning"
python3 01_benchmark.py --scenario offline
python3 01_benchmark.py --scenario server
python3 02_roofline.py
python3 03_profiling.py
python3 04_summary.py
```

## 验收

报告必须回答：性能差距有多大；Roofline 的瓶颈分类是什么；profiler 的哪一项证据支持或修正该分类；下一项改动是什么；改动后如何保持相同口径复测。

FlagPrism 处于持续开发状态，本仓库保存的是固定提交快照。课件路线图属于计划信息，不能当作全部后端能力已经交付。
