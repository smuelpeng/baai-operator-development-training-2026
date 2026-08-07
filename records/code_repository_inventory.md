# 代码仓库来源清单

更新时间：2026-08-08

课程直接使用 5 个 Gitee 仓库，并以 FlagPrism 作为模块 5 的工具阅读材料。所有仓库已移除内嵌 `.git`，以固定提交的普通源码目录保存，GitHub 页面可直接浏览。

| 课程位置 | 上游地址 | 固定版本 | 上游文件数 | 用途 |
|---|---|---|---:|---|
| `modules/01-ai-system-and-triton/labs/triton-basics/` | `https://gitee.com/sunxt-0719/lab1` | `main@1969ed2741c5dbadf8961c85d827e7d05e3f0c30` | 18 | Triton Day 1/Day 2 综合实验 |
| `modules/01-ai-system-and-triton/labs/gemm-tuning/` | `https://gitee.com/sunxt-0719/lab1-day2` | `main@65b4a17c87c552474467115c42d68844df016c1e` | 10 | GEMM Benchmark、tile sweep、autotune |
| `modules/02-high-performance-operators/labs/fused-rmsnorm-student/` | `https://gitee.com/sunxt-0719/lab2` | `main@53debfeec2442cd545371c05e75a7fb5a5abf7d4` | 11 | Fused Add + RMSNorm 学生版 |
| `modules/02-high-performance-operators/labs/fused-rmsnorm-reference/` | `https://gitee.com/sunxt-0719/completed_lab2` | `master@3ba8e2d6d94556ba45be9ce15c8751d44986affe` | 6 | Fused Add + RMSNorm 参考实现 |
| `modules/05-performance-engineering/labs/performance-analysis/` | `https://gitee.com/yihan-long/day2-lab5` | `main@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f` | 35 | Benchmark、Roofline、Profiling 与报告 |
| `modules/05-performance-engineering/tools/FlagPrism/` | `https://github.com/flagos-ai/FlagPrism` | `main@8d2647ba280aa66e328ff606f1ffb9a4e2962947` | 372 | FlagTree Debugger/Profiler |

## 五个 Gitee 仓库的关系

- `lab1` 覆盖 Day 1 与 Day 2；`lab1-day2` 是 Day 2 调优实验的独立发布版本。课程建议使用 `lab1/day1` 接 `lab1-day2`。
- `lab2` 保留学生 TODO；`completed_lab2` 用于教师验收和讲评。
- `day2-lab5` 把 Day 2 GEMM 当作 SUT，继续完成 Benchmark、Roofline 与 Profiling。
- FlagPrism 来自 GitHub，不计入“五个 Gitee 课程仓库”。

## 归档说明

- 版本号来自各仓库默认分支的 HEAD；已有仓库在 2026-08-08 与远端重新核对。
- `lab2` 上游提交包含 5 个 `__pycache__/*.pyc` 文件。课程仓库排除了这些解释器缓存；另清理了少量行末空白，有效源码和 README 均保留。
- “已归档”只说明源码已保存，不代表在 BI-V150 环境运行通过。
- 上游 README 中列出的 Ascend 原材料、平台 Triton 分支和参考链接不重复计入五个课程仓库。
