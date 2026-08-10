# 代码仓库来源清单

更新时间：2026-08-10

课程直接使用 5 个 Gitee 仓库，并以 FlagPrism 作为模块 5 的工具阅读材料。所有仓库已移除内嵌 `.git`，以固定提交的普通源码目录保存，GitHub 页面可直接浏览。

| 课程位置 | 上游地址 | 固定版本 | 上游文件数 | 用途 |
|---|---|---|---:|---|
| `modules/01-ai-system-and-triton/labs/triton-basics/` | `https://gitee.com/sunxt-0719/lab1` | `main@1969ed2741c5dbadf8961c85d827e7d05e3f0c30` | 18 | Triton Day 1/Day 2 综合实验 |
| `modules/01-ai-system-and-triton/labs/gemm-tuning/` | `https://gitee.com/sunxt-0719/lab1-day2` | `main@65b4a17c87c552474467115c42d68844df016c1e` | 10 | GEMM Benchmark、tile sweep、autotune |
| `modules/02-high-performance-operators/labs/fused-rmsnorm-student/` | `https://gitee.com/sunxt-0719/lab2` | `main@53debfeec2442cd545371c05e75a7fb5a5abf7d4` | 11 | Fused Add + RMSNorm 学生版 |
| `modules/02-high-performance-operators/labs/fused-rmsnorm-reference/` | `https://gitee.com/sunxt-0719/completed_lab2` | `master@3ba8e2d6d94556ba45be9ce15c8751d44986affe` | 6 | Fused Add + RMSNorm 参考实现 |
| `modules/05-performance-engineering/labs/performance-analysis/` | `https://gitee.com/yihan-long/day2-lab5` | `main@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f` | 35 | Benchmark、Roofline、Profiling 与报告 |
| `modules/05-performance-engineering/tools/FlagPrism/` | `https://github.com/flagos-ai/FlagPrism` | `main@8d2647ba280aa66e328ff606f1ffb9a4e2962947` | 372 | FlagTree Debugger/Profiler |
| `modules/03-ai-compiler/labs/official-triton-ir/upstream/` | `https://gitee.com/nsddrezsd/lab3` | `master@5334a42b986c103ba617f074f1a1b4cba00c610d` | 7（精选自 175） | Triton 源码到 IR 的候选实验；排除冲突产物与缓存 |
| `modules/04-distributed-training-and-communication/labs/official-inference-deployment/upstream/` | `https://gitee.com/nsddrezsd/lab4-1` | `master@78288624113cfe6ae23025918673adda5db8f3a5` | 6 | Qwen3-4B + vLLM + FlagGems 推理部署扩展 |

## 五个 Gitee 仓库的关系

- `lab1` 覆盖 Day 1 与 Day 2；`lab1-day2` 是 Day 2 调优实验的独立发布版本。课程建议使用 `lab1/day1` 接 `lab1-day2`。
- `lab2` 保留学生 TODO；`completed_lab2` 用于教师验收和讲评。
- `day2-lab5` 把 Day 2 GEMM 当作 SUT，继续完成 Benchmark、Roofline 与 Profiling。
- FlagPrism 来自 GitHub，不计入“五个 Gitee 课程仓库”。

新增两个快照来自官方 OpenCourse 的 2026 教师版 handout，也不改变“群内五个 Gitee 课程仓库”的历史口径。模块三只选取可读源码，模块四推理材料属于扩展案例，不冒充集合通信实验。

## 官方 OpenCourse 对照

官方课程仓库固定为 `https://github.com/flagos-ai/OpenCourse@cb6b9e0a3c01d9cd31bc2187127a6f44f6626990`。本期 handout 引用的 `lab1`、`completed_lab2` 和 `day2-lab5` 已与本仓既有快照重复，因此无需再次导入；本仓额外保留的 `lab1-day2` 和学生版 `lab2` 仍是原研修班教学补充。

官方长期课程另指向三份 BI-V150 实验，本次只登记，不覆盖本地实现：

| 官方长期课程实验 | 固定版本 | 状态与原因 |
|---|---|---|
| `zhao-zhiqiang2023/lab-day1-tianshu-v150` | `3986e25ac089c05f342ec2e6b3c9ace9baefe049` | 与模块 1 重复；MatMul 尾块 mask 维度待复核 |
| `zhao-zhiqiang2023/lab-day2-tianshu-v150` | `986bd2a02f8dc45f598e390d794a62d905fd5b2e` | 32-thread `num_warps` 解释与本仓 BI-V150 warp size 64 证据冲突 |
| `zhao-zhiqiang2023/lab-day3-tianshu-v150` | `2271b31d907c9d3c6fe500468dd9544bd946007a` | vLLM 部署扩展，不是算子内核主线 |

OpenCourse 还内嵌 CANN/Ascend 平台的 Day 1–3 实验。它们属于独立平台变体，不能混入 CoreX 默认命令；本次没有复制。

## 归档说明

- 版本号来自各仓库默认分支的 HEAD；已有仓库在 2026-08-08 与远端重新核对。
- `lab2` 上游提交包含 5 个 `__pycache__/*.pyc` 文件。课程仓库排除了这些解释器缓存；另清理了少量行末空白，有效源码和 README 均保留。
- “已归档”只说明源码已保存，不代表在 BI-V150 环境运行通过。
- 上游 README 中列出的 Ascend 原材料、平台 Triton 分支和参考链接不重复计入五个课程仓库。
- `lab3` 上游的 BI-V150 README 与同提交的 H200 `run.json` 相互冲突；其 `requirements.txt` 还要求会覆盖平台栈的 `torch>=2.11`、`triton>=3.6`。本仓排除了 `artifacts/`、`lab-results/`、`.triton-cache`、`.pyc`、`.so`、设备二进制和幻灯片，只保存核心源码、依赖声明和两个原始 README，运行前必须先适配。
- `lab4-1` 没有单独许可证文件；本仓保持私有、保存来源与提交，不据此扩张转载授权。
