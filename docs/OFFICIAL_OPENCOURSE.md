# 官方 OpenCourse 对照与使用指南

本页把本仓库与 FlagOS 官方 [OpenCourse](https://github.com/flagos-ai/OpenCourse) 对齐。对照版本固定为
`cb6b9e0a3c01d9cd31bc2187127a6f44f6626990`（2026-08-10），后续阅读官方材料时应同时记录新旧提交，不能用滚动的 `main` 代替版本号。

## 三类资源如何使用

| 官方目录 | 定位 | 本仓处理方式 |
|---|---|---|
| [`course/`](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/course) | 持续迭代的 48 学时高校课程主线 | 建立固定提交索引，不整体复制约 1.86 GiB 的课程主体 |
| [`editions/china-2026-summer-faculty/`](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/editions/china-2026-summer-faculty) | 本次 2026 暑期教师研修班发布版 | 24 项逐项对账；同哈希文件复用，缺失原件补档 |
| [`best-practices/`](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/best-practices) | FlagGems、FlagTree、FlagScale、FlagCX 组件专题 | 作为模块延伸阅读，不把性能陈述写成实测结论 |

完整上游树有 503 个文件、约 3.04 GiB。逐文件 Git 对象和大小见
[`official_opencourse_tree.tsv`](../records/official_opencourse_tree.tsv)，本期发布版的 SHA-256、去重和落盘状态见
[`official_opencourse_2026_edition.tsv`](../records/official_opencourse_2026_edition.tsv)。

## 本次研修班发布版对账

官方发布版共 24 项。本仓原有 12 份 PDF 与官方文件 SHA-256 完全相同；本次补入 6 份 PPTX、1 份官方平台 PDF 导出版本和 5 份实验入口说明。由此可逐项定位官方发布版全部 24 项，同时避免重复保存 12 份同内容 PDF。

可编辑课件放在各模块的 `official-editable/` 目录。PPTX 是上游原件，PDF 是群内授课/预习版本；二者都保留，不互相覆盖。

有两项必须单独说明：

- 官方平台 PDF 与群内 `线上实验室平台介绍0804.pdf` 都是 16 页，正文高度一致，但文件哈希、导出软件和压缩方式不同，故作为两个版本保存。
- 官方路径中的 `课件-模块四：分布式并行训练.pptx` 实际 34 页全部为 **AI Compiler: Basics and Optimizations**。它与本地模块三授课版对应，不是分布式训练课件。本仓将原文件隔离到模块三的 `upstream-mislabeled/`，保留上游文件名和异常说明；模块四 61 页群内 PDF 不受影响。官方当前没有提供这份 61 页材料的可编辑原件。

## 48 学时课程主线

官方中文主线按五个模块组织 48 学时：模块 1–4 各 9 学时，模块 5 为 12 学时；课时 46–47 合并在一份 PPTX 中，因此中文课件是 47 个文件。它还提供 13 份作业、作业模板、课程大纲和飞书资源索引。

官方入口：[48 学时中文汇总](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/course/02-课件资源/中文版/完整课程汇总-48课时)、[中文课件、作业与大纲](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/course/02-课件资源/中文版/完整课程汇总-48课时/2-1-slides_zh)、[教师培训指南](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/course/05-教师培训指南)。

本仓继续把 5 天研修班作为默认快速路径；教师需要开设学期课时，再按下表扩展：

| 本仓模块 | 官方课时 | 官方作业 | 本仓主要实践 |
|---|---:|---|---|
| 模块 1：异构计算与 Triton | 01–09 | 1 概念理解、2 Triton 入门、3 内存优化 | Vector Add、Softmax、LayerNorm、GEMM |
| 模块 2：高性能算子 | 10–18 | 4 Attention、5 融合与 Profiling、6 算子库设计 | Fused Add + RMSNorm 全栈 |
| 模块 3：AI 编译器 | 19–27 | 7 IR 与 Pass、8 后端代码生成 | lowering 地图与官方 IR 实验候选 |
| 模块 4：分布式训练与通信 | 28–36 | 9 并行策略、10 通信与显存 | AllReduce 与 α-β 推演 |
| 模块 5：性能工程 | 37–48 | 11 性能分析、12 AI 内核生成、13 全栈优化 | Benchmark、Roofline、Profiler |

官方课程大纲提出 3 学分、32 学时理论与 16 学时实践，成绩建议为实验 40%、课程项目 30%、期末考试 30%。这是官方长期课程设计，不是本次 5 天研修班已经执行的考核结果。

## 实验来源与状态

### 本期发布版

| 模块 | 官方入口 | 本仓状态 |
|---|---|---|
| 1 | [`sunxt-0719/lab1`](https://gitee.com/sunxt-0719/lab1) | 已固定为 `1969ed2741c5dbadf8961c85d827e7d05e3f0c30` |
| 2 | [`sunxt-0719/completed_lab2`](https://gitee.com/sunxt-0719/completed_lab2) | 参考版已固定为 `3ba8e2d6d94556ba45be9ce15c8751d44986affe`；本仓另保留学生版 |
| 3 | [`nsddrezsd/lab3`](https://gitee.com/nsddrezsd/lab3) | 核心源码快照固定为 `5334a42b986c103ba617f074f1a1b4cba00c610d`，尚未在本仓目标镜像运行 |
| 4 推理 | [`nsddrezsd/lab4-1`](https://gitee.com/nsddrezsd/lab4-1) | 手册与图片固定为 `78288624113cfe6ae23025918673adda5db8f3a5`，属于推理部署扩展，不是集合通信实验 |
| 4 分布式训练 | 飞书 Wiki `UnFzwvIYUi3jC6kqtKdc8y8Mnad` | 2026-08-10 未登录访问进入飞书登录流程，未取得正文 |
| 5 | [`yihan-long/day2-lab5`](https://gitee.com/yihan-long/day2-lab5) | 已固定为 `5fb8557f19fffb14f2c0316f0b26fe4fa026a52f` |

模块三上游仓库同时包含 NVIDIA H200 README、BI-V150 README 和生成产物。当前提交里的 `run.json` 实际记录 NVIDIA H200、warp size 32 和 Triton 3.6.0，与 BI-V150 README 的文字不一致。本仓只保存核心源码，排除缓存、二进制和这些相互冲突的结果；运行前先读本地包装说明。

### 官方长期课程的 BI-V150 Day 1–3

官方主线还指向另一套 BI-V150 实验：

| 实验 | 固定提交 | 处理决定 |
|---|---|---|
| Day 1 基础 | `3986e25ac089c05f342ec2e6b3c9ace9baefe049` | 仅索引；与本仓模块 1 重复，MatMul 尾块 mask 维度需复核 |
| Day 2 调优 | `986bd2a02f8dc45f598e390d794a62d905fd5b2e` | 仅索引；文档仍按 32 线程解释 `num_warps`，与本仓实测 warp size 64 冲突 |
| Day 3 推理 | `2271b31d907c9d3c6fe500468dd9544bd946007a` | 仅索引；属于 vLLM + FlagGems 部署扩展 |

这些仓库没有覆盖本地已验证快照，也没有被写成“当前镜像已通过”。

## 组件专题导航

| 官方专题 | 适合接入 | 使用边界 |
|---|---|---|
| [FlagGems](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/best-practices/FlagGems) | 模块 1、2 | 算子编写、Reduce/Scan/Sort、GPU/Triton 背景 |
| [FlagTree](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/best-practices/FlagTree) | 模块 3 | 编译器设计、新芯片后端接入；不等同于 FlagPrism 源码更新 |
| [FlagScale](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/best-practices/FlagScale) | 模块 4 | 分布式训练、推理与多后端案例；本仓没有其可执行快照 |
| [FlagCX](https://github.com/flagos-ai/OpenCourse/tree/cb6b9e0a3c01d9cd31bc2187127a6f44f6626990/best-practices/FlagCX) | 模块 4 | 集合通信、跨芯通信和拓扑优化案例 |

## 来源优先级与安全边界

1. 当前任务的 `TASK.md`、本仓动态实验记录和目标实例指纹优先于通用课程文案。
2. 官方指南中的 Python、PyTorch、Triton、CoreX 版本和资源配额是通用建议，不能覆盖本仓在研修班实例上的实测值。
3. 不运行会用公开 wheel 覆盖 CoreX 适配 `torch` 或 `triton` 的安装命令。上游 README 作为原始快照保留，不代表获得执行许可。
4. 官方仓只有飞书资源索引，没有视频二进制；不能写成“课程视频已下载”。
5. 本仓采用官方中文 48 学时汇总的 `9/9/9/9/12` 作为长期课程映射；其他版本的目录结构不能直接替代这份课时表。
6. OpenCourse 材料采用 CC BY-NC 4.0。本仓保存了许可证原文；对外授课、改编或分发时须署名、标明来源与修改，并遵守非商业限制。外链 Gitee 仓库未发现独立许可证时，只按内部固定快照管理，不扩张其授权范围。
