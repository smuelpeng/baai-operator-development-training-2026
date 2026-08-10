# 从 Triton 算子到系统性能：2026 智源智算研修班学习仓库

这是一套面向高校教师和学生的算子开发课程。主线从一个 Triton kernel 出发，经过算子融合、AI 编译器、分布式通信，最终落到 Benchmark、Roofline 与 Profiler 的性能证据。

课程原始资料来自“2026 年暑期全国高校人工智能自主软硬件技术课程公益研修班（第一期）”。本仓库在保留课件和代码快照的基础上，补充了教学顺序、学习目标、实验任务、产出要求和验收标准。

> 使用范围：仓库包含内部课件、通知和邀请函，必须保持 Private。转发、授课和二次使用前请遵守主办方及原作者的授权要求。

## 用 AI Coding 开始第一个算子任务

仓库已经内置编码助手需要的项目上下文、任务路由、开发规则、验证脚本和 Triton 任务脚手架。

```bash
# 1. 查看 AI 必读入口
make context

# 2. 创建独立算子工作区
make new-task NAME=fused-silu TITLE="Fused SiLU"

# 3. 填写算子合同
open workspaces/fused-silu/TASK.md

# 4. 做不依赖加速卡的仓库检查
make validate
```

然后向 AI Coding 助手发送：

```text
阅读 AGENTS.md、docs/AI_CONTEXT.md、docs/KNOWLEDGE_INDEX.md，
以及 workspaces/fused-silu/TASK.md。
先复述算子合同、正确性矩阵和最小 kernel 设计，再开始修改代码。
正确性通过前不要调优；缺少 BI-V150 时不要生成虚构性能结果。
```

详细流程见 [AI 辅助算子开发手册](docs/OPERATOR_DEVELOPMENT_PLAYBOOK.md)。

## 一条完整的学习路径

```text
异构计算与 Triton
        ↓ 写出正确 kernel
高性能算子与融合
        ↓ 理解调度、注册和融合收益
AI 编译器
        ↓ 看懂 Python → TTIR → TTGIR → LLVM/设备代码
分布式训练与通信
        ↓ 从单卡算子进入多卡系统
性能评测与诊断
        ↓ 用 Benchmark → Roofline → Profiler 形成证据闭环
```

## 五个模块

| 模块 | 核心问题 | 实践入口 | 学习产出 |
|---|---|---|---|
| [01 AI 系统软件与 Triton](modules/01-ai-system-and-triton/) | kernel 如何映射到异构硬件？ | Vector Add、Softmax、LayerNorm、GEMM、Autotune | 正确性记录与 GEMM 调优报告 |
| [02 高性能 AI 算子](modules/02-high-performance-operators/) | 融合为什么会快，什么时候会变慢？ | Fused Add + RMSNorm 前向、反向、Dispatcher | 学生实现、测试结果、性能对照 |
| [03 AI 编译器](modules/03-ai-compiler/) | 一段 Triton 代码如何变成设备指令？ | 编译链追踪、IR 对照阅读 | 一份算子 lowering 地图 |
| [04 分布式训练与通信](modules/04-distributed-training-and-communication/) | 多卡训练的显存、通信与流水如何权衡？ | ZeRO/并行策略分析、AllReduce 推演 | 通信成本表与策略说明 |
| [05 性能工程](modules/05-performance-engineering/) | 如何证明优化有效并定位瓶颈？ | Benchmark、Roofline、Profiling、FlagPrism | 可复核的性能诊断报告 |

## 建议用法

学生按模块顺序学习。每个模块先读模块 README 中的“进入条件”和“概念地图”，再完成课件阅读、实验或推演任务，最后提交列出的产出物。教师可直接使用 [教师授课指南](docs/TEACHING_GUIDE.md) 组织 5 天集中课或 10 次常规课。

```bash
git clone git@github.com:smuelpeng/baai-operator-development-training-2026.git
cd baai-operator-development-training-2026
```

实验依赖 BI-V150、CoreX 适配的 PyTorch/Triton 及 FlagOS 工具。请先阅读 [环境说明](docs/ENVIRONMENT.md)，不要用公开版 `torch` 或 `triton` wheel 覆盖平台环境。

## 教学导航

- [AI Coding 项目上下文](docs/AI_CONTEXT.md)：可直接供编码助手读取的平台事实与代码地图。
- [算子开发知识索引](docs/KNOWLEDGE_INDEX.md)：按问题定位最短代码与课件路径。
- [AI 辅助算子开发手册](docs/OPERATOR_DEVELOPMENT_PLAYBOOK.md)：从任务合同到正确性、调优和交付。
- [AI Coding 任务配方](docs/AI_TASK_RECIPES.md)：可复制的新 kernel、融合、调优和诊断请求。
- [算子开发任务目录](docs/TASK_CATALOG.md)：从 T01 到 T10 的可直接分配任务与最快学习路线。
- [课程地图](docs/COURSE_MAP.md)：先修知识、模块关系、学习节奏，以及王晶老师 165 页课程建设总览与五模块映射。
- [教师授课指南](docs/TEACHING_GUIDE.md)：讲授重点、课堂问题、实验组织和易错点。
- [原班日程与复用课表](docs/SCHEDULE.md)：研修班原日程和可复用的 5 天安排。
- [环境说明](docs/ENVIRONMENT.md)：BI-V150、CoreX、FlagOS 和实验边界。
- [FlagOS 在线实验室指南](docs/FLAGOS_ONLINE_LAB.md)：实例配额、CoreX 环境指纹、上机顺序和数据保全。
- [BI-V150 动态验证报告](records/experiments/2026-08-08-flagos-bi-v150-validation.md)：模块 1、2、5 的命令、结果、失败记录和结论边界。
- [作业与验收](docs/ASSESSMENT.md)：每个模块的提交物和评分口径。
- [课件索引](materials/README.md)：主课件、专题材料、平台资料与行政文件。
- [课程内容总结](records/course_summary.md)：对全部材料的内容级总结。
- [课程建设知识主题](knowledge/course-construction/README.md)：新增王晶课件的原文证据、主线图、词法索引和待回答问题。

## 仓库结构

```text
.
├── AGENTS.md                # AI Coding 默认操作手册
├── Makefile                 # 新建任务与仓库验证入口
├── modules/                 # 五个可教学模块
│   ├── 01-ai-system-and-triton/
│   ├── 02-high-performance-operators/
│   ├── 03-ai-compiler/
│   ├── 04-distributed-training-and-communication/
│   └── 05-performance-engineering/
├── materials/               # 原始课件、专题材料、平台与行政文件
├── knowledge/               # 独立的可审计课程知识主题
├── docs/                    # 课程地图、授课指南、课表、环境和验收
├── templates/               # 可复制的 Triton 算子任务脚手架
├── workspaces/              # 新算子与调优任务的独立工作区
├── scripts/                 # 任务生成和无硬件静态检查
├── records/                 # 来源、版本、文件清单和 SHA-256
└── ai_open_source_components_research_report.md
```

五个 Gitee 教学仓库和 FlagPrism 均已转为普通目录。GitHub 页面可以直接浏览源码，不需要初始化 submodule。每份源码的上游地址和固定提交见 [代码来源清单](records/code_repository_inventory.md)。

## 学习记录的最低要求

一次可信实验需要同时保存：

- 源码 commit、设备型号和软件版本；
- shape、dtype、tile/config 与测试命令；
- correctness gate、误差、NaN/Inf 和失败配置；
- warmup、同步、样本数、median/P99 等统计口径；
- 基线与优化实现的公平对照；
- 结论对应的原始 JSON、trace 或 profiler 证据。

截至 2026-08-08，仓库除资料归档、课程化重组和静态检查外，已在 FlagOS 一卡 BI-V150 实例上动态验证模块 1、2、5 的代表性链路。模块 3、4 以及双卡分布式链路尚未动态验收；复现实验时应以验证报告记录的源码提交、shape 和测试口径为准，重新生成本实例结果。
