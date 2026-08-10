# 智能计算课程建设与 FlagOS 教学融合

- Status: `active`
- Stage: `S6`
- Goal: 从新增课件中提取可用于课程设计的目标、模块、实践方式、证据和边界，支持教师把算子开发、AI 编译、分布式训练与 FlagOS 实验组织成可授课路径。
- Corpus: `closed`
- Manifest revision: `1`
- Index revision: `idx-8a7986963cf4b979-2`
- Graph version: `v0.45.0`
- Registered sources: 1
- 默认学习入口：[`learning-map.md`](learning-map.md)（总图＋局部放大）
- 全量数字台账：[`measurements.md`](measurements.md)

## 主线结论

- **C-002** [accepted/verified_observation]: 该课件明确采用“应用驱动、全栈贯通”的课程组织原则。
  证据：`E-002`
  边界：课件未在该页定义课时、教材版本和验收指标
- **C-004** [accepted/verified_observation]: 课件把系统层优化打榜作为赛教融合方式。
  证据：`E-005`
  边界：未提供比赛题目、评分脚本、防作弊或可复现实验协议
- **C-006** [qualified/reported_claim]: 课件报告校内赛有104人报名，并被用作课程结课期末考核，且报告全部完赛。
  证据：`E-007`, `E-008`
  边界：未独立核验参赛记录、完赛定义与成绩分布
- **C-007** [accepted/verified_observation]: 课件把FlagOS实验和竞赛列为课程实践组成。
  证据：`E-009`
  边界：标题不足以说明实验镜像、任务、代码版本和验收命令
- **C-008** [qualified/reported_claim]: 课件报告相关学生或团队获得FlagOS全球挑战赛天津站桂冠。
  证据：`E-010`
  边界：未用赛事官网、榜单或获奖证书独立核验
- **C-001** [accepted/verified_observation]: 该课件把智能计算课程的人才定位指向智能基础设施的开发者和设计者。
  证据：`E-001`
  边界：这是课件提出的课程定位，不是对全国人才缺口规模的独立调查
- **C-003** [accepted/verified_observation]: 课程路线同时贯通高级语言到机器代码、智能应用到体系结构两条轴线。
  证据：`E-003`, `E-004`
  边界：这是一项教学设计主张，不等于每个层级均已有可运行实验
- **C-005** [accepted/verified_observation]: 课件将大模型辅助用于算子优化分析，并把工作拆为调度优化和内核优化。
  证据：`E-006`
  边界：只支持任务分解；不支持任何速度、成功率或优于人工的结论
- **C-009** [accepted/verified_observation]: 课件提出从注重程序设计转向注重系统设计。
  证据：`E-013`
  边界：未定义系统设计能力的量化评价方法
- **C-010** [accepted/verified_observation]: 课件把智能计算实验组织为软硬件技术栈、分阶段实验和综合实验。
  证据：`E-014`
  边界：未列出每个实验的代码、时长和验收命令
- **C-011** [accepted/verified_observation]: 课件将tiling定义为把运算分解到固定尺寸区块处理。
  证据：`E-015`
  边界：不支持把同页示意命中率当作BI-V150实测
- **C-012** [accepted/verified_observation]: 课件把IR定位为源语言与目标语言之间的程序表示。
  证据：`E-016`
  边界：不支持对TTIR或TTGIR具体结构的推断

另有 5 条已验收结论留在 `claims.jsonl`，因展示预算未在本页展开。

## 主线图谱

| ID | 类型 | 陈述 | 认识状态 | 证据/Claim |
|---|---|---|---|---|
| T-001 | Topic | FlagOS智能计算课程建设 — 新增课件的课程目标、全栈主线、AI辅助算子实践与赛教融合。 | explicit | C-002 |
| P-003 | PipelineStep | 大模型辅助算子优化 — 用大模型辅助分析算子调度优化和内核优化。 | explicit | C-005, E-006 |
| P-004 | PipelineStep | 竞赛式课程考核 — 将系统层优化打榜纳入课程组织。 | explicit | C-004, E-005 |
| P-005 | PipelineStep | FlagOS实验与竞赛 — 课件列出的课程实践组成。 | explicit | C-007, E-009 |
| L-001 | Limitation | 可复现实验协议缺失 — 第128页没有给出模型版本、提示词、基线或完整测量协议。 | inferred | C-005 |
| U-001 | Unknown | 教学效果与赛事结果待独立核验 — 参赛人数、完赛和名次属于讲者报告，当前封闭语料无法独立核验。 | inferred | C-006, C-008, E-007 +2 |
| Q-001 | Question | 如何组织可实践的智能计算课程？ — 从课程目标到五模块实验、竞赛和证据验收的根问题。 | user | — |
| W-001 | WorkVersion | 王晶课程建设课件 v20260808 — 165页PDF的文本抽取与页码证据版本。 | explicit | E-011, E-012 |
| C-101 | Concept | 应用驱动、全栈贯通 — 课程总体组织原则。 | explicit | C-002, E-002 |
| C-102 | Concept | 双轴贯通 — 高级语言到机器代码、智能应用到体系结构。 | explicit | C-003, E-003, E-004 |
| C-103 | Concept | 智能基础设施开发者与设计者 — 课程面向能理解并开发智能计算系统的人才目标。 | explicit | C-001, E-001 |
| C-104 | Concept | 系统设计能力 — 课程能力目标从程序设计推进到系统设计。 | explicit | C-009, E-013 |
| P-006 | PipelineStep | 分阶段与综合实验 — 按软硬件技术栈组织分阶段实验和综合实验。 | explicit | C-010, E-014 |
| C-105 | Concept | Tiling与固定尺寸分块 — 把运算分解到固定尺寸区块处理。 | explicit | C-011, E-015 |
| C-106 | Concept | 编译器中间表示 — 连接源语言与目标语言的程序表示。 | explicit | C-012, E-016 |
| P-007 | PipelineStep | 算子自动调优 — 课件列出的算子优化方法。 | explicit | C-013, E-017 |
| C-107 | Concept | 数据、流水线与张量并行 — 分布式训练部分并列的三种并行入口。 | explicit | C-014, E-018, E-019 +1 |
| C-108 | Concept | Ring全局归约 — 课件用环形通信说明全局归约。 | explicit | C-015, E-021 |
| C-109 | Concept | 课程思政 — 课件七部分提纲中的第七部分。 | explicit | C-016, E-022 |
| C-110 | Concept | FlagOS统一AI系统软件栈 — 课件第163页以统一AI系统软件栈定位FlagOS。 | explicit | C-017, E-023 |

### 关系

- 王晶课程建设课件 v20260808 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 如何组织可实践的智能计算课程？ —`part_of`→ FlagOS智能计算课程建设 [user]
- 应用驱动、全栈贯通 —`answers`→ 如何组织可实践的智能计算课程？ [explicit]
- 双轴贯通 —`implements`→ 应用驱动、全栈贯通 [explicit]
- 大模型辅助算子优化 —`implements`→ 应用驱动、全栈贯通 [inferred]
- 竞赛式课程考核 —`implements`→ 应用驱动、全栈贯通 [inferred]
- FlagOS实验与竞赛 —`implements`→ 应用驱动、全栈贯通 [inferred]
- 大模型辅助算子优化 —`limited_by`→ 可复现实验协议缺失 [inferred]
- 竞赛式课程考核 —`limited_by`→ 教学效果与赛事结果待独立核验 [inferred]
- 智能基础设施开发者与设计者 —`answers`→ 如何组织可实践的智能计算课程？ [explicit]
- 大模型辅助算子优化 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 竞赛式课程考核 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- FlagOS实验与竞赛 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 可复现实验协议缺失 —`part_of`→ FlagOS智能计算课程建设 [inferred]
- 教学效果与赛事结果待独立核验 —`part_of`→ FlagOS智能计算课程建设 [inferred]
- 智能基础设施开发者与设计者 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 系统设计能力 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 分阶段与综合实验 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- Tiling与固定尺寸分块 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 编译器中间表示 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 算子自动调优 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 数据、流水线与张量并行 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- Ring全局归约 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- 课程思政 —`part_of`→ FlagOS智能计算课程建设 [explicit]
- FlagOS统一AI系统软件栈 —`part_of`→ FlagOS智能计算课程建设 [explicit]

## 数字与协议

_尚无通过数字上下文门的 measurement。_

## 冲突与版本差异

_当前账本没有冲突组；这只表示尚未登记冲突。_

## 当前问题

- [critical] AI辅助算子优化教学使用的模型版本、提示词、基线、数据集、正确性门槛和性能测量协议分别是什么？ (`Q-101`, open)
- [critical] FlagOS课程实验对应的镜像、CoreX/PyTorch/Triton版本、仓库提交、实验任务和验收命令是什么？ (`Q-102`, open)
- [high] XPUOJ Agent读取题面、生成或修改代码、确认或自动提交时，账号权限、代码外发和自动提交安全边界是什么？ (`Q-103`, open)
- [medium] 课件报告的104人报名、全部完赛和FlagOS赛事名次能否由原始名单、榜单或证书独立核验？ (`Q-104`, open)

## 停车场

_无。_

## 语料覆盖

- 来源处置：{"excluded":0,"failed":0,"indexed":1,"pending":0,"stale":0}
- 全部来源明细：`sources.jsonl`
- failed、stale 和 excluded 均保留在总数中。
