# 课件与原始材料索引

本目录保存群内课件原件，以及从 FlagOS 官方 OpenCourse 固定提交补齐的可编辑原件。主课程按五个模块分类，专题报告、平台说明、官方入口和行政文件单独存放。PDF/PPTX 保留原文件名和内容，未重新导出或改写。

## 五个主模块

| 模块 | PDF | 官方可编辑原件 | 版本说明 |
|---|---|---|
| 1 | [AI 系统软件基础与异构计算](courseware/module-01/模块一：AI%20系统软件基础与异构计算.pdf) | [PPTX](courseware/module-01/official-editable/课件-模块一：AI%20系统软件基础与异构计算.pptx) | 39 页 |
| 2 | [高性能 AI 算子与算子工程](courseware/module-02/0805-课件-模块二：高性能AI算子与算子工程.pdf) | [PPTX](courseware/module-02/official-editable/课件-模块二：高性能AI算子与算子工程.pptx) | 63 页 |
| 3 | [授课版](courseware/module-03/0806-课件-授课版本-模块三：AI%20编译器原理与优化.pdf) / [预习版](courseware/module-03/0806-课件-预习版本-模块三：AI%20编译器原理与优化.pdf) | [32 页 PPTX](courseware/module-03/official-editable/课件-模块三：AI%20编译器原理与优化.pptx) / [上游错标的 34 页 PPTX](courseware/module-03/official-editable/upstream-mislabeled/课件-模块四：分布式并行训练.pptx) | 34 页文件在官方路径误标为模块四，实际逐页均为 AI 编译器 |
| 4 | [分布式并行训练](courseware/module-04/0806-课件-预习版本-模块四：分布式并行训练.pdf) / [通信授课版](courseware/module-04/0807-课件-授课版本-模块四：分布式通信原语.pdf) / [通信预习版](courseware/module-04/0807-课件-预习版本-模块四：分布式通信原语.pdf) | [通信原语 PPTX](courseware/module-04/official-editable/课件-模块四：分布式通信原语.pptx) | 官方没有 61 页并行训练 PDF 的可编辑原件 |
| 5 | [性能评测与下一代内核生成](courseware/module-05/0807-课件-预习版本-模块五：性能评测与下一代内核生成.pdf) / [Profiler & Debugger](courseware/module-05/基于%20FlagTree%20生态的%20Profiler&Debugger%20初探%20-%20周炽金%20老师.pdf) | [PPTX](courseware/module-05/official-editable/课件-模块五：性能评测与下一代内核生成.pptx) | 主课与工具专题配套 |

## 专题材料

`seminars/` 包含开场介绍、FlagOS AI 开放计算、国产算力课程建设、LearnBuddy、FlagOS 挑战赛、KernelGen、FlagTree TLE、推理框架、开发者说明和国转中心介绍。它们用于补充生态、课程建设和产业转化背景，不替代五个主模块。

2026 年 8 月 10 日补归档王晶老师 165 页专题课件 [《融合 FlagOS 的智能计算课程建设》](seminars/融入FlagOS的智能计算系统课程建设-王晶-20260808.pdf)。文件名使用“融入 FlagOS”，封面题名使用“融合 FlagOS”，索引按封面题名著录、按原文件名保存；课程页段与五模块的对应关系见 [课程地图](../docs/COURSE_MAP.md)。

## 平台与行政材料

- `platform/`：群内线上实验室平台介绍，以及官方 16 页重导出版本；两份正文高度一致但哈希不同，均保留。
- `official-opencourse/`：官方许可证和本期五份实验入口原件；完整对照见 [官方 OpenCourse 指南](../docs/OFFICIAL_OPENCOURSE.md)。
- `administration/`：通知、研修须知、原班日程、8 月 8 日教学研讨安排和实验资源申请空表。
- `administration/private/`：带个人姓名的邀请函。仓库必须保持私有，教学分发时应排除此目录。

## 完整性

文件数量、页数、版本差异和异常说明见 [课件清单](../records/courseware_inventory.md)。SHA-256 见 [课件校验文件](../records/courseware.sha256)。微信群缓存中没有找到对应的 `.ppt` 或 `.pptx` 原件；本次 6 份 PPTX 来自 OpenCourse 官方发布版固定提交，不冒充群缓存文件。
