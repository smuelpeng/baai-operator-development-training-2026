# 2026 智源智算课程研修班 1 期：算子开发资料归档

本仓库用于保存并整理“2026智源智算课程研修班1期”的内部学习资料。课程时间为 2026 年 8 月 4 日至 8 日，地点为北京中关村创业大街 12 号楼 5 层。

> 内部资料：仓库应保持为 GitHub 私有仓库。课件、名单模板、邀请函和群内资料不得公开转载；具体使用范围以主办方要求为准。

## 目录

- `lab1/`：Git 子模块，BI-V150 上的 Triton 两日实验。
- `completed_lab2/`：Git 子模块，Fused Add + RMSNorm 完整算子实验。
- `day2-lab5/`：Git 子模块，Benchmark、Roofline 和 Profiling 实验。
- `FlagPrism/`：Git 子模块，FlagTree Debugger/Profiler 工具套件。
- `chat_files/`：微信群课件原件，按通知和课程日期分类。
- `records/course_summary.md`：课程内容、学习主线与待办总结。
- `records/source_ledger.md`：来源、版本、同步状态与完整性记录。
- `records/courseware_inventory.md`：26份PDF和1份DOCX的分类、页数与版本记录。
- `records/courseware.sha256`：课件原件的SHA-256清单。
- `records/source_manifest.sha256`：已归档文件的 SHA-256 清单。
- `ai_open_source_components_research_report.md`：当前目录已有的算子开发开源组件调研报告。

## 获取仓库

```bash
git clone --recurse-submodules <private-repository-url>
```

若已完成普通 clone：

```bash
git submodule update --init --recursive
```

## 使用原则

1. 群文件原件保持原文件名，不直接修改；整理性说明放在 `records/`。
2. 外部代码仓库以 Git 子模块固定 commit，保留上游来源和版本证据。
3. 任何实验结果必须同时记录设备、软件版本、shape、dtype、正确性结果和性能口径。
4. BI-V150 使用 CoreX 的 CUDA 兼容接口，但不是 NVIDIA GPU；API 兼容、源码可移植和性能可移植应分别判断。

## 当前归档状态

- 26 份唯一 PDF、1 份 DOCX，共 136.09 MiB。
- 4 个实验或工具仓库以子模块引用。
- 微信群消息尚未完成逐条回溯；当前材料范围以 `records/source_ledger.md` 为准。
- Notion 正文草稿保存在 `records/0807_materials_update.md`，尚未同步到原知识库。
