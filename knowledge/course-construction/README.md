# 智能计算课程建设与 FlagOS 教学融合

这是算子开发知识库中的独立“课程建设”主题，服务教师备课、AI Coding 任务设计和赛教融合。当前封闭语料清单只有王晶老师 2026 年 8 月 8 日的 165 页课件；其他历史课件仍由 `materials/` 和 `records/courseware_inventory.md` 管理，尚未静默并入本主题。

## 入口

- [学习总图](learning-map.md)：默认入口，总图与局部关系。
- [主线结论](mainline.md)：稀疏主线、数字状态和当前问题。
- [原文证据册](evidence-notebook.md)：原文证据、页码、支持范围与不支持范围。
- [问题队列](questions.md)：待补的环境、评测、权限和独立核验问题。
- [来源](sources.jsonl)、[结论](claims.jsonl)、[节点](nodes.jsonl)、[关系](edges.jsonl)：机器可读账本。
- [检索与审计回执](receipts/)：每次词法查询和最终内部审计的可审计回执。
- [FTS5 索引](rag.sqlite3)：可重建的 contentless FTS5 索引，不包含外部发现结果。

来源路径使用仓库相对路径。重建索引、查询或审计时应从仓库根目录运行已安装的 `synthesize-knowledge` 执行器；未安装该 Skill 时，Markdown 主线和证据册仍可直接供 AI Coding 使用。

## 当前边界

- 只声明基于原文字符区间的词法检索，不声明语义或混合召回。
- 课件中的课程成效、人数和赛事名次按讲者报告保留，没有改写成独立核验事实。
- AI 辅助算子优化只形成教学任务分解；正确性、性能和环境结论必须回到五模块代码与实验报告。
