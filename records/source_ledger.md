# 来源与同步台账

更新时间：2026-08-10

| 资料 | 来源 | 当前状态 | 本地位置 | 证据边界 |
|---|---|---|---|---|
| 五个 Gitee 课程仓库 | 群内发布的 Gitee 地址 | 固定提交已归档 | `modules/01-*`、`modules/02-*`、`modules/05-*` | 版本与用途见 `code_repository_inventory.md`；未声称实验已运行 |
| FlagPrism | `https://github.com/flagos-ai/FlagPrism` | `main@8d2647b...` 固定快照 | `modules/05-performance-engineering/tools/FlagPrism/` | README 标注 active development |
| 课程课件 | 微信本地文件缓存 | 27 份唯一 PDF、1 份 DOCX | `materials/` | 页数、大小、版本差异见 `courseware_inventory.md` |
| GitHub 私有课程仓库 | `https://github.com/smuelpeng/baai-operator-development-training-2026` | `main` | 当前工作区 | 课件和源码均直接保存，无 submodule |
| 微信群对话 | “2026智源智算课程研修班1期” | 尚未完成逐条回溯 | `records/course_summary.md` | 已核对截图可见信息与本地缓存；未读取消息不写成已总结 |
| WeChat MCP | 第三方 `BiboyQG/WeChat-MCP` | 已注册，当前 UI 可访问性不足 | `logs/wechat_mcp.log` | 并非腾讯官方 API；不能据此声称群消息完整 |
| 腾讯会议 | 会议号 `466-8665-3784` | 只有会议号，尚无可定位录制 ID/回放链接 | 待补 | 会议号本身不能证明录制存在，也不能直接取得逐字稿 |
| 8 月 8 日教学研讨 | 群内 PDF | 已归档并提取日程 | `materials/administration/8月8日教学研讨日程安排_2026年暑期研修班.pdf` | 封面写周六，正文误写周五；实际日期为周六 |
| 王晶课程建设专题 | 群内 PDF，群消息说明“这是王晶老师的 ppt” | 165 页 PDF 已归档、抽取并建立独立知识主题 | `materials/seminars/融入FlagOS的智能计算系统课程建设-王晶-20260808.pdf`、`knowledge/course-construction/` | SHA-256 为 `9a43c50d…`；文件名、封面题名和 PDF 内部 `Title` 不一致，内部 `Author=Zidong` 与封面王晶署名也不一致；课程成效与赛事数字按讲者报告登记，未独立核验 |
| 学生实验资源申请 | 群内 DOCX | 保存空表 | `materials/administration/附录3+学生实验资源申请表v3.docx` | 不代填或扩散学生联系方式 |
| 国转中心介绍 | 群内 PDF | 已归档并初读 | `materials/seminars/0807——国转中心介绍.pdf` | 规模、资金和年度目标按宣介课件记录，未独立核验 |
| Notion 知识库 | 原“算子开发知识库｜智源智算研修班” | 正文草稿保存在本地，尚未同步；本次先写入独立本地知识主题 | `records/0807_materials_update.md`、`records/0810_materials_update.md`、`knowledge/course-construction/` | 连接器登录到另一工作区时已停止写入，避免误建空记录 |

## 完整性约定

- 课程源码与课件的 SHA-256 由 `source_manifest.sha256` 和 `courseware.sha256` 记录。
- 5 个 Gitee 仓库为 `lab1`、`lab1-day2`、`lab2`、`completed_lab2`、`day2-lab5`；FlagPrism 是独立 GitHub 工具仓库。
- 群消息中的 `FlagPrism/pull/68/changes` 当前返回 404，远端也没有可直接 fetch 的 `pull/68/head`，保持为待核验线索。
- 课件内容、群截图和外部仓库分别记账。推断、课件声称和本地实测不能混写。
