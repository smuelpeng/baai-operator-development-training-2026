# 来源与同步台账

更新时间：2026-08-07

| 资料 | 来源 | 当前版本/状态 | 本地位置 | 备注 |
|---|---|---|---|---|
| BI-V150 Triton 试听实验 | `https://gitee.com/sunxt-0719/lab1` | `main@1969ed2741c5dbadf8961c85d827e7d05e3f0c30` | `lab1/` | 2026-08-05 重新克隆到当前归档目录；HEAD 提交时间 2026-08-04 13:05:04 +08:00 |
| Fused Add + RMSNorm Lab | `https://gitee.com/sunxt-0719/completed_lab2` | `master@3ba8e2d6d94556ba45be9ce15c8751d44986affe` | `completed_lab2/` | 2026-08-05 执行 `git pull --ff-only`，结果为 Already up to date；HEAD 提交时间 2026-08-05 08:13:56 +08:00 |
| 模块五配套实验 day2-lab5 | `https://gitee.com/yihan-long/day2-lab5` | `master@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f` | `day2-lab5/` | 2026-08-07 克隆；覆盖 Benchmark、Roofline、ixsmi/ixsys/ixkn Profiling 与报告生成；尚未在 BI-V150 上运行 |
| FlagPrism | `https://github.com/flagos-ai/FlagPrism` | `main@8d2647ba280aa66e328ff606f1ffb9a4e2962947` | `FlagPrism/` | 2026-08-07 克隆；仓库 README 标注 active development，提供 FlagTree Debugger/Profiler 可选组件入口 |
| 微信群通知与课件 | 微信本地文件缓存 | 已归档26份唯一PDF和1份DOCX，共136.09 MiB | `chat_files/` | 覆盖通知、日程及8月4日至8日已落盘材料；另登记1个与预习版 SHA 相同的授课版文件名别名 |
| 微信群对话 | “2026智源智算课程研修班1期”，界面显示 90 名成员 | 待完整回溯 | `records/course_summary.md` | 已记录截图可核验的新文件和仓库链接；微信目前要求扫码重新登录，未把未读取消息视为已总结 |
| WeChat MCP 读取通道 | `BiboyQG/WeChat-MCP`（第三方开源项目） | 已注册并通过 MCP 握手；23:30 再次读取未成功 | Codex MCP 名称 `wechat-mcp` | 微信进程可激活，但当前客户端未向 AX 暴露聊天标题、会话列表或搜索框；新增文件改由本地缓存补齐。它不是腾讯官方 API |
| 8月8日教学研讨 | 微信缓存 PDF | 已归档并提取日程 | `chat_files/00_通知与日程/8月8日教学研讨日程安排_2026年暑期研修班.pdf` | PDF封面写周六，正文标题误写周五；2026-08-08实际为周六。09:00开始，12:00结束 |
| 学生实验资源申请 | 微信缓存 DOCX | 已归档，待按需填写 | `chat_files/00_通知与日程/附录3+学生实验资源申请表v3.docx` | 申请需说明与FlagOS关系、实验窗口、共享存储、卡数/学生人数和学生联系方式；不擅自填写个人信息 |
| 全国高校人工智能区域技术转移转化中心（北京）介绍 | 微信缓存 PDF | 已归档并初读 | `chat_files/05_0807课件/0807——国转中心介绍.pdf` | 介绍高校成果转化、免费算力、项目征集、孵化和基金支持；项目/资金/人数为课件宣介口径，未做外部独立核验 |
| 算子开发开源组件调研 | 当前工作区已有文件 | 未绑定群内来源 | `ai_open_source_components_research_report.md` | 保留为辅助学习资料，不冒充群文件 |
| Notion：算子开发知识库｜智源智算研修班 | `https://app.notion.com/p/f467fd7204e44bd384a29ee3f00cdf74` | 原有19条记录；本次新增待同步5条、待更新1条 | 数据源 `collection://efabea14-d470-4e1b-81ee-3d8d7dc3a01f` | 2026-08-07 连接器登录到了另一 Notion 工作区，为避免误写已停止同步；完整正文草稿见 `records/0807_materials_update.md` |

## 完整性约定

- `已同步` 只表示本地仓库与远端对应分支一致，不代表实验已运行通过。
- 当前归档覆盖截至2026-08-07 23:30已落入微信本地缓存、且能通过名称或群截图确认属于本研修班的文件；已包含8月8日教学研讨安排和实验资源申请表。
- 群消息中的 `FlagPrism/pull/68/changes` 链接当前返回 404，且仓库远端不存在可直接 fetch 的 `pull/68/head`；已保留为待核验线索，不写成已确认变更。
- 微信截图中可见信息与 Gitee 仓库内容分开记账，避免把推断写成群内事实。
- Notion 中的“已入库”表示元数据、学习入口和证据边界已经建立，不表示课件已经逐页精读；精读完成后再更新“学习状态”“事实核验”“突出贡献”和正文笔记。
