---
schema: synthesize-knowledge.questions/v1
graph_version: v0.45.0
generated_at: 2026-08-10T04:52:22+00:00
active_limit: 8
---

# 当前问题队列

<!-- knowledge-system questions v1; 只编辑 USER 区，然后运行 sync-questions -->
把提交状态改为 ready、defer 或 unknown；draft 内容会原样保留，但不会写回机器账本。

## 本轮问题

<!-- AUTO:BEGIN Q-101 -->
### Q-101 · AI辅助算子优化教学使用的模型版本、提示词、基线、数据集、正确性门槛和性能测量协议分别是什么？
- 父问题：—
- 优先级：critical
- 回答类型：gap
- 影响对象：P-003
- 为什么现在问：没有这些信息就不能把第128页的教学形式变成可复现实验。
- 已知边界：当前封闭语料只说明调度优化和内核优化两类任务。
- 建议查看：教师实验说明、课程代码、OJ题面与评分脚本
- 回答契约：给出版本化的任务合同、输入矩阵、验收命令、原始结果与失败项。
<!-- AUTO:END Q-101 -->

<!-- USER:BEGIN Q-101 -->
- 提交状态：draft
- 我的回答：
<!-- USER:ANSWER:BEGIN -->

<!-- USER:ANSWER:END -->
- 原文依据或文件位置：
<!-- USER:EVIDENCE:BEGIN -->

<!-- USER:EVIDENCE:END -->
- 我的置信度：中
- 希望 Skill 继续查：
<!-- USER:FOLLOWUP:BEGIN -->

<!-- USER:FOLLOWUP:END -->
<!-- USER:END Q-101 -->

<!-- AUTO:BEGIN Q-102 -->
### Q-102 · FlagOS课程实验对应的镜像、CoreX/PyTorch/Triton版本、仓库提交、实验任务和验收命令是什么？
- 父问题：—
- 优先级：critical
- 回答类型：gap
- 影响对象：P-005
- 为什么现在问：课件只给出FlagOS实验与竞赛入口，仓库需要可执行边界。
- 已知边界：本仓已有BI-V150单卡代表性链路记录，但不能反推王晶课件所指全部实验。
- 建议查看：FlagOS实验平台配置、教师Lab说明、课程仓提交记录
- 回答契约：逐实验列出环境指纹、固定提交、命令、预期结果和已验证状态。
<!-- AUTO:END Q-102 -->

<!-- USER:BEGIN Q-102 -->
- 提交状态：draft
- 我的回答：
<!-- USER:ANSWER:BEGIN -->

<!-- USER:ANSWER:END -->
- 原文依据或文件位置：
<!-- USER:EVIDENCE:BEGIN -->

<!-- USER:EVIDENCE:END -->
- 我的置信度：中
- 希望 Skill 继续查：
<!-- USER:FOLLOWUP:BEGIN -->

<!-- USER:FOLLOWUP:END -->
<!-- USER:END Q-102 -->

<!-- AUTO:BEGIN Q-103 -->
### Q-103 · XPUOJ Agent读取题面、生成或修改代码、确认或自动提交时，账号权限、代码外发和自动提交安全边界是什么？
- 父问题：—
- 优先级：high
- 回答类型：gap
- 影响对象：P-004
- 为什么现在问：要把AI Coding纳入教学，必须先明确数据和操作权限。
- 已知边界：当前课件展示流程，但未给出权限、审计或防误提交说明。
- 建议查看：OJ使用规范、Agent配置、课程数据管理规则
- 回答契约：列出允许读取/发送的数据、人工确认点、日志保留和撤销机制。
<!-- AUTO:END Q-103 -->

<!-- USER:BEGIN Q-103 -->
- 提交状态：draft
- 我的回答：
<!-- USER:ANSWER:BEGIN -->

<!-- USER:ANSWER:END -->
- 原文依据或文件位置：
<!-- USER:EVIDENCE:BEGIN -->

<!-- USER:EVIDENCE:END -->
- 我的置信度：中
- 希望 Skill 继续查：
<!-- USER:FOLLOWUP:BEGIN -->

<!-- USER:FOLLOWUP:END -->
<!-- USER:END Q-103 -->

<!-- AUTO:BEGIN Q-104 -->
### Q-104 · 课件报告的104人报名、全部完赛和FlagOS赛事名次能否由原始名单、榜单或证书独立核验？
- 父问题：—
- 优先级：medium
- 回答类型：fact
- 影响对象：U-001
- 为什么现在问：这些数字可用于教学案例，但不能与已核验事实混写。
- 已知边界：当前只有讲者课件报告。
- 建议查看：课程报名/完赛记录、赛事官网榜单、获奖证书
- 回答契约：提供可定位原始记录；无法获得时保持reported_claim和qualified状态。
<!-- AUTO:END Q-104 -->

<!-- USER:BEGIN Q-104 -->
- 提交状态：draft
- 我的回答：
<!-- USER:ANSWER:BEGIN -->

<!-- USER:ANSWER:END -->
- 原文依据或文件位置：
<!-- USER:EVIDENCE:BEGIN -->

<!-- USER:EVIDENCE:END -->
- 我的置信度：中
- 希望 Skill 继续查：
<!-- USER:FOLLOWUP:BEGIN -->

<!-- USER:FOLLOWUP:END -->
<!-- USER:END Q-104 -->

## 已归档问题

_无。_
