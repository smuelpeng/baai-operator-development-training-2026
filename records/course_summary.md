# 课程资料总结

更新时间：2026-08-10

## 一、课程与参训信息

- 群名：2026智源智算课程研修班1期。
- 时间：2026 年 8 月 4 日至 8 日；8 月 4 日上午报到，具体日程以群内通知为准。
- 地点：北京中关村创业大街 12 号楼 5 层。
- 装备：笔记本电脑、电源适配器。
- 费用：研修不收费；交通、住宿、餐饮自理，按各单位要求报销。
- 联系人：群内须知消息列出余老师、赵老师和李老师的手机号。为减少私人联系方式扩散，本整理文件不重复抄录；需要时以群内原始须知为准。

## 二、已确认的学习主线

### 1. BI-V150 与 Triton 基础

模块 1 的 `labs/triton-basics` 面向天数智芯 BI-V150，使用 CoreX 适配的 PyTorch 和 Triton；`labs/gemm-tuning` 单独保存 Day 2 的 GEMM 调优实验。两份仓库从环境与设备执行开始，依次覆盖：

1. Vector Add：program、grid、offset、mask、load/store。
2. Softmax、LayerNorm：数值稳定性、尾块处理和 fp32 reduction。
3. Tiled GEMM：二维 tile、stride、K-loop、`tl.dot`，fp16 输入/输出和 fp32 accumulator。
4. Benchmark：warmup、同步、重复采样、中位数和输出复用。
5. 调优：block sweep、scheduler sweep 与 Triton autotune。

Day 1 的核心任务是补全 Vector Add kernel；Day 2 的核心任务是依据 sweep 的 PASS 结果配置少量代表性 autotune candidates。所有性能比较先经过 correctness gate，不能只保留最快结果。

### 2. 算子全栈实验：Fused Add + RMSNorm

模块 2 的 `fused-rmsnorm-student` 是学生实验版，保留前向、反向和注册环节的 TODO；`fused-rmsnorm-reference` 是对应参考实现。两份仓库以一个融合算子贯穿 PyTorch 栈：

1. 编写 Triton 前向 kernel。
2. 推导并实现反向 kernel。
3. 通过 PyTorch dispatcher 注册自定义算子。
4. 验证 `torch.compile` 捕获注册算子且无 graph break。
5. 与 PyTorch、拆分 Triton baseline 和 FlagGems 比较正确性与吞吐。

算子契约为：先计算 `z = x + residual`，再用 `rrms = 1 / sqrt(mean(z*z) + eps)` 归一化，并乘以可学习权重。float32 前向和梯度误差目标低于 `2e-5`；在 `4096 x 4096` 上，融合实现吞吐目标至少达到 FlagGems 的 90%。

### 3. 算子开发生态定位

当前目录已有调研报告把学习生态分为几层：TileLang/Triton 负责底层 kernel；FlagGems/FlagTree/FlagPerf 分别对应 PyTorch 算子接入、多后端编译和评测；MiniCPM/LLaMA Factory 更适合作为真实模型 workload，而不是底层算子开发工具。

### 4. AI 编译器、集合通信与性能工程

8月6日至7日新增材料把学习主线从单算子实现扩展到系统优化：

1. AI 编译器：从计算图、`torch.compile`、TorchDynamo/AOTAutograd/TorchInductor，继续下沉到 Triton/TileLang、MLIR pass 与硬件代码生成。
2. 集合通信：围绕 Broadcast、Reduce、AllReduce、AllGather、ReduceScatter、AlltoAll，比较树形、环形和 halving-doubling 算法，并把通信量、时延和带宽写成可计算模型。
3. 性能诊断：先做 Benchmark，再用 Roofline 判断 compute-bound 或 memory-bound，最后用 `ixsmi`、`ixsys`、`ixkn-cli` 找到 kernel 级停顿证据。
4. Profiler/Debugger：在 Triton/IR 到设备执行之间插入观测算子，分别采集数值摘要、访存地址、完整值和时间线，解决高层语句与底层指令语义难对齐的问题。
5. 配套 Lab：`modules/05-performance-engineering/labs/performance-analysis/` 已归档。2026 年 8 月 8 日已在单卡 BI-V150 上验证环境检查、MatMul 冒烟和最小 LoadGen/JSON 链路；完整 Offline/Server、Roofline 图片、平台 profiler 和双卡路径仍待验收。动态证据见 `records/experiments/2026-08-08-flagos-bi-v150-validation.md`。

模块四授课版在原有 AllReduce 比较之后增加了 Broadcast 专题。分析先约定单端口、同构全连接网络和 `α + nβ` 传输模型，再讨论轮数下界、MST/二项树、消息分块流水化及 simultaneous trees。学习时需要区分“启动时延 α 的轮数优化”和“每字节成本 β 的带宽优化”，并根据消息大小选择树、环或流水方案。

### 5. 教学转化、实验资源和后续支持

8月7日晚新增的国转中心介绍把课程资源延伸到高校成果转化和算力申请。材料介绍免费 GPU/云资源、概念验证、中试、孵化与投融资支持，并列出大模型、具身智能、多模态、AI for Science 和 AI+产业等方向。相关规模、资金和年度目标来自宣介课件，尚未进行外部独立核验。

学生实验资源申请表要求学校和教师说明课程、实验、FlagOS 相关性、共享存储、起止日期、卡数或参与人数，并另附包含手机号或邮箱的学生名单。该表涉及个人联系方式，归档时只保存空表，不代填或扩散学生信息。

8月8日教学研讨安排为：

- 09:00-10:30：王晶教授，人工智能课程体系建设及赛教融合。
- 10:30-10:45：茶歇。
- 10:45-11:00：高校 AI 系统软件教学现状交流。
- 11:00-11:15：课程内容与教学方法讨论。
- 11:15-11:30：实验平台与 AI 教育平台讨论。
- 11:30-12:00：结业仪式与证书颁发。

日程封面写“周六”，正文页误写“周五”；2026年8月8日实际为周六。

8 月 10 日补归档王晶教授主讲的 165 页《融合 FlagOS 的智能计算课程建设》。课件从人才目标和“应用驱动、全栈贯通”出发，覆盖课程体系、AI 技术栈、体系结构与 AI 芯片、AI 编译与算子优化、AI 辅助算子调度与内核优化、赛教融合、并行训练、FlagScale、课程思政和 FlagOS。仓库将其定位为跨模块课程总览；五个主模块的代码、推演、环境记录与性能验收仍是技术事实入口。

课件还展示了校内竞赛、OJ Agent 和 FlagOS 赛事案例。104 人报名、全部完赛及赛事名次属于讲者课件报告，本次没有用报名记录、榜单或证书独立核验。第 128 页列出两个算子优化方向，并展示讲者报告的课程结果，但没有给出模型版本、提示词、基线、数据集或测量协议，不能据此采信教学成效或生成性能结论。

## 三、建议的学习记录模板

每次实验至少记录：

- 仓库 commit SHA、分支和是否有本地修改；
- 设备型号、驱动、CoreX、PyTorch、Triton 版本；
- 测试 shape、dtype、tile/config；
- correctness gate、最大误差、NaN/Inf 检查；
- warmup、同步、采样次数、统计量；
- PASS/FAIL、资源错误和原因；
- 与 baseline/FlagGems 的公平比较口径。

## 四、课件归档状态

截至 2026 年 8 月 10 日，已从微信本地文件缓存归档 27 份唯一 PDF 和 1 份 DOCX；与 FlagOS 官方 OpenCourse 固定提交对账后，又补入 6 份可编辑 PPTX 和 1 份官方平台 PDF 版本。当前 `materials/` 共 28 PDF、6 PPTX、1 DOCX，合计 243,667,406 字节（232.38 MiB）。课件按五个教学模块、专题、平台和行政目录管理，详见 `records/courseware_inventory.md`。

官方 2026 暑期教师版的 24 项现已全部可定位：12 份 PDF 与本地原件 SHA-256 完全相同，故不重复保存；其余 12 项已补档。官方长期课程另有 48 学时主线、13 份中文作业和四个组件 best-practices，本仓只建立固定提交索引，不把 3 GiB 以上的重复课件整仓复制进来。

本次还补入模块三 IR 实验和模块四推理部署手册的精选源码快照。模块三上游提交存在事实冲突：BI-V150 README 与随仓 NVIDIA H200 `run.json` 不一致，因此没有采用其生成结果，也没有把源码归档写成硬件验收通过。模块四分布式训练飞书 Wiki 在 2026-08-10 的未登录访问中进入登录流程，正文仍是缺口。完整边界见 `docs/OFFICIAL_OPENCOURSE.md`。

以下内容尚未声称完成：

- 群聊消息的逐条回溯与完整摘要；
- 8 月 10 日之后群内可能出现的新消息与新文件增量检查；
- 群文件发布者和每条文件消息时间的逐项映射。
- 官方模块四分布式训练飞书实验指南正文，以及官方飞书视频索引的访问权限与内容核验。

后续应继续在 `source_ledger.md` 中补充发布者、发布时间和对应群消息。
