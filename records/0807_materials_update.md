# 2026-08-07 新增材料学习与入库草稿

状态：本地整理已更新至 2026-08-07 23:30；Notion 待同步。

同步目标：算子开发知识库｜智源智算研修班，数据源 `collection://efabea14-d470-4e1b-81ee-3d8d7dc3a01f`。

阻塞原因：2026-08-07 Notion 连接器登录到错误工作区 “Awsim Awsim’s Space”，原数据源返回 404。为防止误写，本次不在该空间创建替代数据库。恢复正确连接后，应更新 1 条旧记录并新增 5 条记录。

## 一、模块三：AI 编译器原理与优化（更新既有记录）

- 日期：2026-08-06
- 版本：授课版，34 页；预习版为 32 页，两版均保留。
- 本地文件：`materials/courseware/module-03/0806-课件-授课版本-模块三：AI 编译器原理与优化.pdf`
- SHA-256：`e47dc4834611d05aff1e273b04edb90fcf057707082764b4ec70e415f338493d`
- 一句话：把 PyTorch 计算图经过 `torch.compile`、DSL、MLIR/LLVM lowering 到硬件代码的路径串成完整编译链，并解释图优化和 pass 在哪一层发生。

### 学习笔记

课程先区分静态图与动态图，再从算子融合、冗余节点消除、常量折叠、公共子图、算子替换和调度解释图级优化。`torch.compile` 的主线是 TorchDynamo 捕获 Python 程序、AOTAutograd 处理前后向图、TorchInductor 生成后端代码。继续下沉后，Triton/TileLang 作为 kernel DSL 表达并行计算，MLIR/LLVM 通过分层 IR 和 pass 完成 lowering、合法化和目标代码生成。

授课版新增 FlagGems 算子实现、SGLang Triton kernel 等代码入口，使课程从概念说明延伸到可读源码。学习时应把一个实际算子沿“PyTorch 调用—图捕获—Inductor/Triton—TTIR/目标 IR—设备代码”逐层追踪，而不是只记组件名。

### 图表与证据边界

- 关键图表：编译流程、计算图优化类型、`torch.compile` 组件关系、MLIR pass/lowering。
- 未嵌入 Notion 图片：课件为本地 PDF，无稳定公开图片 URL；同步时记录页码与图意。
- 事实核验：机器初核。依据为本地 34 页授课 PDF 及 SHA-256；课程代码链接只作为导航，尚未逐仓复现实验。

## 二、模块四：分布式通信原语与拓扑优化（新增）

- 日期：2026-08-07
- 本地文件：`materials/courseware/module-04/0807-课件-预习版本-模块四：分布式通信原语.pdf`
- 页数：39
- SHA-256：`a965c3bba8547724a30688d89660f0b13acd7edc14776e08fb39166592082719`
- 一句话：用集合通信原语、网络拓扑和 α-β 成本模型解释分布式训练中 AllReduce 算法为何要随消息规模切换。

### 核心内容

课程先区分机内 scale-up（NVLink、HCCS、UnifiedBus）与机间 scale-out（InfiniBand、RoCE），再把通信操作分为点对点和集合通信。核心原语包括 Broadcast、Reduce、Gather、Scatter、AllReduce、AllGather、ReduceScatter 和 AlltoAll。FSDP/ZeRO 场景中，AllReduce 可分解为 ReduceScatter 与 AllGather，这使参数切分和通信阶段可以对应起来。

拓扑部分覆盖 Clos、环面、全互联和 Dragonfly。算法部分比较朴素中心化 AllReduce、树形、环形与 halving-doubling。环形 AllReduce 分为 N-1 步 ReduceScatter 与 N-1 步 AllGather；单 rank 数据传输量为 `2 × (N-1) × K / N`，时间模型为 `2 × (N-1) × α + 2 × (N-1) × K / (N × β)`。树形模型为 `2 × log(N) × (α + K/β)`。课件给出的经验区间是：环形适合大消息（大于约 64 MB），树形适合中小消息（小于约 16 MB），halving-doubling 适合极小消息（小于约 1 MB）；这些阈值应视为课程经验值，实际取决于硬件和实现。

### 学习与实验动作

1. 用 4/8 个 rank 手算 Ring ReduceScatter 与 AllGather 的块流向。
2. 对固定 K 改变 N，分别计算 α 项和 β 项占比。
3. 在实际后端比较不同消息大小下的 AllReduce 延迟，验证算法切换点。
4. 把通信拓扑、链路带宽和 collective 实现一起记录，避免只用算法名称解释性能。

### 图表与证据边界

- 关键页：第 34 页给出 Ring AllReduce 伪代码、通信量与时间模型；第 37 页比较算法适用消息规模。
- 未嵌入 Notion 图片：无稳定公开图片 URL；本地已完成页面渲染抽检。
- 事实核验：机器初核。公式和范围依据本地 PDF；尚未在真实集群复测阈值。

## 三、模块五：性能评测与下一代内核生成（新增）

- 日期：2026-08-07
- 本地文件：`materials/courseware/module-05/0807-课件-预习版本-模块五：性能评测与下一代内核生成.pdf`
- 页数：38
- SHA-256：`f985e98bc973ee33f1d709d705367c3149dc335622c0a63058d1364f6e2a5e5c`
- 关联实验：`https://gitee.com/yihan-long/day2-lab5`
- 一句话：建立“Benchmark 定量差距—Roofline 判断瓶颈—Profiler 定位停顿—Autotuning/KernelGen 改进”的闭环。

### 核心内容

性能指标包括 FLOPS、带宽、延迟和利用率；评测部分还介绍 MLPerf 的训练与推理场景及质量约束。Profiler 关注 occupancy、寄存器、共享内存、block size、warp 状态、指令与内存行为。BI-V150 实验工具包括 `ixsmi`、`ixsys` 和 `ixkn-cli`。

Roofline 用两个约束描述性能上限：`P ≤ π`（计算峰值）与 `P ≤ β × I`（带宽乘算术强度）。ridge point 左侧通常是 memory-bound，右侧通常是 compute-bound。课件引用 ASPLOS 2025 的 Ascend component-based Roofline 工作并报告 11 个任务上 1.07–2.15 倍优化；该数字是课件转述的论文结果，不应写成本地复现结果。

Autotuning 的搜索变量包括 tiling、unrolling、vectorization 与线程/块绑定，搜索方法可用模拟退火、遗传算法、贝叶斯优化或强化学习。课程最后把 KernelGen、算子融合、FlashAttention、Amdahl 定律和 FlagOS 组件放入端到端优化视角：局部 kernel 加速只有在真实瓶颈路径上才会转化为整体收益。

### 配套实验闭环

1. `01_benchmark.py`：比较 torch、baseline 与 autotuned，记录 median/P99、吞吐和有效 TFLOP/s。
2. `02_roofline.py`：计算算术强度和 ridge point，判断 compute-bound 或 memory-bound。
3. `03_profiling.py`：用 `ixsmi`、`ixsys`、`ixkn-cli` 获取利用率、timeline 和 kernel 级证据。
4. `04_summary.py`：汇总诊断并填写 `REPORT.md`。

### 图表与证据边界

- 关键页：第 18 页 Roofline 模型；第 37 页对 Benchmark、Roofline、Profiling 三个实验问题进行对照。
- 未嵌入 Notion 图片：无稳定公开图片 URL；本地已完成页面渲染抽检。
- 事实核验：机器初核。课件、仓库 README 与脚本结构已核；实验尚未在 BI-V150 上运行。

## 四、基于 FlagTree 生态的 Profiler & Debugger 初探（新增）

- 日期：2026-08-07
- 讲师：周炽金（水木羽林研究员、华东师范大学助理教授，按课件署名）
- 本地文件：`materials/courseware/module-05/基于 FlagTree 生态的 Profiler&Debugger 初探 - 周炽金 老师.pdf`
- 页数：23
- SHA-256：`7ac96f0508f4cf4cd4f1390b8a54bd1ab99202788d4a4341ee8e0a88c724d140`
- 关联仓库：`https://github.com/flagos-ai/FlagPrism`
- 一句话：在 Triton/IR 与设备执行之间建立共享插桩和元数据层，使数值错误与性能开销都能映射回源语句。

### 要解决的问题

传统 Python/Host profiler 能看到调用和 kernel 边界，却难以回答设备内部哪条操作耗时、NaN/Inf 在哪一步出现、是否越界、mask 是否正确。Triton 语句经过 TTIR、优化和目标指令后可能融合、重排或重叠，因此“高层语句—IR op—设备指令”的语义映射不能靠时间顺序猜测。

### 设计拆解

Debugger 面向正确性、数值和内存，支持 statement/IR 粒度的摘要、完整值与地址观测；Profiler 面向执行和性能归因，记录 timestamp、duration、count、吞吐与厂商指标，并输出 timeline/Hatchet。两者共享 instrumentation 与 metadata。

课件新增四类观测 op：`record_summary_bundle` 采集数值摘要，`capture_memory_address` 采集访存地址，`record_full_value_ref` 保存完整值，`record_timeline` 采集 op 周期。插桩后的 kernel 通过隐藏的 device record buffer 写入固定 slot，避免每次观测都触发 D2H 同步；kernel 结束后再统一解码、聚合和可视化。

Profiler 输出包括 `profile.timeline.json`、`profile.hatchet`、`profile.meta.json` 和 `profile.vendor.json`，可分别服务 Perfetto 时间线、调用树聚合、会话配置与厂商原始字段关联。

### 路线图与限制

课件路线图写明 2026 年 4–7 月在 Ascend/Iluvatar 上形成原型，2026 年 8 月计划以 v0.1 进入 FlagTree，2026 年 12 月目标 v1.0 和至少 6 个后端。这些是课件中的计划/目标，不等同于当前仓库已经全部交付。FlagPrism README 同样标注 active development。

群消息中出现的 `FlagPrism/pull/68/changes` 当前返回 404，且远端没有可 fetch 的 `pull/68/head`，因此 PR 内容保持“有疑点/待核验”，不写成已合并功能。

### 图表与证据边界

- 关键页：第 12 页观测 op 与插桩数据流；第 20 页 Profiler 输出文件和可视化。
- 未嵌入 Notion 图片：无稳定公开图片 URL；本地已完成页面渲染抽检。
- 事实核验：机器初核；路线图按“课件声称”记录，PR68 单独标为有疑点。

## 五、模块五配套实验：day2-lab5（新增）

- 来源：Gitee
- 地址：`https://gitee.com/yihan-long/day2-lab5`
- 本地目录：`modules/05-performance-engineering/labs/performance-analysis/`
- 版本：`main@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f`
- 提交时间：2026-07-31 17:18:35 +0800
- 一句话：把同一 GEMM workload 的速度、瓶颈类型和 kernel 级停顿串成三层证据，并自动汇总实验报告。

### 仓库结构与执行顺序

- `setup.sh`、`00_check_env.py`、`00_smoke_cuda.py`：准备和验证环境。
- `01_benchmark.py`：offline/server 两类负载，比较 torch、baseline、autotuned。
- `02_roofline.py`：计算 arithmetic intensity、ridge point 和距硬件上限的位置。
- `03_profiling.py`：调用 BI-V150/FlagOS 环境中的 `ixsmi`、`ixsys`、`ixkn-cli`。
- `04_summary.py`、`REPORT.md`：生成结构化结果与实验报告。
- `results/`：保存 benchmark JSON、Roofline、diagnosis 和 summary。

### 复现状态

代码与 README 已审阅，仓库已锁定 commit；当前机器没有确认可用的 BI-V150/FlagOS 运行环境，因此未声称实验通过。后续运行必须记录设备、驱动、CoreX/PyTorch/Triton 版本、shape/dtype、warmup、同步、采样次数和 correctness gate。

## 六、FlagPrism：FlagTree Profiler 与 Debugger 工具套件（新增）

- 来源：GitHub
- 地址：`https://github.com/flagos-ai/FlagPrism`
- 本地目录：`modules/05-performance-engineering/tools/FlagPrism/`
- 版本：`main@8d2647ba280aa66e328ff606f1ffb9a4e2962947`
- 提交时间：2026-08-05 18:01:47 +0800
- 一句话：作为 FlagTree 的可选组件集合，为不同后端提供统一的 Debugger/Profiler 构建入口与共享插桩能力。

### 仓库定位

README 将项目标为 active development。FlagTree 可将其作为 `third_party/FlagPrism` 子模块使用，并通过组合 wheel 暴露 `flagtree.debugger` 与 `flagtree.profiler` 入口；构建开关为 `TRITON_BUILD_FLAGPRISM`。在 Ascend/CANN 路径上，Profiler 可复用 Debugger 的插桩能力，再关联厂商侧原始 profiler 字段。

### 风险与后续

仓库状态和课件路线图都说明它仍处于快速开发期，接口、后端覆盖和构建方式可能变化。首次学习应先固定 commit，再做最小构建与 smoke test；不能把课件中的 2026 年 12 月目标当成已经实现。群消息中的 PR68 链接仍待发布者确认。

## 七、Notion 同步清单

### 本轮新增待入库资料

1. `模块四：分布式通信原语（授课版）`：49 页，更新原预习版记录；新增 Broadcast 下界、树形广播、分块流水化和 simultaneous trees。
2. `8月8日教学研讨与结业安排`：记录 09:00-12:00 的讲座、三段教学研讨和结业仪式，并注明星期标注冲突。
3. `学生实验资源申请表`：作为课程管理/算力申请模板入库，正文记录必填字段与个人信息保护要求。
4. `全国高校人工智能区域技术转移转化中心（北京）介绍`：作为算力支持与成果转化资料入库；宣介数字保持“来源声称”标记。

恢复正确工作区后执行：

1. 获取原数据库 schema，按标题和原始链接查重。
2. 更新既有“模块三：AI 编译器原理与优化”记录，保留预习版版本信息并切换主路径到授课版。
3. 新建模块四通信原语、模块五性能评测、Profiler/Debugger、day2-lab5、FlagPrism、8月8日日程、实验资源申请表、国转中心介绍等记录。
4. 课件与仓库双向写入“关联资源”；仓库记录不伪装成课程 PDF。
5. 回读 6 条页面，核对正文标题、链接、SHA-256、页数、核验状态和图表页码说明。
