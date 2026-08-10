# 作业与验收

本页的 `25%/25%/15%/15%/20%` 是本仓为五模块研修路径设计的验收方案。官方 OpenCourse 另提供 13 份作业文件，官方长期课程大纲同时提出“4 次模块实验占 40%”；现有材料没有证明 13 份作业全部等同于这 4 次计分实验。开设学期课时，应先从 [OpenCourse 对照指南](OFFICIAL_OPENCOURSE.md) 选择作业，再由开课单位明确计分关系。

## 总体评分

| 项目 | 权重 | 验收重点 |
|---|---:|---|
| 模块 1：Triton 与 GEMM | 25% | 正确性、边界 shape、可信计时、调参解释 |
| 模块 2：融合算子 | 25% | 前向/反向、注册、graph break、性能公平性 |
| 模块 3：编译链追踪 | 15% | 源码到 IR 的对应关系、pass 作用、证据页 |
| 模块 4：并行与通信 | 15% | 显存/通信模型、算法推演、假设说明 |
| 模块 5：性能诊断 | 20% | Benchmark、Roofline、Profiler 三层证据闭环 |

## 通用验收门槛

以下任一项缺失，性能结论不能评为“可信”：

- 没有 correctness gate；
- 没有设备与软件版本；
- baseline 与优化实现使用了不同 shape、dtype 或计时边界；
- 只给最快单次结果，没有样本和统计量；
- 失败配置、资源错误或异常值被删除；
- 结论没有指向原始结果文件或 profiler 证据。

## 各模块提交物

### 模块 1

- 完成 Vector Add TODO；
- 保存 Softmax、LayerNorm、GEMM 的正确性输出；
- 提交 block sweep 与 autotune 记录；
- 用 300–500 字解释 winner 与不同 shape 的差异。

### 模块 2

- 完成学生版全部 TODO；
- 前向和梯度误差结果；
- Dispatcher 与 `torch.compile` 捕获结果；
- 与 PyTorch、拆分 Triton、FlagGems 的性能表。

### 模块 3

- 一张 Python/Triton → TTIR → TTGIR → LLVM/目标代码的映射表；
- 至少三处 IR 变化及其目的；
- 一段说明 `BLOCK_*`、`num_warps` 或 `num_stages` 如何进入编译决策。

### 模块 4

- 4 或 8 rank Ring AllReduce 的块流向；
- 一组 α-β 成本计算；
- 为一个给定模型选择 DP/TP/PP/ZeRO 组合并说明显存、带宽和拓扑约束。

### 模块 5

- `benchmark_*.json`、`roofline.json`、`diagnosis.json`、`summary.json`；
- profiler 原始输出或命令日志；
- 填写完整的 `REPORT.md`；
- 指出模型判断与 profiler 证据一致或冲突的位置。

## 评分语言

优秀报告会把事实、推断和待验证假设分开；能够解释失败结果；能说明结论在哪些 shape、设备和版本上成立。只给截图、速度倍数或工具名称，无法形成可复核证据。
