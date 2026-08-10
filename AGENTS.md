# AGENTS.md

本文件是 AI Coding 助手在本仓库中的默认操作手册。目标是帮助开发者快速完成算子实现、框架接入、调优和性能诊断，同时保留可复核证据。

## 开始任何任务前

依次阅读：

1. `docs/AI_CONTEXT.md`：平台事实、术语和代码地图；
2. 当前任务的 `TASK.md`：算子合同、目标平台、允许修改范围和验收门槛；
3. `docs/KNOWLEDGE_INDEX.md`：根据问题找到最短参考路径；
4. 对应模块 README 与最接近的代码样例。

需要进入 FlagOS 实例时，另读 `docs/FLAGOS_ONLINE_LAB.md`，先保存实例配额、运行时路径和设备指纹，再执行实验。

如果任务没有 `TASK.md`，先用 `python3 scripts/new_operator_task.py <name>` 创建工作区，再补全任务合同。合同不明确时可以实现参考函数和测试框架，但不能猜测会改变算子语义的行为。

## 任务路由

| 任务 | 首选知识与代码入口 |
|---|---|
| Triton 语法、mask、tile、reduction | `modules/01-ai-system-and-triton/` |
| GEMM、block sweep、autotune | `modules/01-ai-system-and-triton/labs/gemm-tuning/` |
| 算子融合、反向、Dispatcher、Meta | `modules/02-high-performance-operators/` |
| `torch.compile`、TTIR/TTGIR、lowering | `modules/03-ai-compiler/`；官方候选源码见 `labs/official-triton-ir/`，运行前先读其冲突说明 |
| 多卡并行、collective、α-β 模型 | `modules/04-distributed-training-and-communication/` |
| Benchmark、Roofline、Profiler | `modules/05-performance-engineering/` |
| FlagTree Debugger/Profiler 源码 | `modules/05-performance-engineering/tools/FlagPrism/` |

## 源码与资料边界

- `materials/` 是群内原始资料。除归档任务外不要修改、重命名或重新导出。
- `records/` 保存来源与 SHA-256。只有资料同步或版本更新任务才应改动。
- `modules/*/labs/` 和 `modules/05-*/tools/FlagPrism/` 是固定上游快照。学习任务可以按明确要求填写 TODO；新算子开发默认在 `workspaces/<task>/` 进行。
- `official-triton-ir/upstream/` 与 `official-inference-deployment/upstream/` 也是固定上游精选快照。不要直接改写；前者的 BI-V150 README 与随仓 H200 结果曾冲突，后者包含安装、模型下载和服务启动命令，均须先在独立任务中核验。
- `templates/` 是脚手架源文件。创建具体任务后，在生成的工作区中开发。
- 参考实现只能用于理解和最终核对。先完成学生版或工作区实现，再对照参考代码。

## 算子开发顺序

不要跳过前置门槛。

1. **任务合同**
   - 数学定义与输出顺序；
   - shape、dtype、device、layout、stride、contiguous 要求；
   - inplace/out-of-place、别名与副作用；
   - forward、backward、autograd、Meta 和 compile 要求；
   - 误差阈值、性能目标和目标硬件。
2. **可信参考**
   - 先写最直接的 PyTorch/NumPy 参考；
   - 高精度归约使用 fp32 或合同指定精度；
   - 参考实现不进入 kernel 计时。
3. **正确性矩阵**
   - 最小 shape、典型 shape、非整 tile shape、边界 shape；
   - 每个受支持 dtype；
   - 非连续输入仅在合同要求时测试；
   - 检查 NaN/Inf、最大绝对/相对误差；
   - backward 任务检查每个输入梯度。
4. **最小 kernel**
   - 先用清楚的 grid、offset、mask、load/compute/store；
   - 避免在正确性通过前加入复杂 autotune、融合或异步流水；
   - 写清编译期参数和 accumulator dtype。
5. **框架接入**
   - 校验输入合同；
   - 明确 contiguous 和输出分配；
   - 需要时注册 CUDA、AutogradCUDA 和 Meta；
   - 检查 `torch.compile` 图数与 graph break。
6. **可信性能测试**
   - correctness gate 必须先通过；
   - JIT/autotune 与 steady-state 计时分开；
   - warmup、显式同步、多样本、median/P99；
   - baseline 与优化实现使用相同输入、dtype、shape、输出复用和计时边界；
   - 保存失败配置与资源错误。
7. **诊断与改进**
   - Benchmark 定量差距；
   - Roofline 给出计算/带宽假设；
   - Profiler 用 occupancy、warp state、instruction、memory 或 timeline 验证；
   - 每次只做能被实验隔离的改动，再回到正确性门槛。

## BI-V150 / CoreX 约束

- `cuda:0` 和 `torch.cuda` 是 CoreX 兼容接口，不代表 NVIDIA 硬件。
- 课程环境记录的 warp size 为 64。不要默认采用 32-thread warp 假设。
- 不要安装公开版 `torch` 或 `triton` 覆盖平台适配版本。
- CoreX 4.4 中的 `torch.__version__` 可能不含 `+corex`；结合 `torch.__file__`、`COREX_HOME`、库路径、设备属性和最小 kernel 判断环境。
- 其他平台的 tile、二进制和性能数字只能作为候选假设，需要在目标平台重测。
- 缺少 BI-V150 时可以完成静态分析、参考实现、测试设计和语法检查；报告必须写明“未在目标平台运行”。

## 完成定义

只有同时满足以下条件，才能把任务标记为完成：

- 算子合同已写入 `TASK.md`；
- 正确性矩阵通过，失败项有原始错误；
- 验证命令可复制；
- 性能主张包含设备、版本、shape、dtype 和统计口径；
- 原始结果保存在任务工作区的 `results/`；
- `REPORT.md` 区分实测事实、推断和待验证项；
- `python3 scripts/validate_repository.py` 通过；
- 修改范围没有污染原始课件或来源记录。

如果当前机器没有目标硬件，应交付可运行代码、测试矩阵、待执行命令和限制说明，状态写成“实现完成，硬件验证待执行”。

## 推荐输出格式

AI 助手每轮开发结束时报告：

1. 修改了哪些文件；
2. 哪些正确性用例已运行；
3. 哪些性能或 profiler 命令已运行；
4. 关键结果与原始文件位置；
5. 当前限制、失败和下一项最小实验。

不要只给速度倍数、截图或“测试通过”。
