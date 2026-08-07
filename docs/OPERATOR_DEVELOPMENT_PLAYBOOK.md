# AI 辅助算子开发手册

## 0. 选定任务类型

| 类型 | 典型目标 | 主要产物 |
|---|---|---|
| 新 kernel | 从参考函数实现 Triton/CUDA 风格算子 | kernel、wrapper、正确性测试 |
| 融合算子 | 减少中间张量和 launch | 前向/反向、框架注册、性能对照 |
| 算子调优 | 改 tile、schedule、pipeline | sweep、autotune、winner 解释 |
| 编译问题 | graph break、IR、lowering 异常 | 最小复现、IR 对照、定位证据 |
| 性能诊断 | 已正确但速度不理想 | Benchmark、Roofline、Profiler 报告 |
| 分布式算子 | collective 或并行策略 | 通信模型、实现、拓扑实测 |

## 1. 建立工作区

```bash
python3 scripts/new_operator_task.py my-operator
cd workspaces/my-operator
```

先填写 `TASK.md`。AI 可以协助补全缺失项，但 shape、dtype、输出语义、误差和 inplace 行为需要由任务负责人确认。

## 2. 让 AI 先做设计审查

把下面内容作为第一轮请求：

```text
阅读仓库根目录 AGENTS.md、docs/AI_CONTEXT.md、docs/KNOWLEDGE_INDEX.md，
再阅读当前 TASK.md。先不要优化。
请输出：算子合同、边界条件、参考实现方案、正确性测试矩阵、
最小 Triton grid/tile 设计、可能的精度和资源风险、验证命令。
存在合同歧义时逐项列出。
```

设计审查通过后再让 AI 修改代码。

## 3. 正确性闭环

推荐测试矩阵：

| 维度 | 至少覆盖 |
|---|---|
| shape | 1、tile-1、tile、tile+1、典型规模、压力规模 |
| dtype | 合同中的全部 dtype |
| 数值 | 随机、小值、大值、零；需要时包含 NaN/Inf 行为 |
| layout | contiguous；合同要求时加入转置/stride |
| 梯度 | 每个可微输入；必要时与高精度 autograd 对照 |

每次 kernel 改动后先运行最小正确性集。扩大 shape 和 autotune 之前再跑完整矩阵。

## 4. 优化闭环

一次只改变一类因素：

- tile：`BLOCK_M/N/K` 或单维 block；
- 调度：`num_warps`、`num_stages`；
- 算法：融合、重计算、分块归约；
- 内存：访问顺序、连续性、shared memory；
- 框架：减少 graph break、分配或 dispatch 开销。

每一轮记录：假设、改动、正确性、性能、失败配置、修正后的认识。最佳配置需要用独立的 exact-config 测量复核。

## 5. 性能诊断闭环

1. 用相同输入比较 reference、baseline 和 candidate；
2. 计算 FLOPs、理想 Bytes 和 arithmetic intensity；
3. 用 Roofline 形成初步瓶颈假设；
4. 使用 profiler 检查 occupancy、warp stall、instruction 与 memory；
5. 如果 profiler 与模型冲突，保留冲突并设计最小补测；
6. 用 Amdahl 定律评估局部收益能否影响端到端。

## 6. 交付

工作区至少包含：

```text
TASK.md                 # 已确认合同
kernel.py               # 参考、kernel、wrapper
test_operator.py        # 正确性矩阵
benchmark.py            # 公平性能对照
REPORT.md               # 事实、推断、限制与复现命令
results/                # JSON、CSV、trace、日志
```

运行：

```bash
python3 scripts/validate_repository.py
```

目标平台结果尚未取得时，在 `REPORT.md` 明确写出待运行命令和预期观察，不填写虚构数字。
