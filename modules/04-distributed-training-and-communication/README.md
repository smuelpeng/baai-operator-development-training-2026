# 模块 4：分布式并行训练与通信

## 核心问题

当模型和数据进入多卡环境，如何在显存、通信量、启动时延、带宽、拓扑和流水空泡之间选择并行策略？

## 两条主线

### 并行训练

- DP/DDP：副本计算与梯度 AllReduce；
- ZeRO-1/2/3 与 FSDP：依次切分优化器状态、梯度和参数；
- TP：按列/行切分 MLP 和 Attention，常放在高带宽机内；
- PP：GPipe、1F1B、interleaved、zero-bubble、DualPipe；
- 3D 并行：TP × PP × DP，并可继续加入 context/expert parallelism。

### 集合通信

- 原语：Broadcast、Reduce、Gather、Scatter、AllReduce、AllGather、ReduceScatter、AlltoAll；
- 拓扑：Clos、Torus、full mesh、Dragonfly；
- 算法：Tree、Ring、halving-doubling；
- 模型：用 α 表示启动时延，用 β 相关项表示每字节传输成本。

Ring AllReduce 的课件口径为：

```text
单 rank 数据量 = 2(N-1)K/N
时间 = 2(N-1)α + 2(N-1)K/(Nβ)
```

公式符号必须沿用当前课件定义；使用其他教材时要先核对 β 表示带宽还是单位字节成本。

## 课件与练习

- [分布式并行训练](../../materials/courseware/module-04/0806-课件-预习版本-模块四：分布式并行训练.pdf)
- [分布式通信原语授课版](../../materials/courseware/module-04/0807-课件-授课版本-模块四：分布式通信原语.pdf)
- [通信原语预习版](../../materials/courseware/module-04/0807-课件-预习版本-模块四：分布式通信原语.pdf)
- [通信原语官方可编辑 PPTX](../../materials/courseware/module-04/official-editable/课件-模块四：分布式通信原语.pptx)
- [练习：AllReduce 与并行策略](exercises/01-allreduce-and-parallelism.md)
- [官方推理部署扩展](labs/official-inference-deployment/)：Qwen3-4B、vLLM、FlagGems；不替代通信实验

授课版通信课件比预习版多 10 页，新增 Broadcast 下界、树形广播、分块流水化和 simultaneous trees。官方发布版中名为“分布式并行训练”的 34 页 PPTX 实际全部是 AI 编译器内容，已隔离到模块三；当前没有 61 页分布式训练 PDF 的官方可编辑原件。

## 验收

学生需画出 4 或 8 rank Ring AllReduce 两阶段的块流向，计算一组 α-β 成本，并为一个模型给出 DP/TP/PP/ZeRO 组合。报告必须注明模型规模、序列长度、卡数、显存、拓扑和所有带宽假设。
