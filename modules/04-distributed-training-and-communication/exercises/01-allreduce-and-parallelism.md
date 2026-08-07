# 练习：AllReduce 与并行策略

## A. Ring AllReduce 推演

选择 4 或 8 个 rank，将张量切成 N 块：

1. 逐轮画出 ReduceScatter 的发送块、接收块和归约位置；
2. 逐轮画出 AllGather 的块传播；
3. 核对总轮数、单 rank 数据量和最终结果；
4. 代入两组消息大小，比较 α 项和数据传输项。

## B. Tree、Ring 与 halving-doubling

针对小、中、大三种消息，填写：

| 消息规模 | 候选算法 | 主要理由 | 需要实测的变量 |
|---|---|---|---|
| 小 |  |  |  |
| 中 |  |  |  |
| 大 |  |  |  |

课件中的 1 MB、16 MB、64 MB 范围是经验提示。报告中应设计一次实测来确认当前硬件的切换点。

## C. 并行策略设计

为教师给定的 Transformer 规模选择 DP、TP、PP、ZeRO/FSDP 组合，回答：

- 参数、梯度、优化器状态和 activation 分别占多少显存？
- 哪些 collective 在机内，哪些跨节点？
- pipeline bubble 与 activation memory 如何变化？
- 最可能的通信瓶颈是什么，如何用数据验证？

提交一份计算表和 500–800 字策略说明。
