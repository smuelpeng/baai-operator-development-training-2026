# 模块 2：高性能 AI 算子与算子工程

## 核心问题

如何从算法访存路径、kernel 融合和框架注册三个层次，把一个算子做成可用、可微、可编译且可评测的系统组件？

## 学习目标

- 从 IO 复杂度理解 FlashAttention；
- 计算算子融合前后的中间张量与全局内存读写；
- 识别融合带来的带宽收益、launch 减少和资源压力；
- 为 Fused Add + RMSNorm 实现前向、反向与 PyTorch Dispatcher 注册；
- 使用 Meta/Autograd 实现与 `torch.compile` 检查框架兼容性。

## 主案例

```text
z = x + residual
rrms = 1 / sqrt(mean(z * z) + eps)
out = z * rrms * weight
```

未融合路径会把中间结果多次写回并读出全局内存。融合 kernel 将加法、归约、缩放和写回合并，但仍需关注寄存器、占用率、指令缓存和调度压力。

## 课件与实验

- [模块二主课件](../../materials/courseware/module-02/0805-课件-模块二：高性能AI算子与算子工程.pdf)
- [学生版实验](labs/fused-rmsnorm-student/)
- [参考实现](labs/fused-rmsnorm-reference/)

实验顺序：前向 → 反向 → Dispatcher/Meta/Autograd → `torch.compile` → Benchmark。参考实现只用于验收和讲评，不应在学生动手前分发。

## 验收

- float32 前向与梯度误差低于实验指定阈值；
- bf16 保持输出 dtype，内部归约使用 fp32；
- Dispatcher 注册完整，Dynamo 捕获结果可解释；
- 对照 PyTorch、拆分 Triton 和 FlagGems 时使用相同 shape、dtype 和计时口径。
