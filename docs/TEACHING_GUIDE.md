# 教师授课指南

## 组织原则

整套课程围绕同一条证据链组织：学生在模块 1 写出 GEMM，在模块 2 学习融合与框架接入，在模块 3追踪编译链，在模块 4理解多卡执行，模块 5再对同一 workload 做性能诊断。每一章都要回到一个可观察的对象，避免课程退化为组件名词表。

## 模块 1：让学生先相信“结果是对的”

讲授重点：host/device、program/grid、offset/mask、reduction、tiled GEMM、fp32 accumulation。课堂可以用非整 tile 的向量长度追问 mask 缺失后会发生什么。

实验组织：先完成 Vector Add，再运行 Softmax、LayerNorm 和 GEMM；Day 2 使用固定 baseline、控制变量 sweep 和有限 autotune。要求保存失败配置。

常见误区：把 `cuda:0` 当成 NVIDIA 硬件；未经同步就计时；把其他平台的最优 tile 直接照搬；只测整 tile shape。

## 模块 2：从访存流量解释融合

讲授重点：Attention/FlashAttention 的 IO 视角、算子融合收益与资源压力、Fused Add + RMSNorm、PyTorch Dispatcher 和 Meta/Autograd 注册。

课堂问题：未融合实现读写多少个元素？融合后减少了哪些中间张量？一个越来越大的融合 kernel 为什么可能变慢？

实验组织：学生只看学生版；教师在验收或卡住后再开放参考版。前向正确后再进入反向和注册，最后做 benchmark。

常见误区：忽略 bf16 内部归约精度；只验证 forward；把 graph capture 成功等同于 Inductor 全链路性能成功。

## 模块 3：用一条算子路径教编译器

讲授重点：动态图捕获、TorchDynamo/AOTAutograd/Inductor、DSL、MLIR 基础结构、TTIR/TTGIR/LLVM lowering。以模块 1 或 2 的 kernel 为样本，将源代码行、IR op、布局和目标代码串起来。

课堂问题：`BLOCK_M/N/K` 在源码里是参数，在 IR 和生成代码中分别改变了什么？canonicalization、layout propagation、shared-memory promotion、pipelining 各解决哪类问题？

验收方式：要求学生给证据页和对照表，不要求背诵全部 pass 名称。无法导出 IR 时可基于课件和 FlagPrism 源码完成静态追踪，并明确未实测。

## 模块 4：先写假设，再套公式

讲授重点：DP/DDP、ZeRO/FSDP、TP、PP 与 3D 并行；collective 语义；拓扑；Ring、Tree、halving-doubling；α-β 模型。

课堂问题：模型参数、梯度和优化器状态分别放在哪里？Ring AllReduce 的 ReduceScatter 和 AllGather 阶段如何移动数据？增加 rank 后 α 项和 β 项如何变化？

常见误区：把课件中的消息大小阈值当成硬件无关定律；忽略拓扑与链路并发；只报通信量，不报轮数和启动时延。

## 模块 5：让三类证据互相校验

讲授重点：延迟/吞吐/TFLOP/s、Roofline、ridge point、occupancy/warp state/memory、Autotuning/KernelGen、Amdahl 定律。FlagPrism 用于说明如何把源语句、IR 与设备事件重新对应。

实验组织：固定同一 SUT 和输入，依次运行 Benchmark、Roofline、Profiling。学生先写出瓶颈假设，再打开 profiler 结果；如果证据冲突，要修正假设。

常见误区：把高 arithmetic intensity 直接写成“已经达到计算峰值”；混用 end-to-end 与 kernel-only 延迟；局部 kernel 加速后直接声称模型端到端同倍数提升。

## 教师验收提问

每组答辩至少回答：

1. 你如何证明结果正确？
2. 计时边界在哪里，编译与分配是否计入？
3. 为什么选择这些配置或通信算法？
4. 哪一份原始证据支持你的瓶颈判断？
5. 换设备、shape 或软件版本后，哪部分需要重测？

详细评分见 [作业与验收](ASSESSMENT.md)。
