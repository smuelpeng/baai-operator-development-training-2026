# 模块 3：AI 编译器原理与优化

## 核心问题

一个 PyTorch/Triton 算子经过哪些中间表示和优化，最终成为设备可执行代码？

## 编译链

```text
PyTorch program
  └─ TorchDynamo：捕获计算图
      └─ AOTAutograd / AOTDispatcher：前后向与算子语义
          └─ TorchInductor：生成后端程序
              └─ Triton Python / AST + JIT specialization
                  └─ TTIR → TTGIR → LLVM/NVVM → 目标代码
```

课件还以 TileLang 说明更高层 tile 抽象、内存层次、layout 与调度表达；用 MLIR 的 module、region、block、operation、attribute 和 dialect 解释分层 IR。

## 学习重点

- 静态图、动态图和 `torch.compile` 的职责边界；
- DSL 为何需要暴露 tile、layout、memory 和 schedule；
- TTIR 的 canonicalization、constant folding、dead-op removal 与 layout propagation；
- TTGIR 的线程映射、shared-memory promotion、pipelining 与 tensor-core pattern；
- `BLOCK_M/N/K`、`num_warps`、`num_stages`、grid 和 dtype 如何影响 lowering。

## 课件与练习

- [模块三授课版](../../materials/courseware/module-03/0806-课件-授课版本-模块三：AI%20编译器原理与优化.pdf)
- [模块三预习版](../../materials/courseware/module-03/0806-课件-预习版本-模块三：AI%20编译器原理与优化.pdf)
- [官方 32 页可编辑 PPTX](../../materials/courseware/module-03/official-editable/课件-模块三：AI%20编译器原理与优化.pptx)
- [官方上游错标的 34 页可编辑版本](../../materials/courseware/module-03/official-editable/upstream-mislabeled/课件-模块四：分布式并行训练.pptx)：文件名写模块四，逐页内容实际为 AI 编译器
- [练习：追踪一个 Triton kernel](exercises/01-trace-a-kernel.md)
- [官方 Triton IR 实验候选快照](labs/official-triton-ir/)

授课版比预习版多 2 页，增加 FlagGems 与 SGLang Triton kernel 等源码入口。官方 IR 实验只完成源码归档和静态检查；上游 BI-V150 文字与随仓 H200 结果冲突，必须重新生成本机结果后才能用于验收。

## 验收

提交一份 lowering 地图，至少包含源码、TTIR、TTGIR 和目标 IR/代码四层；标出三处优化或布局变化，并说明每一处的语义与性能目的。无法使用目标平台导出 IR 时，提交静态阅读结果并标为“未运行验证”。
