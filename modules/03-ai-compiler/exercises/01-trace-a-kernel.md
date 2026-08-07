# 练习：追踪一个 Triton kernel

## 任务

从模块 1 的 Vector Add、GEMM，或模块 2 的 Fused Add + RMSNorm 中任选一个 kernel，建立编译链证据表。

| 层次 | 要找到的对象 | 证据 |
|---|---|---|
| Python/Triton | grid、pointer、mask、reduction 或 `tl.dot` | 文件与行号 |
| TTIR | 对应 op、shape、dtype、常量与布局 | IR 片段或课件页 |
| TTGIR | 线程/warp 映射、shared memory、pipeline | IR 片段或说明 |
| LLVM/目标层 | load/store、计算与同步痕迹 | 输出片段或工具说明 |

## 分析问题

1. 哪些 Python 参数在 JIT 时变成编译期常量？
2. mask 在 lower 后如何保护尾块？
3. 修改一个 `BLOCK_*` 参数，grid、布局或循环发生什么变化？
4. 找出三项 pass 或 lowering 变化，说明它们是否保持数值语义。
5. 编译器观察与模块 1 的性能 sweep 能否互相解释？

## 提交格式

提交 `compiler-trace.md`，包含环境版本、使用命令、四层映射表、关键片段和结论。不能获取某层输出时，写清失败命令、错误信息和替代证据。
