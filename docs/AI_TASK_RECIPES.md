# AI Coding 任务配方

下面的请求可以直接复制给编码助手。把 `<task>` 替换为工作区路径。

## 配方 1：从数学定义写 Triton kernel

```text
阅读 AGENTS.md、docs/AI_CONTEXT.md、docs/KNOWLEDGE_INDEX.md 和 <task>/TASK.md。
先检查算子合同是否足以确定输出；列出 shape/dtype/边界和精度风险。
实现直观 PyTorch reference 与测试矩阵。reference-only 测试通过后，
参考模块 1 的 Vector Add、归约或 GEMM 代码实现最小 Triton kernel。
先处理非整 tile 和 mask，再考虑性能。把运行命令和限制写入 REPORT.md。
```

## 配方 2：实现融合算子的前向、反向和 PyTorch 注册

```text
阅读模块 2 的学生版 Fused Add + RMSNorm，但不要复制参考版答案。
根据 <task>/TASK.md 分三步工作：前向正确性、反向梯度、Dispatcher/Meta/compile。
每完成一步运行对应测试并汇报误差。确认 eager 与 backward 后，
检查 torch.compile 图数量和 graph break。最后才比较未融合、融合与库实现性能。
```

## 配方 3：调优一个已经正确的 kernel

```text
当前 kernel 已通过正确性门槛。请参考 gemm-tuning 的方法，
先建立固定 baseline 和可信 bench，再设计控制变量 sweep。
候选需覆盖 PASS/FAIL，记录资源错误；不要直接给一个经验配置。
从 sweep 结果构造小型 autotune 空间，对 winner 做 exact-config 复核。
输出假设—实验—数据—修正认识，并保存原始结果。
```

## 配方 4：诊断“正确但很慢”的算子

```text
阅读模块 5 的性能分析实验。固定 SUT、shape、dtype 和计时边界，
先复核 correctness、warmup、同步和统计方式。运行 Benchmark 定量差距，
计算 FLOPs/Bytes/arithmetic intensity 形成 Roofline 假设，
再列出需要从 profiler 获取的 occupancy、warp、instruction、memory 或 timeline 证据。
若当前没有 BI-V150，只生成可运行命令和结果 schema，不编造数字。
```

## 配方 5：定位 `torch.compile` 或 lowering 问题

```text
建立最小复现，区分 eager 语义、Dispatcher/Meta、Dynamo 捕获、Inductor/Triton lowering。
参考模块 3 的编译链练习，把源语句、图、TTIR、TTGIR 和目标层证据放在同一表中。
一次只改变一个注册、shape branch、编译期参数或 pass 条件。
报告明确问题发生在哪一层，以及证据尚未覆盖哪一层。
```

## 配方 6：审查 AI 生成的算子实现

```text
不要直接重写代码。按 AGENTS.md 的完成定义做审查：
核对数学合同、mask/单位元、dtype promotion、stride/contiguous、输出别名、梯度、
JIT 与计时边界、baseline 公平性和证据文件。
按严重度列出会导致错误结论的问题，并给出最小复现用例。
```
