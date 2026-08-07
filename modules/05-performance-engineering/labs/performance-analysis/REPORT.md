# 模块五实验报告：性能分析与研讨

| 字段 | 内容 |
|------|------|
| 姓名 / 单位 | |
| 日期 | |
| 硬件 / 软件栈 | BI-V150；torch / triton / 驱动： |
| `LAB_DAY2_ROOT` | 已设置 / 未设置（内置 Triton） |

---

## 1. 测量结果摘要（贴数即可）

来源：`results/benchmark_*.json`、`roofline.json`。

### Offline（建议 1024³ fp16）

| 实现 | median ms | p99 ms | samples/s | TFLOP/s |
|------|-----------|--------|-----------|---------|
| torch | | | | |
| baseline | | | | |
| autotune | | | | |

### Server

| 实现 | median ms | p99 ms | samples/s | TFLOP/s | 备注 |
|------|-----------|--------|-----------|---------|------|
| torch | | | | | |
| baseline | | | | | |
| autotune | | | | | |

Roofline：π = ______ TFLOP/s，β = ______ GB/s，I ≈ ______ FLOP/Byte；图路径：`results/roofline.png`（若有）。

Profiling：已使用的工具（勾选）□ ixsmi  □ ixsys  □ ixkn-cli

---

## 2. 开放研讨题

请用**本机实测 + 推理**作答，不必复述教材定义。允许「数据不足以支撑某结论」——请说明还缺什么实验。

### Q1. 场景与「快」的定义

Offline 与 Server 下，若实现排序或相对差距发生变化，你如何解释？
在什么应用假设下，你更应优化 median，而不是 p99（或相反）？

### Q2. Roofline 模型的边界

本实验的 Bytes 模型为 \((MK+KN+MN)\times sizeof(dtype)\)。
它可能**系统性高估或低估**真实片外流量的哪些成分？若 Triton 与 `torch.matmul` 的 \(I\) 相同但 TFLOP/s 差一个数量级，仅凭 Roofline **不能**推出哪些结论？

### Q3. 与 Lab1 调优叙事的衔接

Lab1 通过改 tile / autotune 提升 Triton。结合本模块数据：autotune 相对 baseline 的收益，与相对 `torch.matmul` 的差距，哪个更大？
这对「继续搜 tile」vs「换算法/融合/调用 vendor」的优先级意味着什么？

### Q4. 对照实验设计

若只能再跑 **一组** 对照（改一个因素：shape、dtype、tile、或去掉 sync 等），你选什么？
预期 Offline 指标与 Roofline 落点如何变？如何避免把 JIT / 测量噪声当成「优化效果」？

### Q5. Profiler 与 LoadGen 冲突时

若 `ixsmi` 显示 GPU 利用率不高，但 LoadGen 的 TFLOP/s 已接近你对实现的预期，或二者方向相反——你更信任哪条证据链？下一步会补测什么？

### Q6.（可选）迁移与可推广性

同一套方法迁到另一加速器时，哪些结论可能保持（测量流程、问题分层），哪些必须重测（峰值、带宽、vendor 库形态）？

---

## 3. 简短结论（半页以内）

用 3–5 句话概括：当前 SUT 相对 vendor 的位置、你认为的主瓶颈类别、以及你最想做的下一步验证。
