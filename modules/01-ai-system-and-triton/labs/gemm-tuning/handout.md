# Day 2 Handout：在天垓 150 上调优 Triton GEMM Block

**课程：** AI 系统软件基础与异构计算

**实验时长：** 约 2 小时

**目标平台：** 天数智芯天垓 150（BI-V150）
**实验产出：** block sweep 数据、autotune winner、baseline/autotune 汇总表与实验报告

---

## Framing

Day 1 写出了正确的 tiled GEMM。今天要回答：

> 如果数学运算、dtype 和 kernel 源码都不变，仅改变 `BLOCK_M/N/K`，性能为什么会变化？

tile 同时影响：

- 每个 program 计算多少输出；
- A/B 数据在一个 tile 内被复用多少次；
- grid 中有多少 program 可以并行执行；
- K-loop 需要多少次迭代；
- accumulator、寄存器和其他片上资源压力；
- 边界 tile 浪费的计算比例；
- 连续内存访问是否符合硬件偏好的粒度。

本实验反复执行同一个方法：

> **Hypothesis → Experiment → Data → Refined understanding → Code change**

原 Ascend Day 2 使用编译错误推导 Cube/L0C/L1/cbuf 容量。BI-V150 是 ivcore11 架构，那些容量数字不能直接迁移。本版本保留“用实验修正硬件模型”的方法，重新从 block sweep 开始。

---

## Learning Goals

完成本实验后，你应能够：

1. 构造包含 warmup、同步、重复样本和 median 的 benchmark。
2. 用控制变量方法区分 BM、BN、BK 的影响。
3. 区分编译失败、结果错误、正确但慢、正确且快。
4. 从手工 sweep 设计有限且有依据的 autotune 空间。
5. 解释为什么最优配置依赖 `(M,N,K)`。
6. 把“文档建议”“实验假设”和“实测事实”分开表述。

---

## Step 00：Calibrate the Baseline（15 min）

运行：

```bash
python3 00_baseline.py
```

脚本使用固定 `32×32×32` tile，在 `1024³` 和 `2048³` 上完成：

1. 对 Triton 结果执行 correctness check；
2. warmup；
3. 多组计时样本；
4. 取 median latency；
5. 计算 effective TFLOP/s；
6. 分开报告端到端 wrapper 与预分配 output 的计时。

### 为什么使用 median

运行时调度、后台进程和首次触发的内部工作会产生长尾样本。平均值容易被少数慢样本拉高；median 更适合描述一组短 kernel 的典型 steady-state latency。

### TFLOP/s 的计算

矩阵乘法 `C[M,N] = A[M,K] @ B[K,N]` 约包含：

```text
2 * M * N * K FLOPs
```

若时间单位为毫秒：

```text
TFLOP/s = 2*M*N*K / (ms * 1e9)
```

这是 effective throughput，不等同于硬件标称峰值。

### 记录

在 `REPORT.md` 写下：

- baseline max diff；
- baseline 端到端与预分配输出两种口径的 median ms 与 TFLOP/s；
- 重复运行后的波动范围。

### 讨论问题

1. 为什么 correctness check 不能放到大量 benchmark 之后才做？
2. benchmark helper 的同步放在什么位置？
3. 端到端与预分配输出口径相差多少？为什么 block 调优应优先看后者？

---

## Step 01：Controlled Block Sweep（35 min）

打开 `01_block_sweep.py`。Part A 已提供定向 stress tile，用于保存 CoreX 编译/资源证据；Part B 要求保留 baseline，再增加 10–20 个候选。

### 资源 probe

脚本同时打印两个 footprint proxy：

```text
accumulator ≈ BM * BN * 4 bytes
operands    ≈ (BM + BN) * BK * 2 bytes
```

前者对应 fp32 accumulator，后者对应 fp16 A/B tile。它们用于设计控制变量，不等于已经证明的物理 buffer 容量。只有 CoreX 编译错误明确提供资源名称和边界时，才能进一步建立 BI-V150 legality predicate。

### 控制变量设计

不要直接罗列一堆“看起来合理”的配置。至少包含以下对照：

1. 固定 BN/BK，只增加 BM；
2. 固定 BM/BK，只增加 BN；
3. 固定 BM/BN，只增加 BK；
4. 一个更大的对称 tile；
5. 一对互为转置的非对称 tile。

这样才能回答“变化来自哪个维度”。

### 三类 shape

脚本覆盖：

| Shape `(M,N,K)` | 目的 |
|---|---|
| `(1024,1024,1024)` | 方形基线 |
| `(2048,512,1024)` | M 大、N 小 |
| `(512,2048,256)` | N 大、K 较小 |

对非对称矩阵，互换 BM/BN 不一定产生相同性能，因为 grid、边界浪费和 A/B 访问方向不同。

### 128 字节对齐假设

平台建议全局内存访问按 128 字节对齐。fp16 每元素 2 字节：

```text
64 fp16 elements = 128 bytes
```

因此 64 是值得测试的连续 tile 宽度，但请注意：

- 这是候选来源，不是“64 必然最快”的结论；
- stride、边界、编译器 lowering 和并行度同样影响结果；
- BM 对应的访问不一定在内存中连续，不能简单套用同一推理。

### 四类结果

对每个配置分类记录：

| 状态 | 含义 | 下一步 |
|---|---|---|
| Compile FAIL | compiler/backend 无法生成该配置 | 保存配置和错误摘要 |
| Wrong result | kernel 执行但误差超限 | 不进入性能搜索空间 |
| Correct but slow | 合法但权衡不佳 | 与控制变量配置比较原因 |
| Correct and fast | 当前 shape 的有效候选 | 进入 Step 02 |

运行：

```bash
python3 01_block_sweep.py
```

### 如何读结果

#### 增大 BM 或 BN

可能收益：

- 单个 A/B tile 服务更多输出；
- 提高数据复用；
- 减少 program 数量和 launch scheduling 开销。

可能代价：

- accumulator tile 增大；
- 单 program 资源压力增大；
- grid program 数减少，降低并行度；
- 不规则 shape 的边界浪费增加。

#### 增大 BK

可能收益：

- 减少 `ceil(K/BK)` 次 K-loop 迭代；
- 摊薄每次迭代的地址计算和 loop overhead。

可能代价：

- A/B operand tile 增大；
- 资源压力或编译难度上升；
- K 较小时，过大的 BK 产生更多 masked work。

### 讨论问题

1. BM 与 BN 的效果是否对称？用哪一对配置证明？
2. 不同 shape 的最佳手工 tile 是否相同？
3. 某配置失败时，你能否只根据异常名称断言具体硬件容量？为什么？
4. 对 K=256 的 shape，增大 BK 的收益是否和 K=1024 相同？

---

## Step 02：Block Autotune、BK 与 Scheduler（55 min）

打开 `02_autotune.py`，把单个 baseline config 替换为 Step 01 得到的 10–20 个 PASS 候选。

### 为什么不做盲目笛卡尔积

假设 BM、BN、BK 各有 6 个取值，笛卡尔积会产生 216 个配置。首次运行需要逐个编译和 benchmark，其中大量配置可能重复、无意义或直接失败。

一个好的教学搜索空间应：

- 包含 baseline；
- 包含 Step 01 的优胜配置；
- 保留至少一对非对称配置；
- 保留不同 BK 用于检验 K-loop 假设；
- 排除已知错误或编译失败的配置；
- 控制首次编译时间。

### Autotune key

本实验使用：

```text
key = [M, N, K]
```

当 shape key 改变时，autotuner 可以重新选择配置。原因是 tile 的边界浪费、grid 规模和 K-loop 次数都依赖 shape。

### 分阶段控制调度参数

本实验问题限定为 GEMM block dimensions。如果同时改变 block、`num_warps` 和 `num_stages`：

- 搜索空间迅速膨胀；
- 无法从数据判断哪个因素造成变化；
- 32-thread warp 的 NVIDIA 经验不应未经验证套到 64-thread warp 平台。

第一阶段所有 block config 都显式使用相同的 `num_warps=4`、`num_stages=2`，从而隔离 block dimension。选出 block winner 后，第二阶段固定 block，分别 sweep：

```text
num_warps  = 1, 2, 4, 8
num_stages = 1, 2, 3, 4
```

失败组合也要记录。这样可以检验 CoreX backend 是否使用这些参数，而不是未经实验假设它们无效。

### Exact-config 对照

autotune 选出 winner 后，脚本会把 winner 的 BM/BN/BK、`num_warps`、`num_stages` 全部传给 fixed kernel。若 exact-fixed 与 decorated-autotune 仍显著不同，应考虑独立 JIT codegen、频率变化或测量噪声，不能把差异归因于 block。

### BK sweep

脚本固定 winner 的 BM/BN，测试 `BK = 16, 32, 64, 128, 256`，同时输出 K-loop 次数、operand footprint、性能和近似 arithmetic intensity。

运行：

```bash
python3 02_autotune.py
```

### 需要记录

对每个 shape 保存：

- correctness max diff；
- `best_config`；
- median ms；
- effective TFLOP/s；
- 与 Step 01 最佳手工 tile 是否一致。
- exact-fixed/autotuned ratio；
- BK 曲线；
- scheduler sweep winner。

### 讨论问题

1. autotune winner 与手工 sweep winner 不一致时，可能有哪些原因？
2. 为什么首次 autotune 总耗时不能作为 kernel latency？
3. 将 key 只设为 `M` 会有什么风险？
4. config 越多是否一定得到更好的最终性能？

---

## Step 03：Summary（20 min）

运行：

```bash
python3 03_summary.py
```

脚本比较：

1. 未优化的 fixed baseline `32×32×32`；
2. 使用 Step 02 搜索空间得到的 autotuned kernel。

二者使用相同输入、输出 dtype、预分配输出和 benchmark helper。最终汇总只回答：

> 在当前 shape 和当前 BI-V150 软件栈上，autotune 相对固定 baseline 提升了多少？

### 正确解读汇总表

- `vs base > 1`：相对 baseline 更快；
- autotune 快于 baseline：说明配置搜索在当前 shape 上有效；
- autotune 未提升：检查搜索空间是否包含 Step 01 的有效候选，并排查缓存、测量噪声和 winner；
- 某一 shape 的 winner 不能直接推广到全部模型 workload。

Step 02 的 exact-config fixed/autotuned ratio 仍用于检查完整 launch metadata 和测量一致性，但它是诊断项，不是最终性能对比对象。

不要预填“autotune 应达到几倍加速”。结论必须来自你当前 CoreX/PyTorch/Triton 版本和 BI-V150 实测。

---

## What We Did Not Tune

为了保持因果关系清晰，本实验没有调整：

1. program-id grouped ordering；
2. input layout 或 transpose；
3. fp8/bf16 或量化 dtype；
4. bias/activation fusion；
5. persistent GEMM；
6. 多卡并行。

这些都可以成为后续实验，但应一次只引入一个新问题。

---

## The Transferable Pattern

无论目标是 BI-V150、Ascend、NVIDIA 还是其他加速器，都可以复用以下流程：

1. **固定正确性口径。** 明确 shape、dtype、reference 和 tolerance。
2. **建立可信基线。** warmup、同步、重复采样、报告 median。
3. **提出可证伪假设。** 例如“更大的 BN 因连续访问而更快”。
4. **控制变量实验。** 一次只改变一个核心因素。
5. **记录所有结果。** 失败和慢配置同样帮助界定搜索空间。
6. **缩小搜索空间。** 把实测合理候选交给 autotune。
7. **跨 shape 验证。** 避免只优化一个方形 benchmark。
8. **区分事实层级。** 文档事实、代码推断、实验观察不要混写。

这比记住某个“最佳 block”更重要。

---

## Completion Checklist

- [ ] Step 00 baseline 正确并完成 median 测量。
- [ ] Step 01 保存 stress probe 的完整 CoreX 错误，并包含 10–20 个性能候选。
- [ ] 三类 shape 的所有 PASS/FAIL 均写入报告。
- [ ] Step 02 搜索空间来自 Step 01，而非盲目笛卡尔积。
- [ ] 每个 shape 的 `best_config` 已记录。
- [ ] BK sweep 与 scheduler sweep 已记录。
- [ ] exact-fixed/autotuned ratio 已解释。
- [ ] Step 03 已汇总未优化 baseline 与 autotuned 两项结果。
- [ ] autotune 相对 baseline 的加速比已记录并正确解释。
- [ ] 回答 `REPORT.md` 的全部结论问题。

完整参考代码位于教师管理的 `solution/day2`，不附在学生 handout 中。
