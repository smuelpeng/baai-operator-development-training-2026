# CoreX-Triton Lab Day 2

Day 2 延续 Day 1 的 tiled GEMM。今天不再学习新的 Triton 语法，而是通过可信测量调整 `BLOCK_M`、`BLOCK_N`、`BLOCK_K`，建立对 BI-V150 上 tile、并行度、数据复用和资源占用的实证认识。

---

## 项目背景：为什么必须在 BI-V150 上重新调优

AI 系统通过 CPU 与 GPU、NPU 等加速器协同执行。对 GEMM 这类并行算子，
源码正确只是第一步；工作分解、数据复用、并行度和片上资源压力共同决定
kernel 在目标硬件上的实际性能。

本实验运行于天数智芯天垓 150（BI-V150）GPU，软件路径可以简化为：

```text
Python / PyTorch / Triton
          ↓
CoreX-adapted framework and JIT compiler
          ↓
CoreX runtime and driver
          ↓
BI-V150 GPU (ivcore11, 64-thread warp)
```

CoreX 提供 `cuda:0` 和 `torch.cuda` 兼容接口，但 BI-V150 使用自己的架构、
指令和编译后端。因此：

- NVIDIA、Ascend 或其他平台的最佳 tile 不能直接照搬；
- 32-thread warp 等 NVIDIA 假设不适用于 BI-V150；
- 相同 Triton 源码在不同后端上可能得到不同的资源占用和性能；
- 正确性可移植、源码可移植和性能可移植是三个不同层次。

Triton 将 GEMM 的 block dimensions 暴露为编译期 meta-parameters。改变
`BLOCK_M/N/K` 不改变矩阵乘法的数学定义，却会改变每个 program 的输出
tile、grid 规模、K-loop 次数、数据复用和资源需求。Day 2 因此采用：

> **Hypothesis → Experiment → Data → Refined understanding → Code change**

完成本实验后，你应能够：

1. 建立包含 warmup、同步、重复样本和 median 的可信 benchmark；
2. 使用控制变量方法分析 BM、BN 和 BK；
3. 区分编译失败、结果错误、正确但慢和正确且快；
4. 从手工 sweep 构造有限且有依据的 autotune 空间；
5. 解释不同 shape 为什么可能选择不同配置；
6. 测量 autotune 相对固定 `32×32×32` Triton baseline 的提升；
7. 区分平台事实、实验假设和当前实测结论。

---

## Quick Start

```bash
cd modules/01-ai-system-and-triton/labs/gemm-tuning
bash setup.sh
python3 00_baseline.py
python3 01_block_sweep.py
python3 02_autotune.py
python3 03_summary.py
```

请按顺序运行。完成 Step 01 和 Step 02 中的 TODO 后再执行 summary。
学生只修改 `>>> YOUR CODE HERE >>>` 与 `<<< END OF YOUR CODE <<<` 之间的内容。

---

## What You'll Do

| 文件 | 你要完成的内容 | 建议时间 |
|---|---|---:|
| `00_baseline.py` | 校准 1024³/2048³ 的固定 Triton baseline | 15 min |
| `01_block_sweep.py` | 运行资源压力 probe，设计 10–20 个控制变量 tile | 40 min |
| `02_autotune.py` | block autotune、同配置复核、BK 与 scheduler sweep | 55 min |
| `03_summary.py` | 对比未优化 baseline 与 autotuned kernel | 20 min |
| `REPORT.md` | 记录环境、全部配置、结果和解释 | 贯穿全程 |

`common.py` 是已经完成的 Day 1 GEMM 与 benchmark helper，学生无需修改。

详细的“假设 → 实验 → 数据 → 修正认识”叙事见 [`handout.md`](./handout.md)。

---

## The Story

Day 1 留下了一个问题：正确的 tiled GEMM 是否已经使用了适合当前 shape 和硬件的 block 配置？如果没有，autotune 能把固定 baseline 提升多少？

Day 2 将检验三个命题：

1. **基线必须可信。** warmup、同步、重复采样和 median 会改变我们是否相信一个“优化”。
2. **tile 是多目标权衡。** 更大的 tile 可能提高复用，也可能降低并行度、增加寄存器或片上资源压力。
3. **最优配置依赖 shape。** 方形、M/N 不对称、K 较小的矩阵可能选择不同的 `BLOCK_M/N/K`。

本实验不会把原 Ascend Day 2 的 Cube、L0C、L1/cbuf 容量直接迁移到 BI-V150。ivcore11 是不同架构；我们迁移的是测量方法，不是假定的容量数字。

实验主循环是：

> **Hypothesis → Experiment → Data → Refined understanding → Code change**

---

## How This Lab Works

1. Step 00 只运行并记录，确认 baseline 正确且计时稳定。
2. Step 01 先用 stress tile 保存 CoreX 编译证据，再增加分别改变 BM、BN、BK 的候选。
3. 每个候选分别判定编译/运行失败、错误结果和性能结果。
4. Step 02 先固定 scheduler 调 block，再固定 winner 做 BK 和 `num_warps/num_stages` sweep。
5. Step 03 使用相同输入、输出和计时口径，对比未优化 fixed baseline 与 autotuned kernel。

调优采用分阶段控制变量，而不是假设 CoreX 会忽略 scheduler 参数。

---

## Environment

与 Day 1 相同：

| 组件 | 实验要求 |
|---|---|
| Hardware | 天垓 150（BI-V150），ivcore11 |
| Warp | 64 threads |
| Device API | `cuda:0` / `torch.cuda` CoreX 兼容接口 |
| PyTorch | 平台适配版；CoreX 4.4 镜像可能显示普通 `2.7.1` 版本号 |
| Triton | 平台适配版；结合模块路径和 CoreX 运行时路径核验 |
| Input/output dtype | fp16 |
| Accumulator | fp32 |

如果 `bash setup.sh` 能打印设备名，并且 Day 1 的 `00_check_env.py` 已通过，就可以开始。

---

## Measurement Rules

- correctness check 永远先于 benchmark；
- 首次 JIT/autotune 编译不计入 steady-state；
- 计时边界必须调用 `torch.cuda.synchronize()`；
- 每个样本包含多次 inner iteration，最终报告 median；
- baseline 与 autotuned kernel 使用相同输入、输出 dtype 与 shape；
- 分别记录包含 allocation 的端到端计时和复用输出的 kernel-focused 计时；
- 对 fixed/autotuned 对照显式匹配 block、`num_warps`、`num_stages`；
- 不根据单次或单个 shape 的结果下结论；
- 保存失败配置及错误摘要，不只保存最快结果。

平台建议全局内存访问按 128 字节对齐。对 fp16 连续数据，64 个元素等于 128 字节；这是候选 `BLOCK_N`/`BLOCK_K` 的一个假设来源，但不是预先成立的性能结论。

---

## Common Issues

| 现象 | 可能原因 | 处理方式 |
|---|---|---|
| baseline 多次运行波动很大 | shape 太小、系统有其他负载或样本不足 | 保留脚本的 warmup/samples/inner，重跑并报告 median |
| Step 01 某个 tile 编译失败 | tile 超过当前 compiler/hardware 资源限制 | 记录配置和错误，继续测试其他配置 |
| Step 01 PASS 但性能更差 | tile 复用收益小于并行度或资源代价 | 对照只改变一个维度的候选解释 |
| Step 02 首次运行很慢 | autotune 正在编译并测量多个 config | 等待完成，不把编译时间当 kernel latency |
| autotune 因某一配置中止 | 搜索空间包含已知不合法或过大的 tile | 先用 Step 01 筛选，再缩小 config list |
| fixed 与 autotuned 同 tile 仍差很多 | scheduler metadata、独立 JIT 或测量波动 | 使用 Step 02 的 exact-config 对照并重复测量 |
| 不同 shape 的 winner 不同 | 工作分解、边界浪费与 K-loop 次数不同 | 这是预期结果，分别记录并解释 |
| 版本中没有 `+corex` | CoreX 4.4 镜像可能省略版本后缀，也可能误装了公开 wheel | 不要升级；运行 Day 1 环境检查，结合 `torch.__file__`、`COREX_HOME` 和库路径核验 |

---

## Resources

- Day 1 学生实验：https://gitee.com/sunxt-0719/lab1-day1
- BI-V150 平台说明：https://moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu
- Triton autotune API：https://triton-lang.org/main/python-api/generated/triton.autotune.html
- Triton Config API：https://triton-lang.org/main/python-api/generated/triton.Config.html
- Triton Matmul 教程：https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- 原 Ascend Day 2：https://gitee.com/jieran-zhang/lab_day2

---

## License

实验代码沿用原材料的 MIT 使用约定。
