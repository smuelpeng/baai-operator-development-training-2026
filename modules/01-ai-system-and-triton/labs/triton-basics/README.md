# AI 系统软件基础与异构计算：BI-V150 Triton 试听实验

## 实验介绍

本仓库面向课程试听学习者，提供一套可在天数智芯 BI-V150 上运行的
Triton 两日实验。大部分基础代码已经完成，只保留两个核心编程任务：

1. `day1/02_vector_add_triton.py`：补全 Vector Add 的 `add_kernel`；
2. `day2/02_autotune.py`：根据 sweep 结果补全 `AUTOTUNE_CONFIGS`。

需要填写的代码范围由以下标记限定：

```python
# >>> YOUR CODE HERE >>>
...
# <<< END OF YOUR CODE <<<
```

参考答案分别放在 `day1/README.md` 和 `day2/README.md` 的结尾。建议先独立
完成 TODO 并观察运行结果，再对照参考答案。

## 学习目标

完成实验后，可以了解：

- CPU host 与 GPU device 的基本分工；
- Triton program、grid、offset、mask 和 block 的含义；
- Softmax、LayerNorm 和 tiled GEMM 的基本实现方法；
- GEMM 中的 K-loop、fp16 输入/输出和 fp32 accumulator；
- warmup、同步、正确性检查和稳定计时；
- block sweep、scheduler sweep 与 Triton autotune 的作用。

## BI-V150 与 CoreX 软件栈

天垓 150（BI-V150）是面向 AI 计算的 GPU 加速器。本实验使用以下软件路径：

```text
+------------------------------------------------------+
| Python application                                  |
| PyTorch operators / Triton kernels                   |
+------------------------------------------------------+
| CoreX-adapted PyTorch and Triton                     |
+------------------------------------------------------+
| CoreX compiler, runtime and optimized libraries      |
+------------------------------------------------------+
| Driver and device management                         |
+------------------------------------------------------+
| BI-V150 GPU: ivcore11 + device memory                |
+------------------------------------------------------+
```

CoreX 提供 CUDA 兼容设备接口，因此代码使用 `cuda:0` 和
`torch.cuda.synchronize()`。接口兼容不代表底层硬件是 NVIDIA GPU：

- BI-V150 使用 ivcore11 架构及对应的编译后端；
- 本实验环境报告的 warp size 为 64；
- NVIDIA GPU 的二进制不能直接在 BI-V150 上运行；
- 其他平台的最优 tile、scheduler 参数和性能数据不能直接套用；
- API 兼容、源码可移植和性能可移植是三个不同问题。

## Triton 实验主线

PyTorch 提供高层 Tensor 与算子接口；Triton 使用接近 Python 的语言描述
GPU kernel，并允许显式控制 program、grid、tile、offset、mask、reduction
和编译期参数。

```text
Environment and device execution
                ↓
Vector Add: program / offsets / mask
                ↓
Softmax and LayerNorm: stable reductions
                ↓
Tiled GEMM: 2-D tiles / strides / K-loop / tl.dot
                ↓
Benchmark: warmup / synchronization / median
                ↓
Block sweep and Triton autotune
```

GEMM 使用统一的数据类型口径：

```text
A/B input  : fp16
accumulator: fp32
C output   : fp16
```

框架矩阵乘只用于构造正确性参考。Day 2 的最终性能汇总只比较固定
`32×32×32` Triton baseline 与 autotuned Triton kernel。

## 目录结构

```text
lab1/
├── day1/    # 环境检查、Vector Add、Softmax、LayerNorm、tiled GEMM
└── day2/    # baseline、block sweep、autotune、性能汇总
```

## 运行方式

在 BI-V150 官方环境中 clone 仓库，完成相应 TODO 后依次运行：

```bash
cd day1
bash setup.sh
python3 00_check_env.py
python3 01_torch_device_hello.py
python3 02_vector_add_triton.py
python3 03_softmax_compare.py
python3 04_layernorm_compare.py
python3 05_matmul_compare.py

cd ../day2
bash setup.sh
python3 00_baseline.py
python3 01_block_sweep.py
python3 02_autotune.py
python3 03_summary.py
```

详细任务说明见 `day1/README.md` 和 `day2/README.md`。

## 结果判断

### 正确性

- benchmark 前先执行 correctness gate；
- 检查 NaN/Inf、最大误差和 `assert_close`；
- 使用非整 tile shape 验证尾块 mask；
- Softmax、LayerNorm 和 GEMM 的关键归约使用 fp32；
- GEMM 的最大误差需要结合 dtype、输出规模、绝对容差和相对容差判断。

### 性能

- 首次 JIT/autotune 不计入 steady-state latency；
- 使用 warmup、显式同步、重复采样和中位数；
- baseline 与 autotuned kernel 使用相同 shape、dtype 和输出复用方式；
- 保留 PASS、FAIL 和资源错误，不能只记录最快结果；
- 性能数字需要同时注明设备、软件版本、shape 和 dtype。

最终汇总使用 `2048x2048 @ 2048x2048`。Autotune 使用少量经过 sweep
筛选的 block tile，并为候选采用相同的 scheduler 参数。BI-V150 实测相对
固定 `32x32x32` baseline 的加速约为 `2.9x`。绝对时间和最后几位会随
设备负载变化，不应把特定倍数作为通过条件。

## 环境要求

使用平台提供的 BI-V150 官方镜像，其中应包含匹配的 CoreX 驱动、运行时、
PyTorch 和 Triton。不要执行：

```bash
pip install torch
pip install triton
pip install --upgrade torch triton
```

公版 wheel 可能覆盖 CoreX 适配版本。首次运行可保存以下环境信息：

```bash
ixsmi -L
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

## 参考资料

- BI-V150 平台说明：https://moark.com/docs/compute/clusters_gpu/iluvatar/iluvatar_BI-V150_gpu
- Triton 官方教程：https://triton-lang.org/main/getting-started/tutorials/
- Triton Matrix Multiplication：https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html
- 原 Ascend Day 1：https://gitee.com/jieran-zhang/lab_day1
- 原 Ascend Day 2：https://gitee.com/jieran-zhang/lab_day2
