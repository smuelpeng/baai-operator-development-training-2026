# Day 2：BI-V150 GEMM Block 调优

## 实验任务

Day 2 只保留一个未完成任务：根据 `01_block_sweep.py` 的 PASS 结果，补全
`02_autotune.py` 中的 `BLOCK_TILES` 和 `AUTOTUNE_CONFIGS`。

CoreX resource probe、block sweep、BK sweep、scheduler sweep、正确性检查和
最终性能汇总均已提供。

## 运行顺序

```bash
bash setup.sh
python3 00_baseline.py
python3 01_block_sweep.py
python3 02_autotune.py
python3 03_summary.py
```

`02_autotune.py` 和 `03_summary.py` 需要在补全 TODO 后运行。

## 实验流程

### Step 00：Baseline

`00_baseline.py` 使用固定 `32x32x32` tile，分别测试 end-to-end 和复用输出
的 preallocated 路径。后续最终比较统一复用输出，避免输出分配影响结果。

### Step 01：Block Sweep

`01_block_sweep.py` 分为两部分：

- Part A 尝试较大的 tile，观察编译、运行和资源限制；
- Part B 对不同 `BLOCK_M/N/K` 组合进行正确性检查和性能测量。

输出中的 `PASS tiles to consider for Step 02` 表示这些配置能够正常编译、
运行并通过正确性检查，可以作为 Autotune 候选。`PASS` 不等于最快，不同
shape 下的性能排序也可能不同。

### Step 02：Autotune 与参数分析

根据 sweep 结果选择少量有代表性的 block，包括：

- 较快的对称 tile；
- M/N 方向不同的非对称 tile；
- 不同 `BLOCK_K`；
- 在多个 shape 上能够通过正确性检查的配置。

为降低填写量，只需保留少量 block，并为所有候选使用同一组 scheduler
参数。具体候选和参数可以在完成实验后与文末参考答案对照。

Autotune 第一次遇到某个 `M/N/K` 时会测试候选配置并缓存最快结果。文件后半
部分还包含 BK sweep 和 scheduler sweep，用于观察参数变化对 K-loop、资源
占用和性能的影响。

### Step 03：Summary

`03_summary.py` 使用 `2048x2048 @ 2048x2048`，只比较：

- 固定 `32x32x32` Triton baseline；
- Autotune 选出的 Triton kernel。

BI-V150 实测加速约为 `2.9x`。绝对时间和最后几位会随设备负载变化，
不应把特定倍数作为通过条件。

## 结果判断

- 所有性能候选必须先通过正确性检查；
- `max diff` 需要结合 fp16 输出、绝对容差和相对容差判断；
- warmup 和首次 autotune 不计入 steady-state latency；
- 资源探测中的预期 FAIL 是实验结果，不代表整个脚本失败；
- footprint 只是不包含后端实现细节的近似量；
- 不能把 Ascend 或 NVIDIA 平台的资源容量和最佳配置直接套用到 BI-V150。

## 参考答案

建议先根据 `PASS tiles` 独立填写，再与以下配置对照。

```python
BLOCK_TILES = [
    (64, 128, 16),
    (64, 128, 32),
    (128, 64, 32),
    (128, 128, 32),
    (128, 128, 64),
    (64, 256, 32),
]

AUTOTUNE_CONFIGS = [
    triton.Config(
        {"BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk},
        num_warps=8,
        num_stages=1,
    )
    for bm, bn, bk in BLOCK_TILES
]
```
