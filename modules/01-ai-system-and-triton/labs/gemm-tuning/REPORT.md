# Day 2 实验报告

## 1. 环境

- 设备名：
- PyTorch：
- Triton：
- CoreX/镜像信息：

## 2. Baseline

| Shape `(M,N,K)` | 口径 | Tile | Median ms | TFLOP/s | Max diff |
|---|---|---|---:|---:|---:|
| | end-to-end / preallocated | 32×32×32 | | | |

## 3. Block Sweep

记录所有候选，包括编译失败或结果错误的配置。

| Shape | BM | BN | BK | 状态 | Median ms | TFLOP/s | 解释 |
|---|---:|---:|---:|---|---:|---:|---|
| | | | | | | | |

### CoreX Resource Probes

| BM | BN | BK | Acc proxy KiB | Operand proxy KiB | 状态 | 完整错误摘要 |
|---:|---:|---:|---:|---:|---|---|
| | | | | | | |

## 4. Autotune

| Shape | Baseline ms | Best block | Warps | Stages | Autotuned ms | 相对 baseline |
|---|---:|---|---:|---:|---:|---:|
| | | | | | | |

### BK Sweep

| BM | BN | BK | K-loop 次数 | Operand proxy KiB | Median ms | TFLOP/s |
|---:|---:|---:|---:|---:|---:|---:|
| | | | | | | |

### Scheduler Sweep

| Block | Num warps | Num stages | 状态 | Median ms | TFLOP/s |
|---|---:|---:|---|---:|---:|
| | | | | | |

### Exact-config Check

- exact-config fixed / autotuned decorated ratio：
- 是否稳定接近 1：
- 若不是，可能原因：

## 5. 结论

1. autotune 相对固定 `32×32×32` baseline 提升了多少？重复运行后是否稳定？
2. 放大 BM 与放大 BN 的效果是否对称？
3. BK 增大后，K-loop 次数与资源占用如何变化？
4. 128 字节对齐假设是否得到数据支持？
5. 为什么不同 shape 选出了不同 tile？
6. 你的结果中哪些是可迁移的方法，哪些只是本次平台实测事实？
7. CoreX 是否对 `num_warps` 或 `num_stages` 敏感？证据是什么？
