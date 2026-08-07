# 模块 1：AI 系统软件基础与 Triton

## 核心问题

一段 Python 风格的 Triton kernel 如何被拆成可在异构加速器上并行执行的 program 和 tile？

## 学习目标

- 解释 CPU host、GPU device、stream、event 与异步执行；
- 使用 grid、program id、offset 和 mask 表达并行工作；
- 实现或读懂 Vector Add、Softmax、LayerNorm 与 tiled GEMM；
- 建立 correctness gate 和可信 benchmark；
- 用控制变量 sweep 与 autotune 改进 GEMM。

## 概念地图

```text
Host launch → Grid → Program instance → Tile
                                      ├─ offsets / mask
                                      ├─ load / compute / store
                                      └─ reduction / tl.dot

正确性 → warmup 与同步 → baseline → block sweep → autotune → 复核
```

## 课件

- [模块一主课件](../../materials/courseware/module-01/模块一：AI%20系统软件基础与异构计算.pdf)
- [实验平台说明](../../materials/platform/线上实验室平台介绍0804.pdf)
- 相关背景材料见 [课件索引](../../materials/README.md)。

## 实验顺序

1. [Triton 基础综合实验](labs/triton-basics/)：环境、Vector Add、Softmax、LayerNorm、GEMM。
2. [GEMM 调优实验](labs/gemm-tuning/)：baseline、block sweep、scheduler sweep、autotune。

`triton-basics` 内也包含一份 Day 2；独立的 `gemm-tuning` 是群内发布的单独课程仓快照，两份均保留并注明来源。授课时建议使用 `triton-basics/day1` 接 `gemm-tuning`。

```bash
cd modules/01-ai-system-and-triton/labs/triton-basics/day1
bash setup.sh
python3 00_check_env.py
# 按 README 顺序完成 01–05

cd ../../gemm-tuning
bash setup.sh
# 按 README 顺序完成 00–03
```

## 建议时长与验收

- 讲授 2.5 小时；实验 3.5 小时。
- 验收：非整 tile shape 正确；fp32 归约口径明确；计时有 warmup、同步和 median；autotune 候选能由 sweep 结果解释。

## 离开本模块前

学生应能回答：大 tile 为什么可能更快，也可能因并行度或片上资源压力更慢？相同 Triton 源码为什么必须在 BI-V150 上重新调优？
