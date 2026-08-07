# {{TASK_NAME}}：任务合同

状态：草稿 / 已确认 / 实现中 / 硬件验证中 / 完成

## 目标

用一句话说明要实现或优化的算子，以及它在模型中的位置。

## 数学定义

当前脚手架用 `out = x + y` 作为可替换示例。请写出完整公式、归约维度、输出顺序和特殊值行为。

## 输入与输出

| 名称 | shape | dtype | layout/device | 说明 |
|---|---|---|---|---|
| `x` |  |  |  |  |
| `y` |  |  |  |  |
| `out` |  |  |  |  |

## 行为合同

- inplace / out-of-place：
- contiguous / stride 要求：
- 广播规则：
- alias 与副作用：
- forward / backward：
- Dispatcher / Meta / `torch.compile`：

## 正确性矩阵

| 类别 | 用例 | 容差 | 状态 |
|---|---|---|---|
| 最小 shape |  |  |  |
| 非整 tile |  |  |  |
| 典型 shape |  |  |  |
| 压力 shape |  |  |  |
| dtype |  |  |  |
| 梯度 |  |  |  |

## 性能目标

- 目标平台：
- baseline：
- shape / dtype：
- 计时边界：kernel-only / wrapper / end-to-end
- 指标：median / P99 / throughput / TFLOP/s
- 通过门槛：

## 允许修改

- 当前工作区内的源码、测试、报告和 `results/`。
- 如需修改框架或模块快照，先在本节列出具体文件与原因。

## 验收命令

```bash
python3 test_operator.py
python3 benchmark.py
```

## 已知限制

列出目标硬件、软件版本、资源、数据和精度方面的限制。
