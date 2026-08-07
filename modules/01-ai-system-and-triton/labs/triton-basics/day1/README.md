# Day 1：BI-V150 上的 Triton 编程基础

## 实验任务

Day 1 只保留一个未完成任务：补全 `02_vector_add_triton.py` 中
`add_kernel` 的函数体。`add()` launcher、测试用例以及其他基础算子均已完成。

需要实现以下步骤：

1. 使用 program id 确定当前 program 处理的数据块；
2. 生成该 block 对应的元素 offsets；
3. 创建 tail mask，保护最后一个不完整 block；
4. 从设备内存加载 `x` 和 `y`；
5. 完成加法并写回输出。

## 运行顺序

```bash
bash setup.sh
python3 00_check_env.py
python3 01_torch_device_hello.py
python3 02_vector_add_triton.py
python3 03_softmax_compare.py
python3 04_layernorm_compare.py
python3 05_matmul_compare.py
```

## 文件说明

- `setup.sh`：检查 PyTorch、Triton 和 BI-V150 是否可用；
- `00_check_env.py`：打印环境指纹，并运行 PyTorch/Triton smoke test；
- `01_torch_device_hello.py`：展示设备选择、warmup 与同步计时；
- `02_vector_add_triton.py`：练习 program、offset、mask、load 和 store；
- `03_softmax_compare.py`：实现数值稳定的逐行 Softmax；
- `04_layernorm_compare.py`：实现带 fp32 reduction 的 LayerNorm；
- `05_matmul_compare.py`：实现带 K-loop 的 tiled GEMM。

## 结果检查

Vector Add 会测试以下长度：

```text
1, 127, 128, 129, 1000, 4097
```

其中 127、129 和 4097 用于验证最后一个 block 的 mask。完成正确时，每组
测试都会输出 `PASS`。

其他算子会同时输出 PyTorch/Triton 时间和最大误差。Day 1 的重点是理解
编程模型并通过正确性检查；教学用 Triton kernel 在较小 shape 上慢于框架
优化算子不属于异常。

## 注意事项

- `cuda:0` 是 CoreX 提供的兼容设备接口，不需要导入 `torch_npu`；
- 当前 BI-V150 环境报告 warp size 为 64；
- 不要在平台镜像中升级或重新安装 `torch`、`triton`；
- 尾块必须使用 mask；
- Softmax 的 padding identity 应为 `-inf`，避免影响行最大值；
- LayerNorm 的无效 centered lane 应置零，避免污染方差；
- Softmax、LayerNorm 使用 fp32 reduction；
- GEMM 使用 fp16 输入/输出和 fp32 accumulator。

## 参考答案

建议完成 TODO 并运行测试后再查看。

```python
pid = tl.program_id(0)
offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
mask = offsets < n_elements
x = tl.load(x_ptr + offsets, mask=mask)
y = tl.load(y_ptr + offsets, mask=mask)
tl.store(out_ptr + offsets, x + y, mask=mask)
```
