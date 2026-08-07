# 实验环境说明

## 目标平台

课程实验以天数智芯 BI-V150 和 CoreX 软件栈为目标。平台通过 CUDA 兼容接口暴露 `cuda:0` 与 `torch.cuda`，底层架构、编译后端和性能特征仍由 BI-V150 决定。

需要明确区分三件事：API 可以兼容，源码可以移植，最佳性能配置仍可能完全不同。NVIDIA、Ascend 或其他平台上的 tile、warp 和二进制结论不能直接套用。

## 开始前的环境快照

```bash
ixsmi -L
python3 -c "import torch, triton; print(torch.__version__, triton.__version__)"
python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

把输出放进实验报告。平台课件和实验代码使用的关键口径包括：

- 设备：BI-V150 / ivcore11；
- CoreX CUDA 兼容设备接口；
- 课程环境所示 warp size 为 64；
- GEMM 常用 fp16 输入/输出、fp32 accumulator；
- 性能诊断工具包括 `ixsmi`、`ixsys`、`ixkn-cli`。

## 不要破坏平台镜像

不要直接执行：

```bash
pip install torch
pip install triton
pip install --upgrade torch triton
```

公开 wheel 可能覆盖 CoreX 适配版本。如果缺少 Python 包，优先新建隔离环境或只安装与平台框架无冲突的辅助包，并保存变更清单。

## 模块环境依赖

| 模块 | 最低要求 | 额外工具 |
|---|---|---|
| 1 | CoreX PyTorch、Triton、BI-V150 | 无 |
| 2 | 上述环境、FlagGems | `torch.compile` 可用性按平台版本确认 |
| 3 | 可输出 Triton 编译产物的环境 | MLIR/LLVM 工具以平台镜像为准 |
| 4 | 公式推演可离线完成 | 多卡实测需要分布式后端与至少 2 卡 |
| 5 | 模块 1 的 GEMM 结果 | `ixsmi`、`ixsys`、`ixkn-cli`；绘图可用 matplotlib |

## 运行边界

仓库中的源码是固定提交快照，已完成目录与静态内容核对。没有 BI-V150 时仍可阅读代码、推导公式和完成教学设计，但不能把普通 Mac/CPU 上的导入失败或性能数据当作课程实验结果。
