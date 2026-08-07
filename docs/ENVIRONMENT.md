# 实验环境说明

FlagOS 门户、实例配额、环境取证、上机命令和数据保全步骤见 [FlagOS 在线实验室指南](FLAGOS_ONLINE_LAB.md)。本页保留课程级环境原则。

## 目标平台

课程实验以天数智芯 BI-V150 和 CoreX 软件栈为目标。平台通过 CUDA 兼容接口暴露 `cuda:0` 与 `torch.cuda`，底层架构、编译后端和性能特征仍由 BI-V150 决定。

需要明确区分三件事：API 可以兼容，源码可以移植，最佳性能配置仍可能完全不同。NVIDIA、Ascend 或其他平台上的 tile、warp 和二进制结论不能直接套用。

## 开始前的环境快照

```bash
ixsmi -L
python3 -c "import torch, triton; print(torch.__version__, torch.__file__); print(triton.__version__, triton.__file__)"
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

公开 wheel 可能覆盖 CoreX 适配版本。CoreX 4.4 镜像中的 `torch.__version__` 可能显示为普通 `2.7.1`；不能只凭是否含 `+corex` 后缀判断，应同时核对模块路径、`COREX_HOME`、设备属性和最小 kernel。缺少 Python 包时，优先新建隔离环境或只安装与平台框架无冲突的辅助包，并保存变更清单。

## 模块环境依赖

| 模块 | 最低要求 | 额外工具 |
|---|---|---|
| 1 | CoreX PyTorch、Triton、BI-V150 | 无 |
| 2 | 上述环境、FlagGems | `torch.compile` 可用性按平台版本确认 |
| 3 | 可输出 Triton 编译产物的环境 | MLIR/LLVM 工具以平台镜像为准 |
| 4 | 公式推演可离线完成 | 多卡实测需要分布式后端与至少 2 卡 |
| 5 | 模块 1 的 GEMM 结果 | `ixsmi`、`ixsys`、`ixkn-cli`；绘图可用 matplotlib |

## 运行边界

仓库中的源码是固定提交快照，已完成目录与静态内容核对。2026-08-08 的 BI-V150 动态验证范围及原始结果见 [实验报告](../records/experiments/2026-08-08-flagos-bi-v150-validation.md)。没有 BI-V150 时仍可阅读代码、推导公式和完成教学设计，但不能把普通 Mac/CPU 上的导入失败或性能数据当作课程实验结果。
