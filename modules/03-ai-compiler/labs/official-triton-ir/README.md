# 官方模块三 Triton IR 实验（候选快照）

上游：`https://gitee.com/nsddrezsd/lab3`

固定提交：`5334a42b986c103ba617f074f1a1b4cba00c610d`

## 保存范围

`upstream/` 原样保存核心 Python、两个上游 README 和依赖清单。没有保存 `.git`、`__pycache__`、`.pyc`、`.so`、`.triton-cache`、设备二进制、重复调参结果和上游生成的 IR 结果。

排除生成结果是有意为之：当前提交的 `README_new.md` 把 `artifacts/` 描述为 BI-V150 结果，但同一提交的 `run.json` 实际记录 NVIDIA H200、warp size 32 和 Triton 3.6.0。两组证据冲突，不能作为 BI-V150 动态事实入库。

`upstream/requirements.txt` 要求 `torch>=2.11` 和 `triton>=3.6`，`run_lab.py` 的依赖错误提示也要求安装 NVIDIA CUDA 版本；这些要求与研修班实测的 CoreX 4.4、PyTorch 2.7.1、Triton 3.1.0 不兼容。**不要执行 `pip install -r upstream/requirements.txt`。** 当前快照首先用于阅读和适配任务，不是可直接运行的 BI-V150 实验包。

## 使用顺序

1. 先读仓库根目录 `AGENTS.md`、`docs/AI_CONTEXT.md` 和 `docs/FLAGOS_ONLINE_LAB.md`。
2. 先做只读环境指纹，确认 CoreX、PyTorch、Triton 和设备属性；不要执行上游依赖安装或 NVIDIA 安装命令。
3. 阅读 `upstream/kernels.py`、`ir_utils.py` 和 `run_lab.py`，检查后端判断与输出目录。
4. 在独立任务工作区运行，生成新的 TTIR、TTGIR、LLIR 和 `run.json`，不要把上游文字当作本次结果。
5. 将设备、版本、命令、误差和原始产物写入任务 `results/` 与 `REPORT.md`。

## 当前状态

源码已完成静态归档与 Python 语法检查；适配和硬件运行均待完成。它用于填补模块三“IR 导出实验候选来源”的缺口，不构成模块三动态验收通过。
