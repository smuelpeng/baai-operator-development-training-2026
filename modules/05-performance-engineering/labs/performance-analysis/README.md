# 模块五：性能评测与优化实践

Gitee（学生仓）：https://gitee.com/yihan-long/day2-lab5

Lab1 Day1–2 解决「如何写出并调好 Triton GEMM」；本模块解决：

> 如何科学地证明它快不快、慢在算力还是访存，并用平台 profiler 交叉验证。

---

## Quick Start

```bash
cd day2-lab5
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py

export LAB_DAY2_ROOT=$PWD/../lab1-day2   # 推荐：对接 Day2 GEMM

python3 01_benchmark.py --scenario offline
python3 01_benchmark.py --scenario server
# 三方对比（REPORT 需要）：编辑 configs/student_todos.py 加入 autotune，或：
# python3 01_benchmark.py --scenario offline --impl torch baseline autotune

pip install matplotlib   # 仅 Roofline 出图需要；JSON 诊断可不依赖
python3 02_roofline.py
python3 03_profiling.py
python3 04_summary.py
```

只改 `>>> YOUR CODE HERE >>>` … `<<< END OF YOUR CODE <<<`（见 [`configs/student_todos.py`](configs/student_todos.py)）。
详细叙事见 [`handout.md`](handout.md)；书面产出填 [`REPORT.md`](REPORT.md)。

---

## 实验内容

### 环境（`setup.sh` / `00_*`）

**要做什么：** 确认 CoreX torch / Triton / CUDA 可用，完成最小 GPU 冒烟。

**知识点：** FlagOS 软件栈约定；可信测量前的环境基线；与 Lab1 相同的 `cuda:0` 设备约定。

### 实验一：Benchmark（`01_benchmark.py`）

**要做什么：** 对 `torch` / `baseline`（及 `autotune`）跑 Offline 与 Server LoadGen，把延迟、吞吐、有效 TFLOP/s 写入 JSON。

**知识点：**
- 仿 MLPerf 的 Offline vs Server（到达过程）场景
- warmup、`synchronize`、median / p99
- 有效算力 \(2MNK/t\)；多实现横向对比

### 实验二：Roofline（`02_roofline.py`）

**要做什么：** 基于实验一结果计算算术强度，绘制 Roofline，判断 compute-bound / memory-bound 及是否远低于屋顶。

**知识点：**
- \(I=\mathrm{FLOPs}/\mathrm{Bytes}\)，脊点与 \(P=\min(\pi,\beta I)\)
- 简化访存模型的边界；「类属算力侧但仍远低于峰值」的含义

### 实验三：Profiling（`03_profiling.py`）

**要做什么：** 用 `ixsmi` / `ixsys` / `ixkn-cli` 观察同一 baseline GEMM（tile 与实验一对齐），并与 Roofline 交叉验证。（天数平台不支持 Proton。）

**知识点：**
- 端到端指标 vs 系统/Kernel 级证据
- 利用率、trace、occupancy 等观察如何支撑或修正 Roofline 判断

### 汇总与报告（`04_summary.py` + `REPORT.md`）

**要做什么：** 汇总 `results/`，完成开放式研讨问题（不必背诵定义，重在论证）。

---

## 与 Lab1 的关系

| Lab1 Day1–2 | 本模块 |
|-------------|--------|
| 实现与调优 tiled GEMM / tile sweep / autotune | 把该 GEMM（或内置等价实现）当作 **SUT** |
| 建立可信计时习惯 | 嵌入 LoadGen，并扩展到 Roofline + 平台 profiler |
| 常观察到 Triton 仍慢于 `torch.matmul` | 用测量与模型解释「差多少、差在哪一类」 |

推荐目录：

```text
~/labs/
  lab1-day2/
  day2-lab5/
```

---

## 产出物

- `results/benchmark_*.json`、`roofline.json`（及可选 `roofline.png`）
- `results/diagnosis.json`、`summary.json`
- 填写后的 [`REPORT.md`](REPORT.md)

闭环说明：[`docs/WORKFLOW_FLAGOS.md`](docs/WORKFLOW_FLAGOS.md)。
