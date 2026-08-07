# FlagOS 在线实验室：BI-V150 使用与验证指南

本文面向使用课程仓库的学生和助教，说明怎样进入 FlagOS 在线实验室、识别实际资源、保留环境证据，并按低风险顺序运行课程代码。

> 在线实验室是限时容器。门户明确提示：实例释放后，容器内数据会被销毁且无法恢复。代码、日志、JSON 和报告必须在释放前复制到持久存储。

## 1. 本期实例卡

门户在 2026-08-08 显示两台课程实例，时间均为北京时间（UTC+8）。

| 实例 | 加速卡 | CPU 配额 | 内存配额 | 系统盘 | 创建时间 | 计划释放时间 | 本次验证 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `activity_29xtu1xs_1996_1785741080` | BI-V150 × 1 | 6 核 | 32 GiB | 200 G | 2026-08-03 15:11:20 | 2026-08-10 00:00:00 | 已在 `n139` 运行 |
| `activity_29zg93y5_1938_1785728425` | BI-V150 × 2 | 12 核 | 64 GiB | 200 G | 2026-08-03 11:40:25 | 2026-08-10 00:00:00 | 未做动态验证 |

门户资源信息的本地证据为 [FlagOS 在线实验室资源卡截图](../records/evidence/flagos-online-lab-resource-cards-2026-08-07.png)，SHA-256：`8f3e52bb077c7c3e11e6027f2dbad70cd4886e8a8198819e0e54bc62b8abdf47`。

本次使用的 IDE 入口为：

```text
https://flagos.io/flagos-lab/ts/node/n139/port/14014/?folder=/root
```

链接打开的是基于浏览器的 code-server，初始目录为 `/root`。课程实例到期后，这个地址及容器文件都不应被视为长期存储。

## 2. n139 的实测环境

以下数据来自一卡实例中的动态检查，不是从宣传页推定的配置。

| 项目 | 实测值 | 判读方式 |
| --- | --- | --- |
| 容器 | Ubuntu 24.04.2，Docker | `/etc/os-release`、`/.dockerenv` |
| hostname | `6ee5cdea35d2` | `hostname` |
| CPU | cgroup 配额 6 核 | cgroup v1：`cpu.cfs_quota_us=600000`、`cpu.cfs_period_us=100000` |
| 内存 | cgroup 上限 32 GiB | cgroup v1：`memory.limit_in_bytes=34359738368` |
| 根文件系统 | overlay，200 G | `df -h /` |
| 加速卡 | BI-V150 × 1，显存 32 GiB | PyTorch 设备属性与平台工具 |
| CoreX / Driver / IX-ML | 4.4.0 | 平台工具和运行时路径 |
| CUDA 兼容版本 | 10.2 | CoreX 兼容接口报告值 |
| Python | 3.10.18 | `python3 -VV` |
| PyTorch | 2.7.1 | `torch.__version__` |
| Triton | 3.1.0 | `triton.__version__` |
| FlagGems | 4.2.1.rc.0 | `flag_gems.__version__` |
| FlagGems 源码 | `/root/FlagGems`，commit `6f2585dc9` | `git rev-parse --short HEAD` |
| 编译目标 | arch 71，warp size 64 | 设备属性与 Triton/CoreX 输出 |

`torch==2.7.1` 没有 `+corex` 后缀，不能据此判断成公开版 PyTorch。本次 `torch.__file__` 位于 `/usr/local/corex-4.4.0/` 下，且设备、驱动与 Triton kernel 均能运行，这是需要一并保存的证据。

门户没有公开本实例的镜像 tag 或 digest。容器内部文件和软件版本可以用于环境指纹，不能可靠反推出镜像 digest。需要镜像级复现时，应由平台方提供镜像元数据。

## 3. 配额与宿主信息不要混用

容器内的 `lscpu` 和 `free -h` 可能显示宿主机的 112 个 CPU 与约 503 GiB 内存。调度器实际允许本容器使用的资源由 cgroup 决定：n139 为 6 核和 32 GiB。

进入新实例后，使用下面的命令重新确认，不要照抄本页数字：

```bash
date -Is
hostname
cat /etc/os-release
test -f /.dockerenv && echo "Docker container: yes"

if test -r /sys/fs/cgroup/cpu.max; then
  echo "cgroup v2"
  printf 'CPU quota and period: '
  cat /sys/fs/cgroup/cpu.max
  printf 'Memory limit (bytes): '
  cat /sys/fs/cgroup/memory.max
else
  echo "cgroup v1"
  printf 'CPU quota: '
  cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us
  printf 'CPU period: '
  cat /sys/fs/cgroup/cpu/cpu.cfs_period_us
  printf 'Memory limit (bytes): '
  cat /sys/fs/cgroup/memory/memory.limit_in_bytes
fi
df -h /
```

n139 实测使用 cgroup v1。CPU 配额除以周期，`600000 / 100000 = 6`；内存上限按字节换算。cgroup v2 会把 CPU 的两个值放在 `cpu.max` 中。两种版本都不要采用 `free -h` 的宿主总量作为实验配额。

## 4. 设备选择与软件路径

CoreX 提供 CUDA 兼容 API，因此课程代码使用 `cuda:0` 和 `torch.cuda`。这套接口运行在 BI-V150 上，不能把 NVIDIA 的二进制、32-thread warp 假设或调优数字直接套到本环境。

实例中出现的 `/dev/iluvatar11` 是 Linux 设备节点名，其中的 `11` 不是 CUDA 设备序号。单卡容器采用下面的设置：

```bash
unset CUDA_VISIBLE_DEVICES
python3 - <<'PY'
import torch

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 1
torch.cuda.set_device(0)
props = torch.cuda.get_device_properties(0)
print("device:", torch.cuda.get_device_name(0))
print("warp_size:", props.warp_size)
print("memory_GiB:", round(props.total_memory / 2**30, 1))
PY
```

不要执行下面的命令：

```bash
pip install torch
pip install triton
pip install --upgrade torch triton
```

公版 wheel 可能覆盖平台适配包。版本号看起来普通时，先检查模块路径和最小 kernel，再决定环境是否异常。

推荐保存这份不含凭据的环境摘要：

```bash
mkdir -p /root/baai-validation

python3 -VV | tee /root/baai-validation/python-version.txt

python3 - <<'PY' | tee /root/baai-validation/python-stack.txt
import torch
import triton
import flag_gems

print("torch", torch.__version__, torch.__file__)
print("triton", triton.__version__, triton.__file__)
print("flag_gems", flag_gems.__version__, flag_gems.__file__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print("device", torch.cuda.get_device_name(0))
    print("warp_size", p.warp_size)
    print("memory_GiB", round(p.total_memory / 2**30, 1))
PY

git -C /root/FlagGems rev-parse HEAD \
  | tee /root/baai-validation/flaggems-commit.txt
```

不要把完整 `env`、访问令牌、一次性登录码或设备标识复制进日志。

## 5. 私有课程仓库怎样进入实例

课程 GitHub 仓库是 Private。n139 本次没有可用的 GitHub 凭据，设备登录流程也没有在超时前完成，因此没有在远端直接 clone 当前 GitHub 整仓。Chrome 向 IDE 上传压缩包时又被安全策略拒绝。两项限制都不能当作课程代码失败。

先做无交互访问检查：

```bash
GIT_TERMINAL_PROMPT=0 timeout 20 \
  git ls-remote \
  https://github.com/smuelpeng/baai-operator-development-training-2026.git \
  HEAD
```

若命令失败，使用课程管理员批准的源码传递方式，或运行实例中已经提供的 Gitee 固定提交。不要把个人长期令牌写入容器、shell 历史或仓库。源码版本以 [`records/code_repository_inventory.md`](../records/code_repository_inventory.md) 为准。

本次动态验证使用了仓库中已归档并校验的对应 Gitee 快照，涉及：

| 实验 | 固定提交 |
| --- | --- |
| Triton Lab 1 | `main@1969ed2741c5dbadf8961c85d827e7d05e3f0c30` |
| Fused RMSNorm 参考代码 | `master@3ba8e2d6d94556ba45be9ce15c8751d44986affe` |
| 模块 5 性能实验 | `main@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f` |

这说明相应教学快照可以在 BI-V150 上运行，不表示远端已验证私有 GitHub 仓库的最新 commit。

## 6. 推荐的验证顺序

先设置实际的课程仓库目录，并确认路径没有填错：

```bash
export COURSE_ROOT=/root/baai-operator-development-training-2026
test -f "$COURSE_ROOT/AGENTS.md"
```

如果实例使用其他目录，只修改 `COURSE_ROOT`。

### 6.1 模块 1：环境与已完成样例

```bash
cd "$COURSE_ROOT/modules/01-ai-system-and-triton/labs/triton-basics/day1"
bash setup.sh
python3 00_check_env.py
python3 01_torch_device_hello.py
python3 03_softmax_compare.py
python3 04_layernorm_compare.py
python3 05_matmul_compare.py

cd "$COURSE_ROOT/modules/01-ai-system-and-triton/labs/gemm-tuning"
bash setup.sh
python3 00_baseline.py
```

`02_vector_add_triton.py` 保留了学生 TODO。没有填写时跳过；完成 `add_kernel` 后再执行：

```bash
python3 02_vector_add_triton.py
```

本次实测中，环境检查的最小 Triton kernel、Torch 设备样例、Softmax、LayerNorm 和 Day 1 MatMul 全部正常退出。固定 `32×32×32` baseline 的预分配结果为：1024³ 得到 6.696 TFLOP/s，2048³ 得到 10.728 TFLOP/s。它们是 n139 当次结果，不是跨实例性能门槛。

### 6.2 模块 2：Fused Add + RMSNorm

```bash
cd "$COURSE_ROOT/modules/02-high-performance-operators/labs/fused-rmsnorm-reference"
python3 fused_rms_norm.py
python3 grad_kernel.py --m 128 --n 256 --dtype float32
python3 compile_demo.py --m 128 --n 256
python3 benchmark.py --m 4096 --n 4096 --dtype float32
```

原 Gitee `completed_lab2@3ba8e2d` 的 `fused_rms_norm.py` 名为参考实现，但 `rrms` 和 `out` 两行仍是空缺。课程仓库已补回 README 中给出的两行参考答案。运行前可检查：

```bash
rg -n "rrms = 1.0 / tl.sqrt|out = updated \* rrms \* weight" \
  fused_rms_norm.py
```

本次实测达到以下结果：

- float32 前向 `17×257` 最大误差 `4.768e-7`；
- float32 前向 `128×256` 最大误差 `9.537e-7`；
- 反向检查通过，weight 最大误差 `7.629e-6`；
- `torch.compile` 捕获 1 个 graph，graph break 为 0；
- `4096×4096` float32：融合实现 0.5423 ms、495.06 GB/s，FlagGems 0.5280 ms、508.42 GB/s，吞吐比 97.37%，通过 90% 门槛。

`512×1024` 的同一性能比只有 59.55%。小 shape 受启动、tile 和工作量影响明显，这项结果应记录为 FAIL，不能用来否定 `4096×4096` 的课程验收，也不能省略。

### 6.3 模块 5：环境、冒烟与 LoadGen

```bash
cd "$COURSE_ROOT/modules/05-performance-engineering/labs/performance-analysis"
unset CUDA_VISIBLE_DEVICES
source ./env_corex.sh
bash setup.sh

export LAB_DAY2_ROOT="$COURSE_ROOT/modules/01-ai-system-and-triton/labs/gemm-tuning"
python3 00_check_env.py
python3 00_smoke_cuda.py
python3 01_benchmark.py \
  --scenario offline \
  --impl torch baseline \
  --num-queries 4 \
  --M 256 --N 256 --K 256 \
  --out results/benchmark_offline_smoke_256.json
```

本次 64×64 fp16 MatMul 冒烟正常退出。256³ Offline 小样本也正常写出 JSON：Torch median 0.057 ms、0.589 TFLOP/s；Day 2 baseline median 0.334 ms、0.100 TFLOP/s。这组数字只检查 LoadGen、SUT 和结果写入是否连通，不用于课程性能排名。

镜像中缺少 `matplotlib`，所以没有生成 Roofline PNG；JSON 诊断不依赖它。本次没有安装新包，避免改变平台镜像。正式画图时可在隔离环境中只安装绘图依赖，并记录变更。

## 7. 结果怎样保存

每次运行至少保留：

- 实例名、时间、cgroup 配额；
- 设备名、warp size、显存；
- `torch`、`triton`、`flag_gems` 版本与模块路径；
- 课程源码 commit；
- shape、dtype、命令、退出码；
- correctness 最大误差；
- 性能测试的 warmup、样本数、同步方式、median/P99；
- PASS 和 FAIL 的全部配置。

本次资料中的证据文件：

| 证据 | 路径 | SHA-256 | 证明范围 |
| --- | --- | --- | --- |
| 门户资源卡截图 | [`records/evidence/flagos-online-lab-resource-cards-2026-08-07.png`](../records/evidence/flagos-online-lab-resource-cards-2026-08-07.png) | `8f3e52bb077c7c3e11e6027f2dbad70cd4886e8a8198819e0e54bc62b8abdf47` | 两张实例卡的资源、创建时间、释放时间；不证明镜像 tag/digest |

推荐目录：

```text
/root/baai-validation/
├── environment/
├── module-01/
├── module-02/
└── module-05/
```

在实例释放前，将目录复制到本地受控存储，并把适合纳入仓库的摘要写入 `records/experiments/`。内部课件、邀请函和群文件仍须遵守 Private 仓库的使用范围。

## 8. 常见问题

| 现象 | 检查 | 处理 |
| --- | --- | --- |
| `torch.cuda.is_available()` 为 False | `CUDA_VISIBLE_DEVICES` 是否被设成设备节点尾号 | `unset CUDA_VISIBLE_DEVICES`，重新 `source env_corex.sh` |
| PyTorch 显示 `2.7.1`，没有 `+corex` | 查看 `torch.__file__`、`COREX_HOME`，运行最小 Triton kernel | 不要用 pip 覆盖；结合路径和动态结果判读 |
| `lscpu` 显示 112 核 | 按本页命令检查 cgroup v1/v2 配额文件 | 报告 cgroup 配额 6 核 |
| GitHub clone 要求登录 | 使用无交互 `ls-remote` 记录失败 | 采用批准的传递方式或固定 Gitee 快照，不保存个人令牌 |
| Vector Add 报错或没有实现 | 查看学生 TODO 标记 | 完成 TODO 后再运行 |
| `completed_lab2` 前向变量未定义 | 检查 `rrms`、`out` 两行 | 使用课程仓库中的修订版并重新跑前向、反向、compile |
| 小 shape 性能比低 | 检查 shape、计时边界和正式门槛 | 保留失败；按 `4096×4096` 正式配置复测 |
| 缺少 `matplotlib` | 查看 JSON 是否已经生成 | 不改 torch/triton；绘图依赖单独处理 |

本次动态证据的完整摘要见 [`records/experiments/2026-08-08-flagos-bi-v150-validation.md`](../records/experiments/2026-08-08-flagos-bi-v150-validation.md)。
