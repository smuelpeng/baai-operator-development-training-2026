# FlagOS BI-V150 在线实验室课程仓库验证报告

生成时间：2026-08-08（Asia/Shanghai）

## 1. 结果摘要（Result Summary）

- **状态：** 部分通过。三个固定课程快照的指定正确性链路与正式验收项通过；模块 2 的小尺寸性能冒烟未达标。私有 GitHub 整仓没有在远端成功 clone，不能声明最新整仓已经完成动态验证。
- **目标：** FlagOS `n139` 一卡容器，BI-V150 × 1，CoreX 4.4.0。
- **最小运行：** 模块 1 环境检查包含一次 512×512 PyTorch MatMul 和最小 Triton `add_one` kernel；两项均完成，Triton 输出 `PASS`。
- **主要结论：** 模块 1 的已完成样例与 GEMM baseline、模块 2 的前向/反向/`torch.compile`/正式性能门槛、模块 5 的环境/MatMul/LoadGen 冒烟均可在该实例运行。
- **主要限制：** 动态测试使用本仓库已归档并校验的 Gitee 对应提交。GitHub 私有仓库无凭据，设备登录超时；Chrome 上传又被安全策略拒绝。镜像 tag 和 digest 未由门户公开，容器内部也无法可靠反推。
- **整合仓校验范围：** 当前课程整合仓在本地通过 `python3 scripts/validate_repository.py --strict`；该检查覆盖结构、链接、语法、元数据与选定校验和，不等同于在 n139 动态运行整个 GitHub HEAD。
- **数据保全：** 原始远端日志位于 `/root/baai-validation`。该目录随实例释放而销毁；本仓只保存本报告中的摘要，没有把远端日志写成永久已归档状态。

## 2. 环境（Environment）

### 2.1 门户资源卡

| 实例 | 加速卡 | CPU | 内存 | 系统盘 | 创建时间（UTC+8） | 释放时间（UTC+8） | 动态验证 |
| --- | ---: | ---: | ---: | ---: | --- | --- | --- |
| `activity_29xtu1xs_1996_1785741080` | BI-V150 × 1 | 6 核 | 32 GiB | 200 G | 2026-08-03 15:11:20 | 2026-08-10 00:00:00 | n139 已验证 |
| `activity_29zg93y5_1938_1785728425` | BI-V150 × 2 | 12 核 | 64 GiB | 200 G | 2026-08-03 11:40:25 | 2026-08-10 00:00:00 | 未验证 |

门户明确提示实例释放后数据会被销毁且无法恢复。

资源卡证据：[FlagOS 在线实验室资源卡截图](../evidence/flagos-online-lab-resource-cards-2026-08-07.png)，SHA-256：`8f3e52bb077c7c3e11e6027f2dbad70cd4886e8a8198819e0e54bc62b8abdf47`。截图证明门户当时显示的资源与时间，不证明镜像 tag/digest。

### 2.2 n139 动态指纹

| 项目 | 值 |
| --- | --- |
| Host / pod | hostname `6ee5cdea35d2`；Ubuntu 24.04.2；Docker 容器 |
| GPU / CPU | BI-V150 × 1，显存 32 GiB；cgroup v1 CPU 配额 `600000/100000=6` 核 |
| 内存 / 磁盘 | cgroup v1 `memory.limit_in_bytes=34359738368`（32 GiB）；根文件系统 200 G overlay |
| 宿主可见值 | `lscpu` / `free` 可见 112 CPU、约 503 GiB；这两个值不代表容器配额 |
| Driver / CUDA | CoreX、Driver、IX-ML 4.4.0；CUDA 兼容接口报告 10.2 |
| 设备接口 | 代码使用 `cuda:0`；`CUDA_VISIBLE_DEVICES` 留空；`/dev/iluvatar11` 的尾号不作 CUDA index |
| 编译目标 | arch 71；warp size 64 |
| Python | 3.10.18 |
| PyTorch | 2.7.1；模块文件位于 `/usr/local/corex-4.4.0/` 软件树 |
| Triton | 3.1.0 |
| FlagGems | `flag_gems==4.2.1.rc.0`；源码 `/root/FlagGems`，commit `6f2585dc9` |
| Working directory | 运行日志 `/root/baai-validation`；源码目录见第 3 节 |
| Repo commit | Lab 1 `1969ed2…`；completed_lab2 `3ba8e2d…`；day2-lab5 `5fb8557…` |
| Weights / data path | 无；本次测试不使用模型权重或外部数据集 |
| Image tag / digest | 门户未公开，未验证 |

PyTorch 版本没有 `+corex` 后缀。本次同时保存了 `/usr/local/corex-4.4.0/` 下的模块路径、设备可用性与 Triton 动态结果，因此没有只靠版本字符串判断软件来源。

## 3. 设置与精确命令（Setup and Commands）

下面列出能够复现本次结果的命令。环境采集在远端曾合并成一条长命令，这里拆成可读区块；实验脚本、参数和日志名与实际运行保持一致。第 3.1 至 3.5 节在 n139 容器内执行，第 3.6 节在本地课程整合仓执行。

### 3.1 资源与软件指纹

```bash
mkdir -p /root/baai-validation

{
  date -Is
  hostname
  cat /etc/os-release
  test -f /.dockerenv && echo "Docker container: yes"
  echo "cpu quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)"
  echo "cpu period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)"
  echo "memory limit=$(cat /sys/fs/cgroup/memory/memory.limit_in_bytes)"
  df -h /
} > /root/baai-validation/container-environment.log 2>&1

unset CUDA_VISIBLE_DEVICES
python3 -VV > /root/baai-validation/python-version.log 2>&1
python3 - <<'PY' > /root/baai-validation/python-stack.log 2>&1
import torch
import triton
import flag_gems

print("torch", torch.__version__, torch.__file__)
print("triton", triton.__version__, triton.__file__)
print("flag_gems", flag_gems.__version__, flag_gems.__file__)
print("cuda_available", torch.cuda.is_available())
print("device_count", torch.cuda.device_count())
torch.cuda.set_device(0)
p = torch.cuda.get_device_properties(0)
print("device", torch.cuda.get_device_name(0))
print("warp_size", p.warp_size)
print("memory_GiB", p.total_memory / 2**30)
PY

git -C /root/FlagGems rev-parse HEAD \
  > /root/baai-validation/flaggems-commit.log
```

没有保存完整 `env`，也没有把访问凭据或一次性登录信息写进报告。

### 3.2 私有 GitHub 访问探测

```bash
GIT_TERMINAL_PROMPT=0 timeout 20 \
  git ls-remote \
  https://github.com/smuelpeng/baai-operator-development-training-2026.git \
  HEAD
echo $?
```

最后退出码为 `124`。远端没有完成 GitHub 认证，也没有直接取得当前私有仓库 HEAD。

### 3.3 模块 1：Triton 与 GEMM

`/root/lab1` 对应 Gitee `main@1969ed2741c5dbadf8961c85d827e7d05e3f0c30`。

```bash
cd /root/lab1/day1

bash setup.sh \
  > /root/baai-validation/module1-setup.log 2>&1
python3 00_check_env.py \
  > /root/baai-validation/module1-env.log 2>&1
python3 01_torch_device_hello.py \
  > /root/baai-validation/01_torch_device_hello.py.log 2>&1
python3 03_softmax_compare.py \
  > /root/baai-validation/03_softmax_compare.py.log 2>&1
python3 04_layernorm_compare.py \
  > /root/baai-validation/04_layernorm_compare.py.log 2>&1
python3 05_matmul_compare.py \
  > /root/baai-validation/05_matmul_compare.py.log 2>&1

cd /root/lab1/day2
python3 00_baseline.py \
  > /root/baai-validation/gemm-baseline.log 2>&1
```

`02_vector_add_triton.py` 是学生任务，源快照保留 TODO，因此没有把它计入“开箱通过”的脚本列表。

### 3.4 模块 2：Fused Add + RMSNorm

```bash
cd /root
git clone --depth 1 https://gitee.com/sunxt-0719/completed_lab2.git
cd /root/completed_lab2
git rev-parse HEAD
```

Gitee `completed_lab2@3ba8e2d6d94556ba45be9ce15c8751d44986affe` 的 `fused_rms_norm.py` 留下了两行未完成代码。本次只在临时副本补入 README 已给出的答案：

```python
rrms = 1.0 / tl.sqrt(variance + eps)
out = updated * rrms * weight
```

补齐后执行：

```bash
python3 -c 'import torch; from fused_rms_norm import _validate; print("shape=(17,257)", _validate(torch.float32,(17,257))); print("shape=(128,256)", _validate(torch.float32,(128,256)))' \
  > /root/baai-validation/rmsnorm-forward.log 2>&1

python3 grad_kernel.py --m 128 --n 256 --dtype float32 \
  > /root/baai-validation/rmsnorm-backward.log 2>&1

python3 compile_demo.py --m 128 --n 256 --backend eager \
  > /root/baai-validation/rmsnorm-compile.log 2>&1

python3 benchmark.py \
  --m 512 --n 1024 --dtype float32 --warmup 5 --iters 20 \
  > /root/baai-validation/rmsnorm-flaggems-smoke.log 2>&1

python3 benchmark.py \
  --m 4096 --n 4096 --dtype float32 --warmup 10 --iters 30 \
  > /root/baai-validation/rmsnorm-4096-fp32.log 2>&1
```

课程仓库中的参考目录已经补回同样两行；本次动态证据来自远端临时副本。

### 3.5 模块 5：性能实验冒烟

```bash
cd /root
git clone --depth 1 https://gitee.com/yihan-long/day2-lab5.git
cd /root/day2-lab5
git rev-parse HEAD

export LAB_DAY2_ROOT=/root/lab1/day2
python3 00_check_env.py \
  > /root/baai-validation/module5-env.log 2>&1
python3 00_smoke_cuda.py \
  > /root/baai-validation/module5-smoke.log 2>&1
python3 01_benchmark.py \
  --scenario offline \
  --impl torch baseline \
  --num-queries 4 \
  --M 256 --N 256 --K 256 \
  --out /root/baai-validation/module5-benchmark-smoke.json \
  > /root/baai-validation/module5-benchmark-smoke.log 2>&1
```

模块 5 对应 Gitee `main@5fb8557f19fffb14f2c0316f0b26fe4fa026a52f`。

### 3.6 课程整合仓本地 strict validation

```bash
cd /Users/penpen/Documents/算子开发
python3 scripts/validate_repository.py --strict
```

检查结果：source snapshot files verified 447，结构、链接、语法、元数据和选定校验和通过。这个命令没有访问 BI-V150。

## 4. 失败账本（Failure Ledger）

| # | 现象 | 证据 | 原因 | 处理 | 复核结果 |
| --- | --- | --- | --- | --- | --- |
| 1 | 私有 GitHub 仓库没有返回 HEAD | 无交互 `git ls-remote` 最终 `rc=124`；设备登录未在超时前完成 | 远端实例没有 GitHub 凭据 | 不在临时容器保存个人令牌；转用已登记的 Gitee 固定提交 | 三个相关快照完成动态测试；最新 GitHub 整仓仍未验证 |
| 2 | Chrome 向 code-server 上传课程压缩包被拒绝 | 浏览器显示安全策略拒绝 | 受控浏览器限制本次上传 | 没有绕过策略，继续使用实例内已有源码和 Gitee clone | 运行链得到可复核输出 |
| 3 | 名为参考实现的 `completed_lab2` 前向缺少 `rrms`、`out` | 上游 commit `3ba8e2d…` 中两行仍是 TODO | 上游快照未补完 README 给出的答案 | 临时副本补两行；课程仓库参考目录同步修正 | 前向、反向、compile 和 4096² 性能门槛通过 |
| 4 | `512×1024` 融合吞吐只有 FlagGems 的 59.55% | `rmsnorm-flaggems-smoke.log` | 小工作量对启动开销、tile 和并行度更敏感 | 保留 FAIL；按课程正式 shape 单独复测 | `4096×4096` 达到 97.37%，通过 90% 门槛 |
| 5 | 模块 5 环境缺少 `matplotlib` | 环境检查警告；没有 Roofline PNG | 镜像未带可选绘图库 | 没有安装或升级任何平台包；只运行 JSON 路径 | LoadGen JSON 正常写出，退出码 0 |

## 5. 结果矩阵（Result Matrix）

| 样例 | 输入 | 命令 / 配置 | 输出产物 | 运行时间 / 性能 | 判断 |
| --- | --- | --- | --- | --- | --- |
| 环境指纹 | n139 一卡实例 | 第 3.1 节 | `container-environment.log`、`python-version.log`、`python-stack.log` | 不适用 | PASS；确认 6 核、32 GiB、BI-V150 × 1、CoreX 4.4.0 |
| 模块 1 最小环境 | PyTorch 512² MatMul；Triton 2049 元素 add-one | `python3 00_check_env.py` | `module1-env.log` | wall time 未单独保存 | PASS；`Minimal Triton kernel: PASS` |
| Torch 设备样例 | 脚本默认输入 | `01_torch_device_hello.py` | 同名 `.log` | 未保存 | PASS，`rc=0` |
| Softmax | 脚本默认矩阵 | `03_softmax_compare.py` | 同名 `.log` | 未保存 | PASS，`rc=0` |
| LayerNorm | 脚本默认矩阵 | `04_layernorm_compare.py` | 同名 `.log` | 未保存 | PASS，`rc=0` |
| Day 1 MatMul | 脚本默认矩阵 | `05_matmul_compare.py` | 同名 `.log` | 未保存 | PASS，`rc=0` |
| Day 2 GEMM baseline | 1024×1024 @ 1024×1024；fp16 I/O、fp32 accumulator；tile 32³；预分配输出 | `python3 00_baseline.py` | `gemm-baseline.log` | 6.696 TFLOP/s | PASS |
| Day 2 GEMM baseline | 2048×2048 @ 2048×2048；其余同上 | 同上 | `gemm-baseline.log` | 10.728 TFLOP/s | PASS |
| RMSNorm 前向 | float32，`17×257` | 第 3.4 节 `_validate` | `rmsnorm-forward.log` | 最大误差 `4.768e-7` | PASS |
| RMSNorm 前向 | float32，`128×256` | 同上 | `rmsnorm-forward.log` | 最大误差 `9.537e-7` | PASS |
| RMSNorm 反向 | float32，`128×256` | `grad_kernel.py` | `rmsnorm-backward.log` | weight 最大误差 `7.629e-6` | PASS；全部被测梯度在容差内 |
| RMSNorm compile | `128×256`，backend `eager` | `compile_demo.py` | `rmsnorm-compile.log` | graphs=1，graph breaks=0 | PASS |
| RMSNorm 性能冒烟 | float32，`512×1024`；warmup 5、iters 20 | `benchmark.py` | `rmsnorm-flaggems-smoke.log` | 融合 / FlagGems 吞吐比 59.55% | **FAIL**；尺寸敏感，不作为正式门槛 |
| RMSNorm 正式性能 | float32，`4096×4096`；warmup 10、iters 30 | `benchmark.py` | `rmsnorm-4096-fp32.log` | fused 0.5423 ms、495.06 GB/s；FlagGems 0.5280 ms、508.42 GB/s；比值 97.37% | PASS，超过 90% 门槛 |
| 模块 5 环境 | 平台依赖与设备分配 | `00_check_env.py` | `module5-env.log` | 未保存 | PASS，`rc=0`；`matplotlib` 为可选警告 |
| 模块 5 MatMul 冒烟 | 64×64，fp16 | `00_smoke_cuda.py` | `module5-smoke.log` | wall time 未保存 | PASS，`rc=0` |
| 模块 5 LoadGen / Torch | Offline，256³，4 queries | `01_benchmark.py --impl torch baseline ...` | JSON 与日志 | median 0.057 ms，0.589 TFLOP/s | PASS；管线冒烟 |
| 模块 5 LoadGen / Day 2 baseline | Offline，256³，4 queries | 同上 | 同一 JSON 与日志 | median 0.334 ms，0.100 TFLOP/s | PASS；管线冒烟 |
| 课程整合仓静态校验 | 本地共享工作树 | `python3 scripts/validate_repository.py --strict` | stdout | wall time 0.3 s | PASS；仅为本地静态校验 |

小型 LoadGen 使用 4 个 query，只说明 SUT 构建、调度、计时和 JSON 写入可以完成。它不满足正式性能结论所需的样本量。

## 6. 产物清单（Artifact Inventory）

| 产物 | 路径 | 用途 | 检查方式 |
| --- | --- | --- | --- |
| 容器与软件指纹 | `/root/baai-validation/container-environment.log`、`python-version.log`、`python-stack.log`、`flaggems-commit.log` | 设备、配额、版本与模块来源 | 实例释放前用 `less` 查看；避免复制敏感字段 |
| 模块 1 原始日志 | `/root/baai-validation/module1-*.log`、`/root/baai-validation/0*.py.log`、`gemm-baseline.log` | 环境、正确性与 GEMM baseline | `rg -n "PASS|FAIL|TFLOP|Traceback" /root/baai-validation/*.log` |
| 模块 2 原始日志 | `/root/baai-validation/rmsnorm-*.log` | 前向、反向、compile、性能对照 | 查看最大误差、graph 数、延迟、GB/s 和 PASS/FAIL |
| 模块 5 JSON | `/root/baai-validation/module5-benchmark-smoke.json` | 保存每个 SUT 的 latency 与 throughput | `python3 -m json.tool < 文件` |
| 模块 5日志 | `/root/baai-validation/module5-*.log` | 环境、MatMul 和 LoadGen stdout | 查看退出码对应输出与警告 |
| 门户资源卡截图 | [`records/evidence/flagos-online-lab-resource-cards-2026-08-07.png`](../evidence/flagos-online-lab-resource-cards-2026-08-07.png) | 两张实例卡的资源与生命周期证据 | SHA-256 应为 `8f3e52bb077c7c3e11e6027f2dbad70cd4886e8a8198819e0e54bc62b8abdf47` |
| 源码版本清单 | `records/code_repository_inventory.md` | 固定 Gitee 来源与完整 commit | 对照远端 `git rev-parse HEAD` |
| 学生使用指南 | `docs/FLAGOS_ONLINE_LAB.md` | 进入实例、识别配额、运行课程 | 按第 6 节逐项执行 |
| 本报告 | `records/experiments/2026-08-08-flagos-bi-v150-validation.md` | 保存动态结果摘要和结论边界 | 对照本节路径与第 5 节矩阵 |
| 渲染图片 / 视频 | 无 | `matplotlib` 缺失，本次没有输出 PNG 或视频 | 不适用 |
| 模型输出 | 无 | 本次验证算子与基准管线，不加载模型 | 不适用 |

远端 `/root/baai-validation` 是临时产物位置。只有把文件复制出实例并校验哈希后，才能将它们标成长期归档。

## 7. 耗时与成本（Timing and Cost）

| 阶段 | 运行时间 | 说明 |
| --- | --- | --- |
| Model load | 不适用 | 没有加载模型或权重 |
| 环境检查 / 源码准备 | 未单独计时 | 保留命令与结果，未保存统一 wall-clock |
| 推理 / 评测 | 见第 5 节 | 只保存了算子 latency、吞吐或退出结果；部分脚本没有 wall-clock |
| 后处理 / 渲染 | 不适用 | JSON 直接写出；未生成 PNG |
| 总耗时 | 未保存 | 不能由部分 kernel latency 推算整次会话耗时 |
| 额外软件成本 | 0 次平台包安装 | 没有安装或升级 torch、triton、FlagGems、matplotlib |

## 8. 结论边界（Conclusion Boundaries）

### 8.1 由动态产物确认

- n139 是 Ubuntu 24.04.2 Docker 容器，cgroup v1 配额为 6 核和 32 GiB，根文件系统容量 200 G。
- 容器可见一张 32 GiB BI-V150；CoreX / Driver / IX-ML 为 4.4.0；设备接口为 `cuda:0`；warp size 报告为 64。
- PyTorch 2.7.1、Triton 3.1.0 和 `flag_gems` 4.2.1.rc.0 可以完成本报告列出的动态测试。
- Lab 1、completed_lab2 临时补全版和 day2-lab5 的指定脚本得到第 5 节结果。
- 模块 1 只验证了环境、Torch、Softmax、LayerNorm、MatMul 和固定 GEMM baseline；Vector Add 学生 TODO 未运行，不能据此写成“模块 1 全部通过”。
- 模块 2 的正式 `4096×4096` float32 门槛通过；`512×1024` 冒烟性能门槛失败，记录没有被省略。
- 课程整合仓在本地通过 strict validation；这是静态检查结果，不属于 n139 动态证据。

### 8.2 由门户或日志作出的有限判读

- 门户将一卡资源显示为 `activity_29xtu1xs_1996_1785741080`，并给出创建、释放时间；n139 动态资源与该卡一致。
- `/usr/local/corex-4.4.0/` 模块路径、平台驱动和动态运行共同支持“当前解释器使用课程 CoreX 环境”的判读。普通的 `torch==2.7.1` 字符串单独不能证明 wheel 来源。
- 32-thread warp 或其他平台 tile 不能用于解释本次 BI-V150 数据；课程口径和设备属性均为 warp 64。

### 8.3 尚未验证

- FlagOS 镜像 tag、digest 和构建配方。
- 第二台双卡实例只核对了门户资源卡，没有登录该实例验收。
- 私有 GitHub 仓库当前 HEAD 在 n139 的整仓 clone、`make validate` 和全量动态运行。
- 模块 1 的学生 Vector Add TODO、完整 block sweep、autotune 与重复稳定性。
- 模块 2 bfloat16 正式性能矩阵及多轮方差。
- 模块 3 当前只有 IR 阅读练习，本次没有执行编译产物追踪。
- 模块 4 当前只有通信推演练习，本次没有运行多卡通信、collective 或多卡显存测试。
- 模块 5 完整 Offline/Server、Roofline PNG 和平台 profiler。
- 远端原始日志的本地复制与 SHA-256 校验。

### 8.4 支撑更强结论所需的下一轮证据

1. 由平台 API 或管理员提供镜像 tag、digest 与实例到镜像的映射。
2. 使用批准的 GitHub 凭据完成私有仓库 clone，记录 HEAD，并在同一实例运行 `python3 scripts/validate_repository.py`。
3. 在实例释放前下载 `/root/baai-validation`，生成 SHA-256 清单并纳入受控存储。
4. 对正式 shape 重复多轮，保存原始 JSON、warmup、样本数、median、P99 与系统负载。
5. 在双卡实例补做模块 4；在模块 5补做 Roofline 与 profiler，再用同一 workload 对照。
