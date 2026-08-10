# Triton Optimization Pass Lab（Iluvatar BI-V150）

本 lab 用两个 Triton kernel 观察它们在 Iluvatar BI-V150 上的编译过程：

- 主线 kernel：blocked GEMM，用来观察数据分工、片上临时存储、提前搬运和矩阵计算单元。
- 对照 kernel：row-wise fused softmax，用来观察融合、mask、求最大值、求和以及多线程合作。

目标不是写最快的 GEMM，而是让学生能把 Triton Python 代码和中间代码对应起来，并回答三个问题：

1. 这段程序在算什么？
2. 数据由哪些线程处理、暂时放在哪里？
3. 最后的矩阵乘怎样交给这张卡的专门计算部件？

## 1. 本 lab 的已验证环境

本 README 对应 `../artifacts/` 中已经生成的结果：

```text
GPU: Iluvatar BI-V150
计算单元数量: 16
显存: 32 GiB
一个 warp: 64 个线程
CoreX: 4.4
矩阵乘输入: fp16
矩阵乘累加与输出: fp32
```

本次 GEMM 配置：

```text
M=N=K=512
BLOCK_M=64, BLOCK_N=64, BLOCK_K=32
num_warps=4, num_stages=3
```

数值检查结果：

```text
matmul 最大绝对误差: 1.1444e-4
softmax 最大绝对误差: 5.588e-9
```

在这个后端中，一个 warp 有 64 个线程。因此 `num_warps=4` 表示一个 CTA 有 256 个线程。

`run.json` 中出现的 `backend='cuda'` 和 `arch=71` 是 CoreX 兼容层的记录方式；判断实际后端时，应以设备名 `Iluvatar BI-V150`、LLIR 中的 `bi-iluvatar-ilurt` 和 `llvm.bi.*` 为准。

## 2. 输出文件及阅读顺序

本次有效的编译产物为：

```text
../artifacts/
  matmul/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
  softmax/
    00_ttir.mlir
    01_ttgir.mlir
    02_llir.ll
    04_cubin.bin
    run.json
```

按下面的顺序阅读：

```text
Triton Python
  → TTIR：程序想算什么
  → TTGIR：线程怎样分工，数据怎样摆放和提前搬运
  → LLIR：BI-V150 的低层准备、等待、同步和矩阵乘加
  → CUBIN：最终设备二进制
```

`04_cubin.bin` 是最终二进制，不适合逐行阅读。课堂阅读的终点是 `02_llir.ll`。

## 3. 从 TTIR 开始读：先看“算什么”

TTIR 与 Triton Python 最接近。此处仍能看到读内存、写内存、矩阵乘和求和等高层操作，还没有真正安排线程和共享临时区。

### 3.1 Softmax

先读 `../artifacts/softmax/00_ttir.mlir`。一个 program 负责一行 softmax，主线是：

```text
找当前行
→ 读入 1024 个元素
→ 求这一行的最大值
→ 每个元素减最大值
→ 求指数
→ 求指数之和
→ 相除
→ 写回这一行
```

学生应回答：

1. 为什么读越界元素时填充负无穷？因为它不会错误地成为最大值。
2. 为什么整个过程不用把中间结果写回显存？因为读、求最大值、指数、求和、除法和写回都在同一个 kernel 内完成。

### 3.2 Matmul

再读 `../artifacts/matmul/00_ttir.mlir`。找这五类操作：

| 在 TTIR 中找什么 | 它表示什么 |
|---|---|
| `tt.get_program_id` | 当前 program 负责 C 的哪一块 |
| `tt.load` | 读入当前需要的 A、B 小块 |
| `scf.for` | 沿 K 维分多轮累加 |
| `tt.dot` | 当前一轮的小矩阵乘 |
| `tt.store` | 把算完的 C 小块写回 |

BI-V150 的 TTIR 比较长，其中有较多 i64 转换、范围比较和辅助函数。这些主要是地址计算与安全检查，不是新的矩阵乘算法。讲解时跳过它们，只追踪上表中的五类操作。

## 4. TTGIR：看“谁做、放哪里、何时搬”

TTGIR 在 TTIR 的基础上加入了线程分工和片上临时存储。这里最重要的不是每个符号的拼写，而是回答：

```text
一块数据由哪些线程处理？
它从显存读出后放在哪里？
什么时候提前搬下一块？
最后怎样送进矩阵计算单元？
```

本后端的 TTGIR 使用 `#triton_gpu.*` 和 `triton_gpu.*` 名字。

### 4.1 `#triton_gpu.blocked`：普通数据的线程分工

matmul 的 `01_ttgir.mlir` 第 1 行定义：

```mlir
#triton_gpu.blocked<{
  sizePerThread = [1, 8],
  threadsPerWarp = [16, 4],
  warpsPerCTA = [2, 2],
  ...
}>
```

可以这样理解：

| 字段 | 最简单的意思 |
|---|---|
| `sizePerThread=[1,8]` | 每个线程处理 1×8 个元素 |
| `threadsPerWarp=[16,4]` | 一个 64-thread warp 按 16×4 的方式分工 |
| `warpsPerCTA=[2,2]` | 四个 warp 按 2×2 的方式合作 |

它常用于读 A/B、生成地址、生成 mask 和普通的逐元素操作。可以把它看成一张“谁负责哪些元素”的分工表。

文件头里还有一个不同的 `#blocked1`。它用于另一种方向的数据访问：A 和 B 在内存中的方向不同，所以它们不一定适合用同一张分工表。

### 4.2 `#triton_gpu.iluvatar_mma`：矩阵乘结果的分工

matmul 的第 4 行定义：

```mlir
#triton_gpu.iluvatar_mma<{
  versionMajor = 1,
  warpsPerCTA = [2, 2],
  instrShape = [16, 16]
}>
```

它不是普通的线程分工，而是给矩阵计算单元准备的分工。含义是：C 的累加结果要按照 BI-V150 矩阵计算单元喜欢的方式分给四个 warp。

课堂上只需解释：

- `warpsPerCTA=[2,2]`：四个 warp 合作完成一块输出；
- `instrShape=[16,16]`：大矩阵乘会被拆成许多个 16×16 的小块；
- 不需要解释 `versionMajor` 的编号。

### 4.3 `#triton_gpu.shared`：A、B 在共享临时区中的摆放

matmul 的第 5～6 行定义：

```mlir
#triton_gpu.shared<{..., useTcu = true}>
```

它说明 A、B 被搬到 CTA 内大家共用的临时区后，应该怎样摆放。`useTcu=true` 可以简单理解成：这个摆法是为 BI-V150 的矩阵计算单元准备的。

第 59～60 行实际创建了两块三阶段临时区：

```mlir
!tt.memdesc<3x64x32xf16, #shared, mutable>
!tt.memdesc<3x32x64xf16, #shared1, mutable>
```

它们表示：

```text
三份 A 小块，每份 64×32
三份 B 小块，每份 32×64
```

第一个维度的 `3` 就是 `num_stages=3`。可以把它讲成三格抽屉：正在用一格计算时，其余抽屉可以提前装入后面要用的数据。

### 4.4 `dot_op`：送入矩阵乘前的最后整理

第 105～107 行是 GEMM 的核心：

```text
从共享临时区取出 A
→ 整理为 A 的 dot_op
从共享临时区取出 B
→ 整理为 B 的 dot_op
tt.dot(A, B, 当前累加值)
→ 新的累加值
```

这里的 `#triton_gpu.dot_op` 表示：A 或 B 已经被整理成矩阵计算单元可以直接使用的形式。里面的 `useSme` 是本后端留下的额外标记，说明 A、B 的内部准备方式可能不同；不需要推测它的微架构细节。

第 107 行的 `tt.dot` 仍然是最重要的锚点：高层矩阵乘没有消失，只是输入已经按硬件喜欢的方式准备好。

## 5. Software Pipelining：计算时准备下一块数据

K 维循环被分成多轮，每轮处理 `BLOCK_K=32`。如果每轮都先等待读完 A/B，再开始计算，矩阵计算单元会经常空等。

这份结果的做法是：

```text
先准备三份 A/B 小块
→ 使用当前一份做矩阵乘
→ 同时搬运后面的 A/B 小块
→ 真正使用前再等待它准备完成
```

TTGIR 中的对应关系：

| 位置 | 操作 | 最简单的解释 |
|---|---|---|
| 第 59～60 行 | `local_alloc` | 准备三格临时抽屉 |
| 第 68、79 行 | `async_copy_global_to_local` | 提前从显存搬 A/B |
| 第 69、80 行 | `async_commit_group` | 提交这一批搬运 |
| 第 103 行 | `async_wait` | 真正要用前确认数据已经到位 |
| 第 104 行 | `scf.for` | 主循环开始 |
| 第 105～107 行 | `local_load` 和 `tt.dot` | 取当前数据并计算 |
| 第 122～141 行 | 新一轮 copy 与 wait | 计算当前块时准备未来块 |
| 第 144～146 行 | 最后等待并释放临时区 | 循环结束时收尾 |

一句话讲解：

> 做当前一题时，先把后面两题的材料拿到桌上。这样当前一题做完后，不必再等材料送来。

`num_stages` 增大并不保证更快：抽屉更多会减少等待，也会占用更多片上空间和寄存器。

## 6. LLIR：确认真正使用了 BI-V150 的矩阵计算路径

`../artifacts/matmul/02_llir.ll` 的开头写着：

```llvm
target triple = "bi-iluvatar-ilurt"
```

这说明它已经进入 BI-V150 的低层目标代码。此时不再重点讨论高层张量，而是观察四类低层操作：

| LLIR 中的线索 | 它表示什么 |
|---|---|
| `addrspace(1)` | 设备上的普通数据地址，例如 A、B、C |
| `addrspace(3) @global_smem` | CTA 内共享的临时区 |
| `llvm.bi.sme.load...` | 按该硬件喜欢的方式准备矩阵输入 |
| `llvm.bi.sl.waitcnt` | 等待先前的准备或搬运完成 |
| `llvm.bi.sl.barrier.alu` | 让合作线程在此处对齐 |
| `llvm.bi.matrix.mad.f32x4.f16x4` | fp16 输入相乘，并累加到 fp32；这是 `tt.dot` 的低层对应 |

本次 matmul 中：

- `sme.load` 出现在约第 178、319、639 行；
- `waitcnt` 出现在约第 203、320、441、640 行；
- `barrier` 在约第 442 行；
- `matrix.mad` 从约第 503 行开始连续出现。

不要逐行读 LLIR。只需让学生完成这条对应：

```text
TTIR 的 tt.dot
→ TTGIR 中准备好的 A/B、共享临时区和 dot_op
→ LLIR 中的 llvm.bi.matrix.mad
```

这说明高层的矩阵乘最终已经交给该设备的矩阵计算部件。

## 7. Softmax 的低层实现

softmax 没有矩阵乘。它的重点是多个线程一起求最大值和求和。

在 TTGIR 中，一个 CTA 有四个 64-thread warp；一个 warp 先处理自己负责的一部分元素，再把多个 warp 的部分结果合并。

在 LLIR 中：

| 线索 | 最简单的解释 |
|---|---|
| `llvm.bi.slb.shfl.idx.b32.i32` | 同一个 warp 内的线程互相交换局部结果 |
| `addrspace(3) @global_smem` | 不同 warp 把各自答案放到共同白板上 |
| `llvm.nvvm.barrier0()` | 所有 warp 到齐后再读取白板 |
| `llvm.exp2.f32` | 指数计算的低层实现 |

课堂可以这样说：

> 一行 1024 个数太多。先让每个小组找自己的最大值，再把四个小组的答案放到白板上，得到整行最大值。求和也做同样的事。整个过程没有把完整的中间数组写回显存，所以它仍然是融合的。

## 8. 学生任务

### 任务 1：改变 tile

改变 `BLOCK_M`、`BLOCK_N`、`BLOCK_K` 后，观察：

- `tt.dot` 的 A、B、C shape 是否变化；
- `iluvatar_mma` 的 `warpsPerCTA` 与 `instrShape` 是否变化；
- A/B 的三阶段临时区 shape 是否变化；
- `dot_op` 中的布局信息是否变化。

学生应解释：tile 变大可以增加一次搬运后可复用的数据，但也会占用更多临时区和寄存器。

### 任务 2：改变 num_warps

观察：

- module attribute 中的 `num-warps`；
- `#triton_gpu.blocked` 中的 `threadsPerWarp` 与 `warpsPerCTA`；
- `#triton_gpu.iluvatar_mma` 中的 `warpsPerCTA`；
- LLIR 中矩阵乘和临时数据准备的变化。

学生应先算出 CTA 线程数：

```text
CTA 线程数 = num_warps × 64
```

然后解释：更多 warp 能让更多线程分担一个 tile，但也会增加资源使用，因此不一定更快。

### 任务 3：改变 num_stages

观察：

- `local_alloc<STAGES x ...>` 的第一个维度；
- 主循环开始前预取了多少数据；
- `async_wait` 的位置和等待深度；
- benchmark 是否发生变化。

学生应解释：stage 数代表提前准备几批数据，而不是有几条独立执行流。

### 任务 4：比较 matmul 和 softmax

matmul 的重点是把 A/B 送进矩阵计算单元；softmax 的重点是融合和合作求最大值、求和。softmax 没有 `tt.dot`，因此不走 `llvm.bi.matrix.mad` 这条路径。

### 任务 5：从低层代码验证结果

对 matmul，找到 `llvm.bi.sme.load`、`llvm.bi.sl.waitcnt`、`llvm.bi.sl.barrier.alu` 和 `llvm.bi.matrix.mad`，解释它们分别对应“准备输入、等待、线程对齐、矩阵乘加”。

对 softmax，找到 shuffle、shared memory、barrier 和指数计算，解释它们分别对应“组内合并、组间合并、同步、指数”。

## 9. 主讲助教的讲解流程

### 第一段：先讲不变的东西

展示 Python kernel，告诉学生：矩阵乘和 softmax 的数学含义没有变。换 GPU 后，变化的是线程分工、临时存储和最后使用的专门计算部件。

### 第二段：TTIR，只追踪主线

矩阵乘只追踪：读 A/B、K 循环、`tt.dot`、写 C。

softmax 只追踪：读一行、最大值、指数、求和、写回。

如果学生被长长的地址检查干扰，提醒他们：那是编译器在核对地址，不是算法主线。

### 第三段：TTGIR，只讲三个问题

1. 谁做：一个 warp 有 64 人，四个 warp 按 2×2 合作。
2. 放哪：A/B 先进入共享临时区，准备了三格缓冲。
3. 何时搬：计算当前块时，提前搬后面的块。

讲解时可以把 shared memory 叫“公共临时桌”，把三阶段叫“三格抽屉”，把 `tt.dot` 叫“交给矩阵计算单元”。

### 第四段：LLIR，只找终点

不逐行阅读。只展示：

```text
准备输入：sme.load
等待准备完成：waitcnt
线程对齐：barrier
真正的矩阵乘：matrix.mad
```

然后把 `matrix.mad` 指回 TTIR 的 `tt.dot`，完成从 Python 到硬件相关操作的闭环。

### 第五段：用 softmax 收尾

强调 softmax 的优势是融合：中间结果不回显存。它需要线程合作求最大值和求和，但不需要矩阵计算单元。

最后让学生用一句话总结：

> Triton Python 说要算什么；TTGIR 安排谁来算、数据放哪；LLIR 把这个安排变成 BI-V150 真正执行的低层操作。

## 10. 容易讲错的点

- 不要把一个 warp 说成 32 个线程；本环境是 64 个线程。
- 不要把 `iluvatar_mma` 的编号解释成另一种设备的编号。
- 不要把 TTIR 中额外的 i64 检查当成算法优化。
- 不要把 `num_stages` 说成多条独立执行流；它是提前准备的数据缓冲深度。
- 不要逐行阅读 LLIR；只追踪输入准备、等待、同步和 `matrix.mad`。
- 不要用不存在于本次有效产物中的低层文本作为硬件证据；本 lab 的阅读终点是 BI LLIR 和最终设备二进制。
