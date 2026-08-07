# Module 5 Handout: Performance Measurement, Roofline, and Profiling

> **Operations follow the root [`README.md`](README.md) (Chinese).** This handout keeps
> the longer English narrative and discussion prompts; it is not the grading checklist.

**Course:** AI Systems Software & Heterogeneous Computing

**Lab length:** ~2.5–3 hours on BI-V150

**Platform:** Iluvatar BI-V150 (FlagOS / CoreX)
**Deliverables:** `results/*.json`, optional `roofline.png`, filled `REPORT.md`

---

## Framing

Day 1–2 built a correct tiled Triton GEMM and practiced trustworthy timing
(warmup, sync, median). A recurring observation on BI-V150 is that a tuned
Triton GEMM may still lag `torch.matmul` (vendor library).

This module does **not** ask you to invent a new kernel. It asks:

> How do we measure fairly, decide whether we are compute- or memory-bound,
> and use vendor profilers to cross-check that diagnosis?

Method (same as Day 2):

> **Hypothesis → Experiment → Data → Refined understanding**

---

## Learning Goals

After this lab you should be able to:

1. Run Offline and Server-style LoadGen workloads and report latency / throughput / TFLOP/s.
2. Compute GEMM arithmetic intensity and place implementations on a Roofline.
3. Use `ixsmi`, `ixsys`, and `ixkn-cli` at a basic level (Proton is not supported on Iluvatar).
4. Separate **document claims**, **code defaults**, and **measured facts**.
5. Write a short, evidence-based performance conclusion in `REPORT.md`.

---

## Layout

```text
day2-lab5/
  setup.sh  env_corex.sh
  00_check_env.py  00_smoke_cuda.py
  01_benchmark.py  02_roofline.py  03_profiling.py  04_summary.py
  configs/student_todos.py   # YOUR CODE HERE
  kernels/  loadgen/  tools/ # completed helpers — do not edit
  REPORT.md
```

Recommended sibling clone for Day2 GEMM as SUT:

```bash
export LAB_DAY2_ROOT=$PWD/../lab1-day2
```

---

## Step 00: Environment and CUDA smoke (15 min)

```bash
bash setup.sh
python3 00_check_env.py
python3 00_smoke_cuda.py    # must finish with [smoke] OK on FlagOS
```

Local without GPU:

```bash
python3 00_check_env.py --allow-cpu
```

### Discussion

1. Why must we synchronize (`torch.cuda.synchronize`) before trusting a timer?
2. Why is median preferred over mean for short kernel latency samples?

### Notes

- `/dev/iluvatarN` is a **device node name**, not necessarily the CUDA index.
  Inside a one-GPU container the runtime usually still exposes **logical `cuda:0`**.
  Do **not** set `CUDA_VISIBLE_DEVICES=N` just because you see `iluvatarN`.
- If native `bandwidthTest` hangs as well as PyTorch, the instance is broken —
  restart or contact the TA (not a Module-5 code bug).

---

## Step 01: Benchmark / LoadGen (40 min)

Edit [`configs/student_todos.py`](configs/student_todos.py):

```python
# >>> YOUR CODE HERE >>>
OFFLINE_IMPLEMENTATIONS = ["torch", "baseline"]  # add "autotune" for REPORT
# <<< END OF YOUR CODE <<<
```

Defaults stay runnable (`torch` + `baseline`). For the report’s three-way table,
add `"autotune"` (first JIT on CoreX is slow) **after** torch/baseline succeed.

```bash
python3 01_benchmark.py --scenario offline --M 1024 --N 1024 --K 1024
python3 01_benchmark.py --scenario server
```

If FlagOS lacks matplotlib:

```bash
pip install matplotlib
```

`02_roofline.py` still writes JSON without PNG when matplotlib is missing.

Outputs: `results/benchmark_offline.json`, `results/benchmark_server.json`.

### Metrics

| Metric | Meaning |
|--------|---------|
| median / p99 latency (ms) | Per-query latency distribution |
| samples/s | Throughput |
| effective TFLOP/s | `2*M*N*K / (time_s) / 1e12` |

### Discussion

1. Why use shapes like 1024³ rather than tiny matrices for “steady-state” TFLOP/s?
2. What does Offline vs Server (Poisson arrivals) change in the interpretation of p99?

Record numbers in `REPORT.md` §1.

---

## Step 02: Roofline (30 min)

In `configs/student_todos.py`, verify:

```python
PEAK_TFLOPS_FP16 = ...    # π
HBM_BANDWIDTH_GBPS = ...  # β
```

Hint (course outline / datasheet): BI-V150 FP16 peak is often cited as **192 TFLOP/s**;
HBM2e bandwidth must be confirmed from the platform datasheet (placeholder in JSON: 1228.8 GB/s).

```bash
python3 02_roofline.py
```

Outputs: `results/roofline.json`, `results/roofline.png`.

For GEMM:

```text
FLOPs  = 2 * M * N * K
Bytes  ≈ (M*K + K*N + M*N) * sizeof(dtype)   # simple traffic model
I      = FLOPs / Bytes
P_roof = min(π, β * I)
```

### Discussion

1. Is your best Triton point compute-bound or memory-bound under this model?
2. If Triton is far below `torch.matmul`, does Roofline alone explain it? What might be missing?

Fill `REPORT.md` §2.

---

## Step 03: Profiling (40 min)

```bash
# Terminal A
ixsmi dmon -s pucvmet -c 60

# Terminal B
python3 03_profiling.py
```

Optional deeper captures (paths may vary by CoreX install):

```bash
ixsys -t cuda -o results/profiling/gemm.trace python3 03_profiling.py --workload-only
ixkn-cli --set default python3 03_profiling.py --workload-only
```

| Tool | Role |
|------|------|
| `ixsmi` / `ixsmi dmon` | Utilization, memory, power |
| `ixsys` | System / CUDA API-level trace |
| `ixkn-cli` | Kernel-level instruction / occupancy style views |
| Proton | **Not supported** on Iluvatar — skip or note “N/A” |

### Cross-check questions (REPORT §3)

1. Compute-bound or memory-bound?
2. If memory: HBM bandwidth vs on-chip reuse / cache?
3. If compute: ALU utilization vs launch / occupancy issues?
4. One concrete next optimization to try (tile, stages, warps, fusion, …).

---

## Step 04: Summary (15 min)

```bash
python3 04_summary.py
```

Copy key fields into `REPORT.md` and answer the conclusion prompts.

---

## What We Did Not Cover

- KernelGen / LLM-assisted codegen (optional lecture topic)
- Multi-GPU, quantized dtypes, persistent GEMM, fused epilogues

Change one variable at a time if you continue after class.

---

## Completion Checklist

- [ ] `setup.sh` / `00_check_env` / `00_smoke_cuda` OK on FlagOS
- [ ] Offline + Server benchmarks written under `results/`
- [ ] `student_todos.py` includes `autotune` for the three-way table (or CLI `--impl`)
- [ ] Hardware π / β verified
- [ ] Roofline PNG + diagnosis filled
- [ ] At least `ixsmi dmon` observation recorded; ixsys/ixkn attempted if available
- [ ] `REPORT.md` complete with evidence-based conclusions

Reference solutions (if any) live in a teacher-managed `solution/` tree — not in this student repo.
