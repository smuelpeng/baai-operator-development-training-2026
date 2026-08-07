# Fused Add + RMSNorm Lab

This lab follows one operator through the PyTorch stack:

1. Write a fused Triton forward kernel.
2. Derive and implement its backward kernel.
3. Register the operator with the PyTorch dispatcher.
4. Verify that torch.compile captures the registered operator without a graph break.
5. Compare the fused implementation with PyTorch, a split Triton baseline, and FlagGems.

The complete reference implementation remains one directory above this lab.
Do not copy it into the lab directory while solving the exercises.

## Operator Contract

For x and residual of shape [M, N], and weight of shape [N]:

```text
z = x + residual
rrms = 1 / sqrt(mean(z * z) + eps)
out = z * rrms * weight
return out, z
```

Inputs and outputs support float32 and bfloat16.  For bfloat16, arithmetic
inside the RMSNorm reduction must use float32, while the returned tensors keep
the original input dtype.

## Exercise Order

- TODO(1-2), `fused_rms_norm.py`: implement the fused forward kernel and its launch.
- TODO(3-4), `grad_kernel.py`: derive the backward pass and accumulate the
  weight gradient across rows with atomic addition.
- TODO(5), `custom_op.py`: define the dispatcher schema and register CUDA,
  AutogradCUDA, and Meta implementations.
- TODO(6), `compile_demo.py`: this is a checkpoint reminder; the code already
  exercises the registered `torch.ops` entry point after TODO(5) is complete.

`benchmark.py` is intentionally complete.  It is the final acceptance harness
and must use the installed FlagGems implementation rather than a local fallback.

## Validation Commands

Run from this directory with the flaggems conda environment:

```bash
python fused_rms_norm.py
python grad_kernel.py --m 128 --n 256 --dtype float32
python compile_demo.py --m 128 --n 256
python benchmark.py --m 4096 --n 4096 --dtype float32
python benchmark.py --m 4096 --n 4096 --dtype bfloat16
```

Expected targets:

- float32 forward error below 2e-5.
- float32 gradient errors below 2e-5 for the supplied validation shape.
- one Dynamo graph and zero graph breaks in `compile_demo.py`.
- fused throughput at least 90 percent of FlagGems at 4096 by 4096.

The current environment supports Dynamo graph capture with
`torch.compile(..., backend="eager")`.  Its PyTorch and Triton versions do not
support the Inductor backend together; this is an environment issue rather
than a lab requirement.
