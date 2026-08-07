# Fused Add + RMSNorm Reference

This lab follows one operator through the PyTorch stack:

1. Write a fused Triton forward kernel.
2. Derive and implement its backward kernel.
3. Register the operator with the PyTorch dispatcher.
4. Verify that torch.compile captures the registered operator without a graph break.
5. Compare the fused implementation with PyTorch, a split Triton baseline, and FlagGems.

This directory is the runnable reference implementation. The untouched student
exercise is the sibling directory `../fused-rmsnorm-student/`; keep that copy
when demonstrating the TODO workflow.

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

## Reference Coverage

- `fused_rms_norm.py`: complete fused forward kernel and numerical checks;
- `grad_kernel.py`: complete backward kernel and gradient comparison;
- `custom_op.py`: PyTorch dispatcher registration;
- `compile_demo.py`: eager and `torch.compile` graph-capture validation;
- `benchmark.py`: PyTorch, split Triton, fused Triton and FlagGems comparison.

`benchmark.py` is intentionally complete.  It is the final acceptance harness
and must use the installed FlagGems implementation rather than a local fallback.

## Validation Commands

From the repository root, enter this directory and use the FlagGems environment
supplied by the course image:

```bash
cd modules/02-high-performance-operators/labs/fused-rmsnorm-reference
python3 fused_rms_norm.py
python3 grad_kernel.py --m 128 --n 256 --dtype float32
python3 compile_demo.py --m 128 --n 256
python3 benchmark.py --m 4096 --n 4096 --dtype float32
python3 benchmark.py --m 4096 --n 4096 --dtype bfloat16
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

## Forward Kernel Detail

The two lines below are the completed normalization step in
`_fused_add_rms_norm_kernel`:

```python
rrms = 1.0 / tl.sqrt(variance + eps)
out = updated * rrms * weight
```

The first line computes the reciprocal RMS scaling factor. The second line
normalizes the updated residual and applies the learnable RMSNorm weight.
