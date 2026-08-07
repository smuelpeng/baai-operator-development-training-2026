"""{{TASK_NAME}} starter: reference, Triton kernel, and launch wrapper.

The initial contract is vector addition. Replace it only after TASK.md records
the new operator semantics and validation matrix.
"""

from __future__ import annotations

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only machines can still develop the reference/tests.
    triton = None
    tl = None


def torch_reference(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Correctness reference; keep it outside benchmark timing."""
    return x + y


if triton is not None:

    @triton.jit
    def _kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
        """Replace this TODO with the smallest correct implementation."""
        # Suggested first implementation:
        # pid → offsets → tail mask → load → compute → store.
        pass
else:
    _kernel = None


def operator(x: torch.Tensor, y: torch.Tensor, block_size: int = 256) -> torch.Tensor:
    """Validate the contract, allocate output, and launch the kernel."""
    if x.shape != y.shape:
        raise ValueError(f"shape mismatch: {tuple(x.shape)} vs {tuple(y.shape)}")
    if x.dtype != y.dtype:
        raise ValueError(f"dtype mismatch: {x.dtype} vs {y.dtype}")
    if not x.is_cuda or not y.is_cuda:
        raise ValueError("the Triton candidate requires accelerator tensors")
    if triton is None or _kernel is None:
        raise RuntimeError("Triton is unavailable in this Python environment")
    x = x.contiguous()
    y = y.contiguous()
    out = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), block_size),)
    _kernel[grid](x, y, out, x.numel(), BLOCK_SIZE=block_size)
    return out
