"""Register fused add + RMSNorm as a PyTorch dispatcher operator.

PyTorch 2.3 uses ``torch.library.Library`` registrations directly.  The three
registrations below map the same operator schema to CUDA, Autograd, and Meta
dispatch keys so that eager execution, backward, and ``torch.compile`` graph
capture each have an appropriate implementation.
"""

import torch
from torch._subclasses.fake_tensor import FakeTensor

from fused_rms_norm import fused_add_rms_norm
from grad_kernel import FusedAddRMSNorm


def _cuda_impl(x, residual, weight, eps=1e-6):
    """CUDA dispatch: invoke the raw Triton forward kernel."""
    return fused_add_rms_norm(x, residual, weight, eps)


def _autograd_impl(x, residual, weight, eps=1e-6):
    """Autograd dispatch: use the Triton backward implementation."""
    # In PyTorch 2.3, a FakeTensor for a logical CUDA tensor can dispatch to
    # AutogradCUDA before Meta.  Do not launch Triton while Dynamo is tracing.
    if isinstance(x, FakeTensor):
        return _meta_impl(x, residual, weight, eps)
    return FusedAddRMSNorm.apply(x, residual, weight, eps)


def _meta_impl(x, residual, weight, eps=1e-6):
    """Fake/Meta dispatch used by Dynamo while it captures a compiled graph."""
    # Keep this path free of Python shape branches: Dynamo may supply symbolic
    # FakeTensor dimensions while tracing a compiled graph.
    return torch.empty_like(x), torch.empty_like(residual)


# TODO(5): Register torch.ops.lab.fused_add_rms_norm with torch.library.
#
# 1. Create a DEF Library and define this schema:
#    fused_add_rms_norm(Tensor x, Tensor residual, Tensor weight,
#                       float eps=1e-6) -> (Tensor, Tensor)
# 2. Register _cuda_impl for the CUDA dispatch key.
# 3. Register _autograd_impl for AutogradCUDA, not generic Autograd.  In
#    PyTorch 2.3, generic Autograd can intercept FakeTensor dispatch.
# 4. Register _meta_impl for the Meta dispatch key.
# 5. Keep each Library object in a module-level variable.  Dropping the
#    objects releases their registrations.


def fused_add_rms_norm_op(x, residual, weight, eps=1e-6):
    """Python convenience wrapper for ``torch.ops.lab.fused_add_rms_norm``."""
    return torch.ops.lab.fused_add_rms_norm(x, residual, weight, eps)
