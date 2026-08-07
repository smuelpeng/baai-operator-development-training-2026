"""Correctness matrix for {{TASK_NAME}}."""

from __future__ import annotations

import argparse

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from kernel import operator, torch_reference


SHAPES = [(1,), (127,), (128,), (129,), (4097,)]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()

    if torch is None:
        print("SKIP: PyTorch is unavailable; run in the target course environment")
        return 0
    if not torch.cuda.is_available() and not args.reference_only:
        print("SKIP: target accelerator unavailable; run with --reference-only for CPU checks")
        return 0

    device = "cpu" if args.reference_only else "cuda:0"
    for dtype in (torch.float32, torch.float16):
        for shape in SHAPES:
            torch.manual_seed(0)
            x = torch.randn(shape, device=device, dtype=dtype)
            y = torch.randn(shape, device=device, dtype=dtype)
            expected = torch_reference(x, y)
            if args.reference_only:
                actual = torch_reference(x, y)
            else:
                actual = operator(x, y)
            if not torch.isfinite(actual).all():
                raise AssertionError(f"non-finite output: dtype={dtype}, shape={shape}")
            torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)
            print(f"PASS dtype={dtype} shape={shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
