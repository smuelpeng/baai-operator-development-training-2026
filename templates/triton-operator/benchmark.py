"""Fair steady-state benchmark starter for {{TASK_NAME}}."""

from __future__ import annotations

import statistics
import time

try:
    import torch
except ImportError:
    torch = None

if torch is not None:
    from kernel import operator, torch_reference


def bench_ms(fn, warmup: int = 10, samples: int = 20, inner: int = 10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    times = []
    for _ in range(samples):
        start = time.perf_counter()
        for _ in range(inner):
            fn()
        torch.cuda.synchronize()
        times.append((time.perf_counter() - start) * 1e3 / inner)
    return statistics.median(times)


def main() -> int:
    if torch is None:
        print("SKIP: PyTorch is unavailable; run in the target course environment")
        return 0
    if not torch.cuda.is_available():
        print("SKIP: target accelerator unavailable")
        return 0
    shape = (1 << 20,)
    x = torch.randn(shape, device="cuda:0", dtype=torch.float16)
    y = torch.randn(shape, device="cuda:0", dtype=torch.float16)

    expected = torch_reference(x, y)
    actual = operator(x, y)
    torch.testing.assert_close(actual, expected, atol=1e-3, rtol=1e-3)

    reference_ms = bench_ms(lambda: torch_reference(x, y))
    candidate_ms = bench_ms(lambda: operator(x, y))
    print(f"shape={shape} dtype={x.dtype}")
    print(f"reference median_ms={reference_ms:.6f}")
    print(f"candidate median_ms={candidate_ms:.6f}")
    print(f"speedup={reference_ms / candidate_ms:.3f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
