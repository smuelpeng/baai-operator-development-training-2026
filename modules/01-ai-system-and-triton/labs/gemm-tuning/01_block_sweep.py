"""Step 01: probe CoreX limits, then design a controlled block sweep."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from common import (
    DEFAULT_NUM_STAGES,
    DEFAULT_NUM_WARPS,
    bench_ms,
    check_close,
    compact_error,
    make_inputs,
    make_reference,
    matmul_fixed,
    tflops,
    tile_footprint_kib,
)

# >>> YOUR CODE HERE >>>
# TODO: retain the baseline and add 10-20 power-of-two candidates that isolate
# BM, BN, and BK. Include symmetric and transposed asymmetric pairs. Copy only
# configurations that PASS into Step 02.
PERF_TILES = [
    (32, 32, 32),
]
# <<< END OF YOUR CODE <<<

# Run as provided. These deliberately push different footprint proxies. Do not
# label a failure as a particular physical buffer until the CoreX error says so.
STRESS_TILES = [
    (128, 128, 32), (128, 128, 64), (128, 128, 128),
    (256, 64, 32), (64, 256, 32), (256, 256, 16),
    (64, 64, 256), (64, 64, 512),
]

SHAPES = [
    (1024, 1024, 1024),
    (2048, 512, 1024),
    (512, 2048, 256),
]


def probe_config(a, b, expected, bm, bn, bk, measure):
    """Classify compile/runtime, correctness, and performance separately."""
    M, K = a.shape
    N = b.shape[1]
    out = torch.empty((M, N), device=a.device, dtype=a.dtype)
    fn = lambda: matmul_fixed(
        a, b, bm, bn, bk,
        num_warps=DEFAULT_NUM_WARPS,
        num_stages=DEFAULT_NUM_STAGES,
        out=out,
    )
    try:
        actual = fn()
        torch.cuda.synchronize()
    except Exception as exc:
        return {"status": "COMPILE/RUNTIME FAIL", "error": compact_error(exc)}
    try:
        diff = check_close(actual, a, b, expected)
    except (AssertionError, RuntimeError) as exc:
        return {"status": "WRONG RESULT", "error": compact_error(exc)}
    result = {"status": "PASS", "diff": diff}
    if measure:
        try:
            ms = bench_ms(fn, warmup=2, samples=5, inner=10)
            result.update(ms=ms, throughput=tflops(M, N, K, ms))
        except Exception as exc:
            return {"status": "BENCHMARK FAIL", "error": compact_error(exc)}
    return result


def print_result(tile, result, M, N, K):
    bm, bn, bk = tile
    acc_kib, operands_kib = tile_footprint_kib(bm, bn, bk)
    loops = (K + bk - 1) // bk
    prefix = (f"{bm:3d} {bn:3d} {bk:3d}  acc~{acc_kib:6.1f} KiB  "
              f"operands~{operands_kib:6.1f} KiB  K-loops={loops:3d}")
    if result["status"] == "PASS" and "ms" in result:
        print(f"{prefix}  PASS  {result['ms']:8.4f} ms  "
              f"{result['throughput']:7.3f} TFLOP/s  diff={result['diff']:.3e}")
    elif result["status"] == "PASS":
        print(f"{prefix}  PASS  diff={result['diff']:.3e}")
    else:
        print(f"{prefix}  {result['status']}: {result['error']}")


def resource_probe():
    print("=" * 100)
    print("Part A: BI-V150/CoreX resource probes (do not import Ascend limits)")
    print("=" * 100)
    M = N = K = 1024
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    for tile in STRESS_TILES:
        result = probe_config(a, b, expected, *tile, measure=False)
        print_result(tile, result, M, N, K)


def sweep_shape(M, N, K):
    print("\n" + "=" * 100)
    print(f"Part B: controlled performance sweep at shape={(M, N, K)}")
    print("=" * 100)
    a, b = make_inputs(M, N, K)
    expected = make_reference(a, b)
    rows = []
    for tile in PERF_TILES:
        result = probe_config(a, b, expected, *tile, measure=True)
        print_result(tile, result, M, N, K)
        if result["status"] == "PASS":
            rows.append((result["ms"], *tile, result["diff"], result["throughput"]))
    if rows:
        best = min(rows)
        print(f"best tile={(best[1], best[2], best[3])}, {best[0]:.4f} ms")
        print("PASS tiles to copy into Step 02:")
        print([tuple(row[1:4]) for row in sorted(rows)])
    return rows


def main():
    torch.cuda.set_device(0)
    resource_probe()
    for shape in SHAPES:
        sweep_shape(*shape)


if __name__ == "__main__":
    main()
