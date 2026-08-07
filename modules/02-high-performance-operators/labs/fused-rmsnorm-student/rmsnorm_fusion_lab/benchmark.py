import argparse
import importlib
import inspect
import time

import torch

from fused_rms_norm import add_rms_norm, fused_add_rms_norm, torch_add_rms_norm


def import_flaggems():
    """Import the installed FlagGems package; never substitute another kernel."""
    try:
        module = importlib.import_module("flag_gems")
    except Exception as exc:
        raise RuntimeError(
            "Unable to import the installed flag_gems package. "
            "Benchmarking is stopped so no compatible local kernel can be mistaken "
            "for a FlagGems result."
        ) from exc

    if not hasattr(module, "fused_add_rms_norm"):
        raise RuntimeError(
            "Installed flag_gems does not export fused_add_rms_norm; "
            "cannot run the requested reference benchmark."
        )

    source_file = inspect.getsourcefile(module.fused_add_rms_norm)
    print(f"Using installed FlagGems package: {module.__file__}")
    print(f"FlagGems fused_add_rms_norm implementation: {source_file}")
    return module


def flaggems_reference(x, residual, weight, eps=1e-6):
    x_work = x.clone()
    residual_work = residual.clone()
    return flaggems_reference.module.fused_add_rms_norm(
        x_work, residual_work, (x.shape[-1],), weight, eps
    )


flaggems_reference.module = None


def flaggems_inplace(x_work, residual_work, weight, eps=1e-6):
    return flaggems_reference.module.fused_add_rms_norm(
        x_work, residual_work, (x_work.shape[-1],), weight, eps
    )


def benchmark(fn, *args, warmup=25, iters=100, **kwargs):
    for _ in range(warmup):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    start = time.perf_counter()
    for _ in range(iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iters


def throughput_gbs(shape, dtype, ms):
    element_size = torch.empty((), dtype=dtype).element_size()
    # x read, residual read, weight read, normed output write, updated residual write.
    bytes_moved = shape[0] * shape[1] * element_size * 4 + shape[1] * element_size
    return bytes_moved / (ms * 1e-3) / 1e9


def run_case(shape, dtype, eps, warmup, iters):
    torch.manual_seed(0)
    x = torch.randn(shape, dtype=dtype, device="cuda")
    residual = torch.randn(shape, dtype=dtype, device="cuda")
    weight = torch.randn((shape[-1],), dtype=dtype, device="cuda")

    ref_out, ref_residual = torch_add_rms_norm(x, residual, weight, eps)
    fused_out, fused_residual = fused_add_rms_norm(x, residual, weight, eps)
    gems_out, gems_residual = flaggems_reference(x, residual, weight, eps)

    fused_diff = (ref_out - fused_out.float()).abs().max().item()
    fused_residual_diff = (ref_residual - fused_residual.float()).abs().max().item()
    gems_diff = (ref_out - gems_out.float()).abs().max().item()

    gems_x = x.clone()
    gems_residual = residual.clone()

    torch_ms = benchmark(
        torch_add_rms_norm, x, residual, weight, eps, warmup=warmup, iters=iters
    )
    split_ms = benchmark(
        add_rms_norm, x, residual, weight, eps, warmup=warmup, iters=iters
    )
    fused_ms = benchmark(
        fused_add_rms_norm, x, residual, weight, eps, warmup=warmup, iters=iters
    )
    gems_ms = benchmark(
        flaggems_inplace,
        gems_x,
        gems_residual,
        weight,
        eps,
        warmup=warmup,
        iters=iters,
    )

    ratio = gems_ms / fused_ms
    return {
        "shape": shape,
        "dtype": str(dtype).replace("torch.", ""),
        "torch_ms": torch_ms,
        "split_ms": split_ms,
        "fused_ms": fused_ms,
        "gems_ms": gems_ms,
        "ratio": ratio,
        "fused_gbs": throughput_gbs(shape, dtype, fused_ms),
        "gems_gbs": throughput_gbs(shape, dtype, gems_ms),
        "fused_diff": fused_diff,
        "fused_residual_diff": fused_residual_diff,
        "gems_diff": gems_diff,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--m", type=int, default=4096)
    parser.add_argument("--n", type=int, default=4096)
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--warmup", type=int, default=25)
    parser.add_argument("--iters", type=int, default=100)
    args = parser.parse_args()

    flaggems_reference.module = import_flaggems()

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    result = run_case((args.m, args.n), dtype, args.eps, args.warmup, args.iters)

    print(f"shape={result['shape']} dtype={result['dtype']} eps={args.eps}")
    print(f"torch stepwise: {result['torch_ms']:.4f} ms")
    print(f"triton split : {result['split_ms']:.4f} ms")
    print(
        f"triton fused : {result['fused_ms']:.4f} ms, "
        f"{result['fused_gbs']:.2f} GB/s"
    )
    print(
        f"FlagGems    : {result['gems_ms']:.4f} ms, "
        f"{result['gems_gbs']:.2f} GB/s"
    )
    print(f"fused / FlagGems throughput ratio: {result['ratio'] * 100:.2f}%")
    print(
        "diffs: "
        f"fused_out={result['fused_diff']:.3e}, "
        f"fused_residual={result['fused_residual_diff']:.3e}, "
        f"flaggems_out={result['gems_diff']:.3e}"
    )

    if result["ratio"] >= 0.90:
        print("PASS: fused kernel reaches at least 90% of FlagGems throughput.")
    else:
        print("FAIL: fused kernel is below 90% of FlagGems throughput.")


if __name__ == "__main__":
    main()
