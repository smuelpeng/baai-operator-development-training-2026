#!/usr/bin/env python3
"""Exp 2: Roofline plot and bottleneck diagnosis from Exp 1 results."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.student_todos import HBM_BANDWIDTH_GBPS, PEAK_TFLOPS_FP16
from tools.plot_roofline import (
    arithmetic_intensity,
    diagnose,
    dtype_nbytes,
    gemm_bytes,
    gemm_flops,
    plot_roofline,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Module-5 Exp2: Roofline analysis")
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "results" / "benchmark_offline.json",
        help="benchmark JSON from experiment 01",
    )
    parser.add_argument(
        "--hardware",
        type=Path,
        default=ROOT / "configs" / "hardware_bi_v150.json",
        help="hardware peak parameters JSON",
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=ROOT / "results" / "roofline.json",
    )
    parser.add_argument(
        "--out-png",
        type=Path,
        default=ROOT / "results" / "roofline.png",
    )
    args = parser.parse_args()

    if not args.input.exists():
        print(f"[error] input not found: {args.input}")
        print("        run: python3 01_benchmark.py --scenario offline")
        return 1

    bench = json.loads(args.input.read_text(encoding="utf-8"))
    hw = json.loads(args.hardware.read_text(encoding="utf-8"))
    # Prefer student TODO peaks; JSON is the datasheet mirror / fallback.
    peak = float(PEAK_TFLOPS_FP16 if PEAK_TFLOPS_FP16 else hw["peak_tflops_fp16"])
    bw = float(HBM_BANDWIDTH_GBPS if HBM_BANDWIDTH_GBPS else hw["hbm_bandwidth_GBps"])
    print(f"[info] peak_tflops_fp16={peak}  hbm_bandwidth_GBps={bw}", flush=True)

    wl = bench.get("workload", {})
    M, N, K = int(wl["M"]), int(wl["N"]), int(wl["K"])
    dtype = wl.get("dtype", "float16")
    # If CPU fallback promoted to fp32, prefer sut dtype when present.
    nbytes = dtype_nbytes(dtype)
    flops = gemm_flops(M, N, K)

    points = []
    diagnoses = []
    for item in bench.get("comparison", []):
        name = item.get("resolved") or item.get("requested") or "unknown"
        sut = item.get("sut") or {}
        sut_dtype = sut.get("dtype", dtype)
        try:
            nb = dtype_nbytes(sut_dtype)
        except ValueError:
            nb = nbytes
        bytes_moved = gemm_bytes(M, N, K, nb)
        intensity = arithmetic_intensity(flops, bytes_moved)
        tflops = float(item.get("throughput", {}).get("effective_tflops") or float("nan"))
        diag = diagnose(intensity, tflops, peak, bw)
        diag["name"] = name
        diag["tile"] = sut.get("tile")
        diagnoses.append(diag)
        points.append({"name": name, "intensity": intensity, "tflops": tflops})
        print(f"[{name}] I={intensity:.2f} FLOP/Byte  measured={tflops:.3f} TFLOP/s")
        print(f"         {diag['conclusion']}")

    png = plot_roofline(
        points,
        peak_tflops=peak,
        bandwidth_GBps=bw,
        out_path=args.out_png,
        title=f"Roofline GEMM {M}x{N}x{K} - BI-V150",
    )

    payload = {
        "experiment": "02_roofline",
        "input": str(args.input),
        "hardware": {
            "name": hw.get("name"),
            "arch": hw.get("arch"),
            "peak_tflops_fp16": peak,
            "hbm_bandwidth_GBps": bw,
            "source_json": str(args.hardware),
        },
        "workload": {"M": M, "N": N, "K": K, "dtype": dtype, "flops": flops},
        "points": points,
        "diagnoses": diagnoses,
        "plot": str(png) if png else None,
        "plot_skipped": png is None,
    }
    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] wrote {args.out_json}")
    if png:
        print(f"[done] wrote {png}")
    else:
        print("[done] roofline JSON written without PNG (install matplotlib to plot)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
