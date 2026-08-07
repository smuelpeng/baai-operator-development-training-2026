#!/usr/bin/env python3
"""Exp 1: MLPerf-style LoadGen GEMM benchmark (Offline / Server)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from configs.student_todos import OFFLINE_IMPLEMENTATIONS, SERVER_IMPLEMENTATIONS
from kernels.gemm_triton import build_sut, triton_available
from loadgen.runner import run_benchmark, save_result


def load_config(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="Module-5 Exp1: GEMM LoadGen Benchmark")
    parser.add_argument(
        "--scenario",
        choices=["offline", "server"],
        default="offline",
        help="benchmark scenario",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="path to config JSON (default: configs/gemm_{scenario}.json)",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="output JSON path (default: results/benchmark_{scenario}.json)",
    )
    parser.add_argument(
        "--M", type=int, default=None, help="override M"
    )
    parser.add_argument(
        "--N", type=int, default=None, help="override N"
    )
    parser.add_argument(
        "--K", type=int, default=None, help="override K"
    )
    parser.add_argument(
        "--num-queries", type=int, default=None, help="override query count"
    )
    parser.add_argument(
        "--impl",
        nargs="+",
        default=None,
        help="override SUT list, e.g. --impl torch  or  --impl baseline torch",
    )
    args = parser.parse_args()

    cfg_path = args.config or (ROOT / "configs" / f"gemm_{args.scenario}.json")
    cfg = load_config(cfg_path)
    wl = cfg["workload"]
    meas = cfg.get("measurement", {})
    sut_cfg = cfg.get("sut", {})

    M = args.M or wl["M"]
    N = args.N or wl["N"]
    K = args.K or wl["K"]
    dtype = wl.get("dtype", "float16")
    num_queries = args.num_queries or wl.get("num_queries", 32)
    warmup = int(meas.get("warmup", 5))
    sync = bool(meas.get("sync", True))
    # Prefer student TODO list; JSON remains a readable default mirror.
    if args.impl:
        implementations = args.impl
    elif args.scenario == "server":
        implementations = list(SERVER_IMPLEMENTATIONS)
    else:
        implementations = list(OFFLINE_IMPLEMENTATIONS)
    baseline_tile = sut_cfg.get(
        "baseline_tile", {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )
    autotune_configs = sut_cfg.get("autotune_configs")
    sla = cfg.get("sla")
    resource_interval = float(cfg.get("resources", {}).get("sample_interval_s", 0.2))
    enable_resources = bool(cfg.get("resources", {}).get("enable", False))

    print(f"[info] config={cfg_path}", flush=True)
    print(f"[info] triton_available={triton_available()}", flush=True)
    print(
        f"[info] shape=({M},{N},{K}) dtype={dtype} scenario={cfg['scenario']}",
        flush=True,
    )
    print(f"[info] implementations={list(implementations)}", flush=True)
    print(
        "[info] tip: first Triton JIT on CoreX may take minutes (GPU util≈0%). "
        "Smoke: python3 01_benchmark.py --impl torch --num-queries 4. "
        "Prefer: export LAB_DAY2_ROOT=$PWD/../lab1-day2",
        flush=True,
    )

    runs = []
    for kind in implementations:
        print(f"[run] SUT={kind} ...", flush=True)
        sut, resolved = build_sut(
            kind=kind,
            M=M,
            N=N,
            K=K,
            dtype=dtype,
            baseline_tile=baseline_tile,
            autotune_configs=autotune_configs,
        )
        print(f"[run] SUT ready resolved={resolved}", flush=True)
        if resolved != kind:
            print(f"[warn] requested {kind}, resolved to {resolved} (fallback)", flush=True)

        result = run_benchmark(
            sut=sut,
            scenario=cfg["scenario"],
            num_queries=num_queries,
            warmup=warmup,
            sync=sync,
            target_qps=float(wl.get("target_qps", 8.0)),
            arrival=str(wl.get("arrival", "poisson")),
            sla=sla,
            resource_interval_s=resource_interval,
            enable_resource_monitor=enable_resources,
            extra={
                "hardware_hint": "see configs/hardware_bi_v150.json",
                "requested_implementation": kind,
                "resolved_implementation": resolved,
            },
        )
        # Drop per-query detail from nested compare to keep file smaller; keep in individual later if needed
        runs.append(result)
        lat = result["latency_ms"]
        thr = result["throughput"]
        print(
            f"  median={lat['median']:.3f} ms  "
            f"p99={lat['p99']:.3f} ms  "
            f"samples/s={thr['samples_per_s']:.3f}  "
            f"TFLOP/s={thr['effective_tflops']:.3f}"
        )

    # Compact comparison table (without full query lists for readability)
    comparison = []
    for r in runs:
        comparison.append(
            {
                "requested": r.get("requested_implementation"),
                "resolved": r.get("resolved_implementation"),
                "sut": r.get("sut"),
                "latency_ms": r.get("latency_ms"),
                "throughput": r.get("throughput"),
                "resources": r.get("resources"),
                "sla": r.get("sla"),
            }
        )

    out_path = args.out or (ROOT / "results" / f"benchmark_{args.scenario}.json")
    payload = {
        "experiment": "01_benchmark",
        "config_path": str(cfg_path),
        "scenario": cfg["scenario"],
        "workload": {
            "M": M,
            "N": N,
            "K": K,
            "dtype": dtype,
            "num_queries": num_queries,
        },
        "triton_available": triton_available(),
        "comparison": comparison,
        "runs": runs,
    }
    save_result(payload, out_path)
    print(f"[done] wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
