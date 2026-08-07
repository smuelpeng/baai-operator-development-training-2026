"""MLPerf-inspired LoadGen runner producing structured JSON results."""

from __future__ import annotations

import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .metrics import (
    effective_tflops,
    evaluate_sla,
    samples_per_second,
    summarize_latencies_ms,
)
from .scenarios import generate_offline_schedule, generate_server_schedule
from tools.resource_monitor import ResourceMonitor


def _software_info() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
    }
    try:
        import torch

        info["torch"] = torch.__version__
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["torch"] = None
    try:
        import triton

        info["triton"] = getattr(triton, "__version__", "unknown")
    except Exception:
        info["triton"] = None
    return info


def run_benchmark(
    sut: Any,
    scenario: str,
    num_queries: int,
    warmup: int = 5,
    sync: bool = True,
    target_qps: float = 8.0,
    arrival: str = "poisson",
    sla: Optional[Dict[str, float]] = None,
    resource_interval_s: float = 0.2,
    enable_resource_monitor: bool = False,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Execute Offline or Server scenario against a SUT with run_once()/synchronize().

    Resource monitoring via ixsmi is off by default: concurrent ixsmi while a Triton
    JIT/CUDA workload runs has been observed to stall on some CoreX stacks.
    """
    scenario_norm = scenario.strip().capitalize()
    if scenario_norm == "Offline":
        schedule = generate_offline_schedule(num_queries)
    elif scenario_norm == "Server":
        schedule = generate_server_schedule(num_queries, target_qps=target_qps, arrival=arrival)
    else:
        raise ValueError(f"unsupported scenario: {scenario}")

    print(f"[loadgen] warmup x{warmup} (first call may JIT-compile Triton)...", flush=True)
    for i in range(warmup):
        sut.run_once()
        print(f"[loadgen] warmup {i + 1}/{warmup} done", flush=True)
    if sync:
        sut.synchronize()
    print("[loadgen] warmup finished, starting measured queries", flush=True)

    monitor: Optional[ResourceMonitor] = None
    if enable_resource_monitor:
        monitor = ResourceMonitor(interval_s=resource_interval_s)
        monitor.start()
    else:
        print("[loadgen] resource monitor disabled (set resources.enable=true to turn on)", flush=True)

    latencies_ms: List[float] = []
    per_query: List[Dict[str, Any]] = []
    wall_start = time.perf_counter()

    if scenario_norm == "Offline":
        for q in schedule:
            if sync:
                sut.synchronize()
            t0 = time.perf_counter()
            sut.run_once()
            if sync:
                sut.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1e3
            latencies_ms.append(dt_ms)
            per_query.append(
                {
                    "query_id": q.query_id,
                    "issue_time_s": q.issue_time_s,
                    "latency_ms": dt_ms,
                }
            )
            if q.query_id == 0 or (q.query_id + 1) % 8 == 0:
                print(
                    f"[loadgen] query {q.query_id + 1}/{len(schedule)} "
                    f"latency={dt_ms:.3f} ms",
                    flush=True,
                )
    else:
        # Server: wait until issue time relative to wall_start, then run.
        for q in schedule:
            target = wall_start + q.issue_time_s
            now = time.perf_counter()
            if target > now:
                time.sleep(target - now)
            if sync:
                sut.synchronize()
            t0 = time.perf_counter()
            sut.run_once()
            if sync:
                sut.synchronize()
            dt_ms = (time.perf_counter() - t0) * 1e3
            latencies_ms.append(dt_ms)
            per_query.append(
                {
                    "query_id": q.query_id,
                    "issue_time_s": q.issue_time_s,
                    "latency_ms": dt_ms,
                }
            )

    wall_s = time.perf_counter() - wall_start
    resources = monitor.stop() if monitor is not None else {"sample_count": 0, "enabled": False}

    latency_summary = summarize_latencies_ms(latencies_ms)
    median_ms = latency_summary["median"]
    flops = int(sut.flops())
    tflops = effective_tflops(flops, median_ms)

    result: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "scenario": scenario_norm,
        "software": _software_info(),
        "sut": sut.config() if hasattr(sut, "config") else {},
        "workload": {
            "num_queries": num_queries,
            "warmup": warmup,
            "target_qps": target_qps if scenario_norm == "Server" else None,
            "arrival": arrival if scenario_norm == "Server" else None,
            "flops_per_query": flops,
            "bytes_per_query": int(sut.bytes_moved()) if hasattr(sut, "bytes_moved") else None,
        },
        "latency_ms": latency_summary,
        "throughput": {
            "samples_per_s": samples_per_second(num_queries, wall_s),
            "wall_time_s": wall_s,
            "effective_tflops": tflops,
            "median_latency_ms": median_ms,
        },
        "resources": resources,
        "sla": evaluate_sla(latency_summary, sla),
        "queries": per_query,
    }
    if extra:
        result.update(extra)
    return result


def save_result(result: Dict[str, Any], path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
