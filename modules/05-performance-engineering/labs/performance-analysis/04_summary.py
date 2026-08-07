#!/usr/bin/env python3
"""Step 04: summarize Module-5 results (aligned with lab1-day2 03_summary)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
EXPECTED_MN = 1024


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _warn_stale_server(data: Dict[str, Any]) -> List[str]:
    """Detect local CPU / tiny-shape leftovers that must not be used in REPORT."""
    warns: List[str] = []
    wl = data.get("workload") or {}
    m, n, k = wl.get("M"), wl.get("N"), wl.get("K")
    if m != EXPECTED_MN or n != EXPECTED_MN or k != EXPECTED_MN:
        warns.append(
            f"shape {m}x{n}x{k} ≠ expected {EXPECTED_MN}³ "
            "(likely stale local smoke — delete and re-run Server on BI-V150)"
        )
    if data.get("triton_available") is False:
        warns.append("triton_available=false (unexpected on FlagOS BI-V150)")
    for item in data.get("comparison") or []:
        resolved = str(item.get("resolved") or "")
        device = str((item.get("sut") or {}).get("device") or "")
        if "fallback" in resolved.lower() or device.startswith("cpu"):
            warns.append(
                f"impl={resolved!r} device={device!r} looks like CPU/fallback — "
                "not valid BI-V150 Server data"
            )
            break
    return warns


def _print_benchmark(name: str, data: Optional[Dict[str, Any]]) -> None:
    print(f"\n== {name} ==")
    if not data:
        print("  (missing)")
        return
    wl = data.get("workload", {})
    print(f"  shape: {wl.get('M')}x{wl.get('N')}x{wl.get('K')}  dtype={wl.get('dtype')}")
    print(f"  scenario: {data.get('scenario')}  triton_available={data.get('triton_available')}")
    print(
        f"  {'impl':<28} {'median_ms':>10} {'p99_ms':>10} {'samples/s':>12} {'TFLOP/s':>10}"
    )
    for item in data.get("comparison", []):
        impl = str(item.get("resolved") or item.get("requested") or "?")
        lat = item.get("latency_ms") or {}
        thr = item.get("throughput") or {}
        print(
            f"  {impl:<28} "
            f"{lat.get('median', float('nan')):10.3f} "
            f"{lat.get('p99', float('nan')):10.3f} "
            f"{thr.get('samples_per_s', float('nan')):12.3f} "
            f"{thr.get('effective_tflops', float('nan')):10.3f}"
        )


def main() -> int:
    env = _load(RESULTS / "env_check.json")
    offline = _load(RESULTS / "benchmark_offline.json")
    server = _load(RESULTS / "benchmark_server.json")
    roof = _load(RESULTS / "roofline.json")
    diag = _load(RESULTS / "diagnosis.json")

    print("Module-5 summary")
    print(f"results dir: {RESULTS}")

    if env:
        print("\n== env_check ==")
        print(f"  device: {env.get('device_name')}  cuda={env.get('cuda_available')}")
        fails = [c for c in env.get("checks", []) if c.get("status") == "fail"]
        warns = [c for c in env.get("checks", []) if c.get("status") == "warn"]
        print(f"  fails={len(fails)}  warns={len(warns)}")
    else:
        print("\n== env_check ==\n  (missing; run: bash setup.sh)")

    _print_benchmark("benchmark_offline", offline)
    _print_benchmark("benchmark_server", server)

    stale_server: List[str] = []
    if server:
        stale_server = _warn_stale_server(server)
        if stale_server:
            print("\n[WARN] stale or non-BI-V150 server results:")
            for w in stale_server:
                print(f"  - {w}")
            print(
                "  Fix: rm -f results/benchmark_server.json && "
                "python3 01_benchmark.py --scenario server"
            )

    print("\n== roofline ==")
    if roof:
        for d in roof.get("diagnoses", []):
            print(f"  [{d.get('name')}] bound={d.get('bound')}  below={d.get('below_roofline')}")
            print(f"    {d.get('conclusion')}")
        print(f"  plot: {roof.get('plot')}")
        if roof.get("plot_skipped"):
            print("  (PNG skipped — pip install matplotlib)")
    else:
        print("  (missing; run: python3 02_roofline.py)")

    print("\n== diagnosis ==")
    if diag:
        pre = diag.get("preliminary_answers") or {}
        if pre:
            print(f"  compute_or_memory: {pre.get('compute_or_memory')}")
            print(f"  from_roofline: {pre.get('from_roofline')}")
        tools = (diag.get("profiling") or {}).get("iluvatar_tools") or []
        for t in tools:
            st = "OK" if t.get("available") else "SKIP"
            print(f"  [{st}] {t.get('tool')}")
    else:
        print("  (missing; run: python3 03_profiling.py)")

    summary = {
        "experiment": "04_summary",
        "env_check_present": env is not None,
        "benchmark_offline_present": offline is not None,
        "benchmark_server_present": server is not None,
        "roofline_present": roof is not None,
        "diagnosis_present": diag is not None,
        "server_stale_warnings": stale_server,
        "roofline_bounds": [
            {"name": d.get("name"), "bound": d.get("bound")}
            for d in (roof or {}).get("diagnoses", [])
        ],
    }
    out_path = RESULTS / "summary.json"
    out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n[done] wrote {out_path}")
    print("Fill REPORT.md using the tables above.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
