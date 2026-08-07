#!/usr/bin/env python3
"""实验三：Profiling 封装（天数 ix* 主路径 + 可选 Proton 参考路径）。"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kernels.gemm_triton import build_sut, triton_available


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


def run_sut_workload(M: int, N: int, K: int, dtype: str, iters: int) -> Dict[str, Any]:
    sut, resolved = build_sut(
        kind="baseline",
        M=M,
        N=N,
        K=K,
        dtype=dtype,
        baseline_tile={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
    )
    for _ in range(3):
        sut.run_once()
    sut.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        sut.run_once()
    sut.synchronize()
    elapsed = time.perf_counter() - t0
    return {
        "resolved": resolved,
        "iters": iters,
        "elapsed_s": elapsed,
        "sut": sut.config(),
    }


def try_ixsmi_snapshot(out_dir: Path) -> Dict[str, Any]:
    path = which("ixsmi")
    if not path:
        return {
            "tool": "ixsmi",
            "available": False,
            "skip_reason": "ixsmi not found (expected on Iluvatar / 天数 machines)",
            "suggested_command": "ixsmi dmon -s pucvmet -c 20",
        }
    out_file = out_dir / "ixsmi_query.txt"
    try:
        proc = subprocess.run(
            [path, "-q"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        out_file.write_text(proc.stdout or proc.stderr or "", encoding="utf-8")
        return {
            "tool": "ixsmi",
            "available": True,
            "output": str(out_file),
            "returncode": proc.returncode,
            "suggested_dmon": "ixsmi dmon -s pucvmet -c 30 > results/profiling/ixsmi_dmon.log",
        }
    except Exception as exc:
        return {"tool": "ixsmi", "available": True, "error": str(exc)}


def try_ixsys_commands(out_dir: Path) -> Dict[str, Any]:
    path = which("ixsys")
    if not path:
        return {
            "tool": "ixsys",
            "available": False,
            "skip_reason": "ixsys not found",
            "suggested_command": (
                f"ixsys -t cuda -o {out_dir / 'gemm.trace'} "
                "python3 03_profiling.py --workload-only"
            ),
        }
    return {
        "tool": "ixsys",
        "available": True,
        "note": "tool present; run suggested_command manually for full trace",
        "suggested_command": (
            f"ixsys -t cuda -o {out_dir / 'gemm.trace'} "
            "python3 03_profiling.py --workload-only"
        ),
    }


def try_ixkn_commands(out_dir: Path) -> Dict[str, Any]:
    path = which("ixkn-cli")
    if not path:
        return {
            "tool": "ixkn-cli",
            "available": False,
            "skip_reason": "ixkn-cli not found",
            "suggested_command": (
                "ixkn-cli --set default python3 03_profiling.py --workload-only"
            ),
            "doc_ref": "《软件栈工具使用指南》§6",
        }
    return {
        "tool": "ixkn-cli",
        "available": True,
        "suggested_command": (
            "ixkn-cli --set default python3 03_profiling.py --workload-only"
        ),
        "output_dir": str(out_dir / "ixkn"),
    }


def try_proton(M: int, N: int, K: int, dtype: str, out_dir: Path) -> Dict[str, Any]:
    """Optional NVIDIA/reference path. Iluvatar does not support Proton today."""
    info: Dict[str, Any] = {
        "tool": "proton",
        "platform_note": "天数 BI-V150 暂不支持 Proton；仅作 NVIDIA/参考路径",
        "references": [
            "https://github.com/Deep-Learning-Profiling-Tools/CGO-26-AE",
            "https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonInstrument/IR/TritonInstrumentOps.td",
            "https://github.com/triton-lang/triton/tree/main/third_party/proton",
        ],
    }
    if not triton_available():
        info.update({"available": False, "skip_reason": "triton/CUDA not available"})
        return info
    try:
        import triton.profiler as proton  # type: ignore
    except Exception as exc:
        info.update({"available": False, "skip_reason": f"triton.profiler import failed: {exc}"})
        return info

    try:
        sut, resolved = build_sut(
            kind="baseline",
            M=M,
            N=N,
            K=K,
            dtype=dtype,
            baseline_tile={"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32},
        )
        session_name = str(out_dir / "proton_gemm")
        session = proton.start(session_name, context="shadow")
        with proton.scope("gemm", {"flops": 2 * M * N * K}):
            for _ in range(10):
                sut.run_once()
        sut.synchronize()
        proton.finalize(session)
        info.update(
            {
                "available": True,
                "resolved_sut": resolved,
                "session": session_name,
                "note": "check generated .hatchet / chrome-trace beside session name",
            }
        )
    except Exception as exc:
        info.update({"available": True, "error": str(exc)})
    return info


def merge_diagnosis(
    roofline_path: Path,
    profiling: Dict[str, Any],
    out_path: Path,
) -> Dict[str, Any]:
    diag: Dict[str, Any] = {
        "experiment": "03_diagnosis",
        "questions": [
            "瓶颈是计算还是访存？",
            "如果是访存，是 HBM 带宽不足还是片上缓存命中率低？",
            "如果是计算，是 ALU 利用率低还是指令发射效率低？",
            "（可选）基于诊断结果，提出一个具体的优化方向",
        ],
        "profiling": profiling,
    }
    if roofline_path.exists():
        roof = json.loads(roofline_path.read_text(encoding="utf-8"))
        diag["roofline"] = {
            "diagnoses": roof.get("diagnoses"),
            "points": roof.get("points"),
            "plot": roof.get("plot"),
        }
        # Seed answers from Roofline when profiling tools are absent.
        if roof.get("diagnoses"):
            d0 = roof["diagnoses"][0]
            diag["preliminary_answers"] = {
                "compute_or_memory": d0.get("bound"),
                "from_roofline": d0.get("conclusion"),
                "need_ix_tools_for": [
                    "HBM vs on-chip cache",
                    "ALU util vs issue efficiency",
                ],
            }
    else:
        diag["roofline"] = {
            "missing": str(roofline_path),
            "hint": "run python3 02_roofline.py first",
        }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, ensure_ascii=False), encoding="utf-8")
    return diag


def main() -> int:
    parser = argparse.ArgumentParser(description="Module-5 Exp3: Profiling + diagnosis")
    parser.add_argument("--M", type=int, default=1024)
    parser.add_argument("--N", type=int, default=1024)
    parser.add_argument("--K", type=int, default=1024)
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument(
        "--enable-proton",
        action="store_true",
        help="try Triton Proton (NVIDIA/reference; not supported on 天数)",
    )
    parser.add_argument(
        "--workload-only",
        action="store_true",
        help="only run GEMM workload (for wrapping with ixsys/ixkn)",
    )
    parser.add_argument(
        "--roofline",
        type=Path,
        default=ROOT / "results" / "roofline.json",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=ROOT / "results" / "diagnosis.json",
    )
    args = parser.parse_args()

    out_dir = ROOT / "results" / "profiling"
    out_dir.mkdir(parents=True, exist_ok=True)

    workload = run_sut_workload(args.M, args.N, args.K, args.dtype, args.iters)
    print(f"[workload] {workload['resolved']} elapsed={workload['elapsed_s']:.4f}s")

    if args.workload_only:
        return 0

    tools: List[Dict[str, Any]] = [
        try_ixsmi_snapshot(out_dir),
        try_ixsys_commands(out_dir),
        try_ixkn_commands(out_dir),
    ]
    for t in tools:
        status = "OK" if t.get("available") else "SKIP"
        print(f"[{status}] {t.get('tool')}: {t.get('skip_reason') or t.get('suggested_command') or 'ok'}")

    proton_info: Optional[Dict[str, Any]] = None
    if args.enable_proton:
        proton_info = try_proton(args.M, args.N, args.K, args.dtype, out_dir)
        print(f"[proton] available={proton_info.get('available')} {proton_info.get('skip_reason') or proton_info.get('note') or ''}")
    else:
        proton_info = {
            "tool": "proton",
            "available": False,
            "skip_reason": "disabled by default; pass --enable-proton on NVIDIA",
            "platform_note": "天数 BI-V150 暂不支持 Proton",
        }

    profiling = {
        "workload": workload,
        "iluvatar_tools": tools,
        "proton": proton_info,
        "on_machine_checklist": [
            "Terminal A: ixsmi dmon -s pucvmet -c 60",
            "Terminal B: python3 03_profiling.py --workload-only",
            "Optional: ixsys -t cuda -o results/profiling/gemm.trace python3 03_profiling.py --workload-only",
            "Optional: ixkn-cli --set default python3 03_profiling.py --workload-only",
        ],
    }

    (out_dir / "profiling_summary.json").write_text(
        json.dumps(profiling, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    diag = merge_diagnosis(args.roofline, profiling, args.out)
    print(f"[done] wrote {args.out}")
    if diag.get("preliminary_answers"):
        print("[hint] Roofline preliminary:", diag["preliminary_answers"].get("compute_or_memory"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
