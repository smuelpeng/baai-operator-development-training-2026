"""Load Day2 GEMM (lab1-day2/common.py) when LAB_DAY2_ROOT is set.

Expected layout (sibling clones)::

    ~/labs/lab1-day2/          # export LAB_DAY2_ROOT=$PWD/../lab1-day2
    ~/labs/lab5-performance/

Does not vendor Day2 sources into this public repo.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def day2_root() -> Optional[Path]:
    raw = os.environ.get("LAB_DAY2_ROOT", "").strip()
    if not raw:
        return None
    path = Path(raw).expanduser().resolve()
    if (path / "common.py").is_file():
        return path
    return None


def load_day2_common():
    root = day2_root()
    if root is None:
        return None
    common_path = root / "common.py"
    mod_name = "lab1_day2_common"
    if mod_name in sys.modules:
        return sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(mod_name, common_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load Day2 common.py from {common_path}")
    module = importlib.util.module_from_spec(spec)
    # Ensure day2 package dir is importable for relative helpers if any.
    root_str = str(root)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)
    spec.loader.exec_module(module)
    sys.modules[mod_name] = module
    return module


@dataclass
class Day2GEMM:
    """SUT wrapping lab1-day2 ``matmul_fixed`` with preallocated output."""

    M: int
    N: int
    K: int
    dtype: str = "float16"
    tile: Dict[str, int] = field(
        default_factory=lambda: {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )
    name: str = "day2_baseline"
    num_warps: int = 4
    num_stages: int = 2

    def __post_init__(self) -> None:
        import torch

        self._common = load_day2_common()
        if self._common is None:
            raise RuntimeError("LAB_DAY2_ROOT is not set or common.py missing")
        if not torch.cuda.is_available():
            raise RuntimeError("Day2 GEMM requires CUDA (BI-V150 / GPU)")
        self._device = getattr(self._common, "DEVICE", "cuda:0")
        torch.cuda.set_device(0)
        self.A, self.B = self._common.make_inputs(self.M, self.N, self.K)
        self.C = torch.empty((self.M, self.N), device=self.A.device, dtype=self.A.dtype)
        self._best_tile = dict(self.tile)

    def synchronize(self) -> None:
        import torch

        torch.cuda.synchronize()

    def run_once(self) -> None:
        t = self._best_tile
        self._common.matmul_fixed(
            self.A,
            self.B,
            block_m=int(t["BLOCK_M"]),
            block_n=int(t["BLOCK_N"]),
            block_k=int(t["BLOCK_K"]),
            num_warps=self.num_warps,
            num_stages=self.num_stages,
            out=self.C,
        )

    def autotune(self, configs: Optional[List[Dict[str, int]]] = None, warmup: int = 3, repeats: int = 5) -> Dict[str, int]:
        import statistics
        import time

        candidates = configs or [self.tile]
        best = candidates[0]
        best_ms = float("inf")
        for cfg in candidates:
            self._best_tile = dict(cfg)
            for _ in range(warmup):
                self.run_once()
            self.synchronize()
            samples = []
            for _ in range(repeats):
                self.synchronize()
                t0 = time.perf_counter()
                self.run_once()
                self.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            med = statistics.median(samples)
            if med < best_ms:
                best_ms = med
                best = cfg
        self._best_tile = dict(best)
        self.tile = dict(best)
        return self._best_tile

    def config(self) -> Dict[str, Any]:
        return {
            "implementation": self.name,
            "backend": "lab1-day2/common.matmul_fixed",
            "lab_day2_root": str(day2_root()),
            "device": str(self._device),
            "dtype": "float16",
            "shape": {"M": self.M, "N": self.N, "K": self.K},
            "tile": dict(self._best_tile),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
        }

    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_moved(self) -> int:
        # fp16 operands + fp16 output (ideal traffic)
        return (self.M * self.K + self.K * self.N + self.M * self.N) * 2


def try_build_day2_sut(
    kind: str,
    M: int,
    N: int,
    K: int,
    baseline_tile: Dict[str, int],
    autotune_configs: Optional[List[Dict[str, int]]] = None,
) -> Optional[Tuple[Any, str]]:
    """Return (sut, resolved) if LAB_DAY2_ROOT usable; else None."""
    if day2_root() is None:
        return None
    kind = kind.lower()
    if kind == "torch":
        return None
    try:
        if kind == "baseline":
            sut = Day2GEMM(M=M, N=N, K=K, tile=dict(baseline_tile), name="day2_baseline")
            return sut, "day2_baseline"
        if kind == "autotune":
            sut = Day2GEMM(
                M=M,
                N=N,
                K=K,
                tile=dict(baseline_tile),
                name="day2_autotune",
            )
            sut.autotune(autotune_configs)
            return sut, "day2_autotune"
    except Exception as exc:
        print(f"[warn] LAB_DAY2_ROOT set but Day2 SUT failed: {exc}")
        return None
    return None
