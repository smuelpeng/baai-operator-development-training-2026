"""Triton tiled GEMM SUT — used on GPU when Triton is available.

Kernel launch layout matches lab1-day2/common.py (2D grid + num_warps/num_stages),
which is the configuration known to work on BI-V150 / CoreX.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch

from .gemm_torch import TorchGEMM, _resolve_dtype, pick_device

_TRITON_OK = False
try:
    import triton
    import triton.language as tl

    _TRITON_OK = True
except Exception:  # pragma: no cover - optional dependency
    triton = None  # type: ignore
    tl = None  # type: ignore

DEFAULT_NUM_WARPS = 4
DEFAULT_NUM_STAGES = 2


def triton_available() -> bool:
    return _TRITON_OK and torch.cuda.is_available()


def _log(msg: str) -> None:
    print(msg, flush=True)


if _TRITON_OK:

    # Same structure as lab1-day2/common.py::matmul_kernel
    @triton.jit
    def _matmul_kernel(
        a_ptr,
        b_ptr,
        c_ptr,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_K: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_n = tl.program_id(1)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
        b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn
        acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
        for k_start in range(0, tl.cdiv(K, BLOCK_K)):
            remain = K - k_start * BLOCK_K
            a = tl.load(
                a_ptrs,
                mask=(offs_m[:, None] < M) & (offs_k[None, :] < remain),
                other=0.0,
            )
            b = tl.load(
                b_ptrs,
                mask=(offs_k[:, None] < remain) & (offs_n[None, :] < N),
                other=0.0,
            )
            acc += tl.dot(a, b)
            a_ptrs += BLOCK_K * stride_ak
            b_ptrs += BLOCK_K * stride_bk
        c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
        tl.store(c_ptrs, acc, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))


@dataclass
class TritonGEMM:
    """System Under Test wrapping a Triton tiled GEMM kernel."""

    M: int
    N: int
    K: int
    dtype: str = "float16"
    device: Optional[str] = None
    tile: Dict[str, int] = field(
        default_factory=lambda: {"BLOCK_M": 32, "BLOCK_N": 32, "BLOCK_K": 32}
    )
    name: str = "triton"
    autotune_configs: Optional[List[Dict[str, int]]] = None
    num_warps: int = DEFAULT_NUM_WARPS
    num_stages: int = DEFAULT_NUM_STAGES

    def __post_init__(self) -> None:
        if not triton_available():
            raise RuntimeError(
                "Triton GEMM requires CUDA + triton. "
                "Use TorchGEMM fallback on this machine."
            )
        from .gemm_torch import init_cuda_device

        self._device = (
            torch.device(self.device)
            if self.device
            else init_cuda_device(0)
        )
        self._dtype = _resolve_dtype(self.dtype)
        _log(f"[sut] allocate tensors on {self._device} shape=({self.M},{self.N},{self.K})")
        self.A = torch.randn(self.M, self.K, device=self._device, dtype=self._dtype)
        _log("[sut] A ready")
        self.B = torch.randn(self.K, self.N, device=self._device, dtype=self._dtype)
        _log("[sut] B ready")
        self.C = torch.empty(self.M, self.N, device=self._device, dtype=self._dtype)
        _log("[sut] C ready; synchronize ...")
        torch.cuda.synchronize()
        _log("[sut] allocate done")
        self._best_tile = dict(self.tile)
        self._autotuned = False
        self._compiled_once = False

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def _launch(self, tile: Dict[str, int]) -> None:
        assert _TRITON_OK
        bm, bn, bk = int(tile["BLOCK_M"]), int(tile["BLOCK_N"]), int(tile["BLOCK_K"])
        grid = (triton.cdiv(self.M, bm), triton.cdiv(self.N, bn))
        if not self._compiled_once:
            _log(
                f"[sut] first Triton launch (JIT compile may take several minutes on CoreX); "
                f"tile={bm}x{bn}x{bk} grid={grid} — GPU util can stay ~0% while compiling"
            )
        _matmul_kernel[grid](
            self.A,
            self.B,
            self.C,
            self.M,
            self.N,
            self.K,
            self.A.stride(0),
            self.A.stride(1),
            self.B.stride(0),
            self.B.stride(1),
            self.C.stride(0),
            self.C.stride(1),
            BLOCK_M=bm,
            BLOCK_N=bn,
            BLOCK_K=bk,
            num_warps=self.num_warps,
            num_stages=self.num_stages,
        )
        if not self._compiled_once:
            self.synchronize()
            self._compiled_once = True
            _log("[sut] first launch finished (compile + sync ok)")

    def run_once(self) -> None:
        self._launch(self._best_tile)

    def autotune(
        self,
        configs: Optional[List[Dict[str, int]]] = None,
        warmup: int = 3,
        repeats: int = 5,
    ) -> Dict[str, int]:
        """Manual tile sweep; returns best config by median latency."""
        import statistics
        import time

        candidates = configs or self.autotune_configs or [self.tile]
        best_tile = candidates[0]
        best_ms = float("inf")
        for i, cfg in enumerate(candidates):
            _log(f"[sut] autotune candidate {i + 1}/{len(candidates)}: {cfg}")
            for _ in range(warmup):
                self._launch(cfg)
            self.synchronize()
            samples = []
            for _ in range(repeats):
                self.synchronize()
                t0 = time.perf_counter()
                self._launch(cfg)
                self.synchronize()
                samples.append((time.perf_counter() - t0) * 1e3)
            med = statistics.median(samples)
            _log(f"[sut]   median={med:.3f} ms")
            if med < best_ms:
                best_ms = med
                best_tile = cfg
        self._best_tile = dict(best_tile)
        self._autotuned = True
        self.tile = dict(best_tile)
        _log(f"[sut] autotune winner: {self._best_tile} ({best_ms:.3f} ms)")
        return self._best_tile

    def config(self) -> Dict[str, Any]:
        return {
            "implementation": self.name,
            "backend": "triton",
            "device": str(self._device),
            "dtype": str(self._dtype).replace("torch.", ""),
            "shape": {"M": self.M, "N": self.N, "K": self.K},
            "tile": dict(self._best_tile),
            "num_warps": self.num_warps,
            "num_stages": self.num_stages,
            "autotuned": self._autotuned,
        }

    def flops(self) -> int:
        return 2 * self.M * self.N * self.K

    def bytes_moved(self) -> int:
        elem = torch.tensor([], dtype=self._dtype).element_size()
        return (self.M * self.K + self.K * self.N + self.M * self.N) * elem


def build_sut(
    kind: str,
    M: int,
    N: int,
    K: int,
    dtype: str,
    baseline_tile: Dict[str, int],
    autotune_configs: Optional[List[Dict[str, int]]] = None,
) -> Tuple[Any, str]:
    """
    Build a SUT by logical name: baseline | autotune | torch.

    Priority:
    1. Day2 ``common.matmul_fixed`` when ``LAB_DAY2_ROOT`` points at lab1-day2
    2. Built-in Triton GEMM when CUDA+Triton available
    3. Torch matmul fallback (local CPU / no Triton)
    """
    kind = kind.lower()
    if kind == "torch":
        _log("[sut] building torch.matmul SUT")
        return TorchGEMM(M=M, N=N, K=K, dtype=dtype, name="torch"), "torch"

    from .day2_adapter import day2_root, try_build_day2_sut

    root = day2_root()
    if root is not None:
        _log(f"[sut] trying Day2 SUT from LAB_DAY2_ROOT={root}")
    day2 = try_build_day2_sut(kind, M, N, K, baseline_tile, autotune_configs)
    if day2 is not None:
        _log(f"[sut] using Day2 SUT resolved={day2[1]}")
        return day2

    if not triton_available():
        return (
            TorchGEMM(M=M, N=N, K=K, dtype=dtype, name=f"torch_fallback_{kind}"),
            f"torch_fallback_{kind}",
        )

    if kind == "baseline":
        _log(f"[sut] building built-in Triton baseline tile={baseline_tile}")
        sut = TritonGEMM(
            M=M, N=N, K=K, dtype=dtype, tile=dict(baseline_tile), name="baseline"
        )
        return sut, "baseline"

    if kind == "autotune":
        _log("[sut] building built-in Triton + manual autotune (slow: many JIT compiles)")
        sut = TritonGEMM(
            M=M,
            N=N,
            K=K,
            dtype=dtype,
            tile=dict(baseline_tile),
            name="autotune",
            autotune_configs=autotune_configs,
        )
        sut.autotune(autotune_configs)
        return sut, "autotune"

    raise ValueError(f"unknown SUT kind: {kind}")
