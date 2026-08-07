"""Print a reproducible BI-V150/CoreX fingerprint and run two smoke tests."""
import os
import platform
import subprocess
import sys
import time

import torch
import triton
import triton.language as tl

DEVICE = "cuda:0"
COREX_PATH_TOKENS = ("corex", "iluvatar")


def corex_runtime_evidence(*module_files):
    """Return concise path evidence for a CoreX runtime/image.

    CoreX 4.4 images may expose upstream-looking package versions such as
    ``torch==2.7.1``. A version suffix is therefore useful evidence, but it is
    not the only supported signal. This helper deliberately reports path
    evidence rather than claiming that an arbitrary wheel is platform-adapted.
    """

    evidence = []
    for label, value in module_files:
        value = str(value or "")
        if value and any(token in value.lower() for token in COREX_PATH_TOKENS):
            evidence.append(f"{label}={value}")

    corex_home = os.environ.get("COREX_HOME", "").strip()
    if corex_home:
        evidence.append(f"COREX_HOME={corex_home}")

    for name in ("LD_LIBRARY_PATH", "LIBRARY_PATH", "PYTHONPATH", "LD_PRELOAD"):
        value = os.environ.get(name, "")
        hits = [
            entry
            for entry in value.split(os.pathsep)
            if entry and any(token in entry.lower() for token in COREX_PATH_TOKENS)
        ]
        if hits:
            evidence.append(f"{name} includes {hits[0]}")
    return evidence


def report_corex_package(name, version, module_file, runtime_evidence):
    """Classify package evidence without treating a missing suffix as failure."""

    version = str(version or "")
    module_file = str(module_file or "")
    direct_evidence = "+corex" in version.lower() or any(
        token in module_file.lower() for token in COREX_PATH_TOKENS
    )
    if direct_evidence:
        print(f"{name} CoreX package evidence: version/path marker present")
    elif runtime_evidence:
        print(
            f"INFO: {name} {version} has no '+corex' suffix; "
            "the active CoreX image/runtime is identified by path evidence."
        )
        print(
            "NOTE: runtime paths identify the platform context but do not, by "
            "themselves, prove wheel provenance. Keep the course image unchanged."
        )
    else:
        print(
            f"WARNING: {name} has no '+corex' marker and no CoreX path evidence; "
            "verify the official platform image before running the labs."
        )


def run_version(command):
    try:
        result = subprocess.run(
            command, capture_output=True, text=True, check=False, timeout=10
        )
        return (result.stdout or result.stderr).strip()
    except FileNotFoundError:
        return "command not found"
    except subprocess.TimeoutExpired:
        return "command timed out"


@triton.jit
def add_one_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    tl.store(y_ptr + offsets, x + 1.0, mask=mask)


def main():
    print("Platform:", platform.platform())
    print("Python:", sys.version.split()[0])
    print("PyTorch:", torch.__version__)
    print("Triton:", triton.__version__)
    print("torch.__file__:", torch.__file__)
    print("triton.__file__:", triton.__file__)
    print("ixsmi -L:\n", run_version(["ixsmi", "-L"]))
    assert torch.cuda.is_available(), "CoreX accelerator is not available"
    assert torch.cuda.device_count() >= 1
    torch.cuda.set_device(0)
    properties = torch.cuda.get_device_properties(0)
    print("Selected:", DEVICE, torch.cuda.get_device_name(0))
    print("Reported warp size:", getattr(properties, "warp_size", "unavailable"))
    print("Total memory (GiB):", f"{properties.total_memory / 2**30:.1f}")
    runtime_evidence = corex_runtime_evidence(
        ("torch.__file__", torch.__file__),
        ("triton.__file__", triton.__file__),
    )
    if runtime_evidence:
        print("CoreX runtime evidence:", "; ".join(runtime_evidence))
    report_corex_package(
        "PyTorch", torch.__version__, torch.__file__, runtime_evidence
    )
    report_corex_package(
        "Triton", triton.__version__, triton.__file__, runtime_evidence
    )

    a = torch.randn((512, 512), device=DEVICE)
    b = torch.randn((512, 512), device=DEVICE)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    c = a @ b
    torch.cuda.synchronize()
    print(f"PyTorch matmul first call: {(time.perf_counter() - t0) * 1e3:.3f} ms")
    assert c.shape == (512, 512)

    x = torch.zeros(2049, device=DEVICE)
    y = torch.empty_like(x)
    grid = (triton.cdiv(x.numel(), 256),)
    add_one_kernel[grid](x, y, x.numel(), BLOCK_SIZE=256)
    torch.cuda.synchronize()
    torch.testing.assert_close(y, torch.ones_like(y))
    print("Minimal Triton kernel: PASS")


if __name__ == "__main__":
    main()
