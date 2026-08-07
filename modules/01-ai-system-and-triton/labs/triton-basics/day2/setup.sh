#!/usr/bin/env bash
set -euo pipefail

python3 - <<'PY'
import torch
import triton

print("PyTorch:", torch.__version__)
print("Triton:", triton.__version__)
print("Accelerator available:", torch.cuda.is_available())
if not torch.cuda.is_available():
    raise SystemExit("BI-V150/CoreX device is not visible to PyTorch")
torch.cuda.set_device(0)
print("Selected device: cuda:0")
print("Device name:", torch.cuda.get_device_name(0))
print("Do not pip-install/upgrade torch or triton in this environment.")
PY
