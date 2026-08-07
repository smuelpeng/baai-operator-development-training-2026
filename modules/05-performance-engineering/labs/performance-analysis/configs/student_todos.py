"""Student TODO markers for Module-5 configs (lab1-day2 style).

Edit only between the YOUR CODE HERE markers.

Recommended order on BI-V150:
1. Keep defaults (torch + baseline) and run Offline/Server once.
2. For REPORT.md three-way table, add \"autotune\" (first Triton JIT is slow).
3. Verify π / β against the datasheet for Roofline.
"""

from __future__ import annotations

# Offline / Server SUT list (used by 01_benchmark.py).
# JSON configs/gemm_*.json should stay in sync as a readable mirror.
#
# >>> YOUR CODE HERE >>>
# TODO (required for REPORT three-way comparison):
#   OFFLINE_IMPLEMENTATIONS = ["torch", "baseline", "autotune"]
#   SERVER_IMPLEMENTATIONS = ["torch", "baseline", "autotune"]
# Or CLI: python3 01_benchmark.py --impl torch baseline autotune
OFFLINE_IMPLEMENTATIONS = ["torch", "baseline"]
SERVER_IMPLEMENTATIONS = ["torch", "baseline"]
# <<< END OF YOUR CODE <<<

# BI-V150 Roofline peaks (must match configs/hardware_bi_v150.json).
#
# >>> YOUR CODE HERE >>>
# TODO: look up BI-V150 FP16 peak (TFLOP/s) and HBM bandwidth (GB/s),
# then set the values below AND copy them into hardware_bi_v150.json.
PEAK_TFLOPS_FP16 = 192.0
HBM_BANDWIDTH_GBPS = 1228.8
# <<< END OF YOUR CODE <<<
