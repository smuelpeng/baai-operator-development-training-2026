# FlagOS / BI-V150 闭环

学生仓：https://gitee.com/yihan-long/day2-lab5

## 与 Day1–2

| 仓库 | 作用 |
|------|------|
| [lab1-day2](https://gitee.com/sunxt-0719/lab1-day2) | 推荐 SUT（GEMM） |
| 本仓库 | Benchmark → Roofline → Profiling |

```text
~/labs/
  lab1-day2/
  day2-lab5/
```

```bash
export LAB_DAY2_ROOT=$PWD/../lab1-day2
```

## 闭环

```text
本地编辑 → git push → FlagOS git pull → 跑 00–04
         → 写入 results/ 与 REPORT.md → git push → 本地 pull
```

## 首次上机

```bash
git clone git@gitee.com:yihan-long/day2-lab5.git
git clone git@gitee.com:sunxt-0719/lab1-day2.git
cd day2-lab5
source env_corex.sh
export LAB_DAY2_ROOT=$PWD/../lab1-day2
bash setup.sh
python3 00_smoke_cuda.py
```

若需 Roofline 图：`pip install matplotlib`（无图时仍有 `roofline.json`）。

操作说明以根目录 [`README.md`](../README.md) 为准。
