"""Compare two solve_random dumps plan by plan.

Usage: .venv/bin/python scripts/compare_runs.py out/solve_random_1000_A.json out/solve_random_1000_B.json
"""
import json
import sys

import numpy as np

a_path, b_path = sys.argv[1], sys.argv[2]
A = {r["id"]: r for r in json.load(open(a_path))}
B = {r["id"]: r for r in json.load(open(b_path))}
ok = lambda r: r["status"] == "solved" and not r.get("failed") and "H" in r
va, vb = [i for i in A if ok(A[i])], [i for i in B if ok(B[i])]
com = [i for i in A if ok(A[i]) and i in B and ok(B[i])]
dH = np.array([B[i]["H"] - A[i]["H"] for i in com])
dJ = np.array([B[i]["J"] - A[i]["J"] for i in com])
ta, tb = np.array([r["s"] for r in A.values() if "s" in r]), np.array([r["s"] for r in B.values() if "s" in r])
row = lambda name, x, y: print(f"{name:<34}{x:>14}{y:>14}")
row("", "A", "B")
row("valid", len(va), len(vb))
row("crashes", sum(r["status"] == "crash" for r in A.values()), sum(r["status"] == "crash" for r in B.values()))
row("mean hubs (common)", f"{np.mean([A[i]['H'] for i in com]):.3f}", f"{np.mean([B[i]['H'] for i in com]):.3f}")
row("median J (common)", f"{np.median([A[i]['J'] for i in com]):.2f}", f"{np.median([B[i]['J'] for i in com]):.2f}")
row("total J (common)", f"{sum(A[i]['J'] for i in com):.1f}", f"{sum(B[i]['J'] for i in com):.1f}")
row("crossings total (common)", sum(A[i]["X"] for i in com), sum(B[i]["X"] for i in com))
row("plans with X>0 (common)", sum(A[i]["X"] > 0 for i in com), sum(B[i]["X"] > 0 for i in com))
row("time total s", f"{ta.sum():.1f}", f"{tb.sum():.1f}")
row("time median / p95 / max s", f"{np.median(ta):.3f}/{np.percentile(ta, 95):.2f}/{ta.max():.2f}",
    f"{np.median(tb):.3f}/{np.percentile(tb, 95):.2f}/{tb.max():.2f}")
print(f"common {len(com)}: hubs up {(dH > 0).sum()} down {(dH < 0).sum()}; "
      f"J up {(dJ > 1e-6).sum()} down {(dJ < -1e-6).sum()} same {(np.abs(dJ) <= 1e-6).sum()}; "
      f"ΔJ total {dJ.sum():+.1f}, worst {dJ.max():+.2f} (plan {com[int(dJ.argmax())]}), best {dJ.min():+.2f}")
