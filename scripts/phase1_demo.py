"""Phase 1 demo: bend-aware shortest routes between every pair of device nodes.

Prints the min-cost matrix (L + 0.5·B, best over leave/arrive headings) and
draws every pairwise route on the plan.

Usage:
  .venv/bin/python scripts/phase1_demo.py 0 3          # plans by position in the 300 sample
  .venv/bin/python scripts/phase1_demo.py --random 1   # a random plan from the 300 sample
  .venv/bin/python scripts/phase1_demo.py --verify     # check every route on all 300 plans
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.grid import build_grid                                     # noqa: E402
from hubring.plan import ROOT, build_plan, load_plans                   # noqa: E402
from hubring.routing import BEND_COST, HEADINGS, polyline_cost, route_table  # noqa: E402
from scripts.phase0_demo import ROOM_COLORS, _fill                      # noqa: E402

OUT = ROOT / "out"


def plot_routes(plan, grid, rt, path):
    fig, ax = plt.subplots(figsize=(11, 11))
    for r in plan.rooms:
        _fill(ax, r.poly, color=ROOM_COLORS[r.type], zorder=0)
    for _, g in plan.excluded:
        _fill(ax, g, color="#f2f2f2", zorder=0)
    _fill(ax, plan.walls, color="#333", zorder=1)
    _fill(ax, plan.windows, color="#7fc8f8", zorder=1)
    for d in plan.doors:
        _fill(ax, d, color="#e78ac3", zorder=1)
    ax.add_collection(matplotlib.collections.LineCollection(
        np.stack([grid.pts[grid.edges[:, 0]], grid.pts[grid.edges[:, 1]]], axis=1),
        colors="#9db4e8", linewidths=0.3, zorder=2))

    n = len(plan.rooms)
    colors = plt.cm.tab10.colors
    k = 0
    for a in range(n):
        for b in range(a + 1, n):
            cost, s, e = rt.best(a, b)
            if not np.isfinite(cost):
                continue
            pl = rt.polyline(a, s, b, e)
            L, B = polyline_cost(pl)
            off = (k % 5 - 2) * 0.03          # tiny offset so overlapping routes stay visible
            ax.plot(pl[:, 0] + off, pl[:, 1] + off, color=colors[k % 10], lw=2, zorder=3,
                    label=f"{plan.rooms[a].id} → {plan.rooms[b].id}: L={L:.2f} B={B} cost={cost:.2f}")
            k += 1
    nodes = grid.pts[rt.node_pt]
    ax.scatter(*nodes.T, s=140, c="#f28e2b", edgecolors="black", zorder=5)
    for r, (x, y) in zip(plan.rooms, nodes):
        ax.annotate(r.id, (x, y), xytext=(6, 6), textcoords="offset points", fontsize=9, weight="bold")
    ax.set_aspect("equal")
    ax.set_title(f"Plan {plan.id}: shortest L + 0.5·B route between every pair of devices")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def show(raws, indices):
    OUT.mkdir(exist_ok=True)
    for i in indices:
        plan = build_plan(raws[i])
        grid = build_grid(plan)
        t = time.perf_counter()
        rt = route_table(grid)
        dt = time.perf_counter() - t
        ids = [r.id for r in plan.rooms]
        print(f"[{i}] plan {plan.id}: {len(ids)} devices, {4 * len(grid.pts)} search states, "
              f"Dijkstra from {4 * len(ids)} sources in {dt * 1000:.1f} ms")
        w = max(len(x) for x in ids)
        print("  min cost L + 0.5·B (m), best headings:")
        print("  " + " " * w + "".join(f"{x:>12}" for x in ids))
        for a, ra in enumerate(ids):
            cells = []
            for b in range(len(ids)):
                cost, s, e = rt.best(a, b)
                cells.append("—" if a == b else f"{cost:.2f} {HEADINGS[s]}{HEADINGS[e]}")
            print("  " + f"{ra:<{w}}" + "".join(f"{c:>12}" for c in cells))
        a, b = 0, 1
        cost, s, e = rt.best(a, b)
        pl = rt.polyline(a, s, b, e)
        print(f"  route {ids[a]} → {ids[b]} (leave {HEADINGS[s]}, arrive {HEADINGS[e]}): "
              + " → ".join(f"({x:.2f},{y:.2f})" for x, y in pl))
        plot_routes(plan, grid, rt, OUT / f"plan_{plan.id}_routes.png")


def verify(raws):
    """For every plan and every ordered device pair, rebuild the best route and
    check its cost recomputed from the polyline equals Dijkstra's cost."""
    worst, checked, times, unreachable = 0.0, 0, [], 0
    for raw in raws:
        plan = build_plan(raw)
        grid = build_grid(plan)
        t = time.perf_counter()
        rt = route_table(grid)
        times.append(time.perf_counter() - t)
        n = len(plan.rooms)
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                cost, s, e = rt.best(a, b)
                if not np.isfinite(cost):
                    unreachable += 1
                    continue
                L, B = polyline_cost(rt.polyline(a, s, b, e))
                worst = max(worst, abs(cost - (L + BEND_COST * B)))
                # Symmetry: reversed route costs the same.
                worst = max(worst, abs(cost - rt.best(b, a)[0]))
                checked += 1
    print(f"{len(raws)} plans, {checked} routes checked, {unreachable} unreachable pairs")
    print(f"max |Dijkstra cost − polyline L + 0.5·B| = {worst:.2e}")
    print(f"route_table ms: median {np.median(times) * 1e3:.1f}, max {max(times) * 1e3:.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indices", nargs="*", type=int)
    ap.add_argument("--random", type=int, metavar="N")
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args()
    raws = load_plans()
    if args.verify:
        verify(raws)
    else:
        idx = args.indices + (random.sample(range(len(raws)), args.random) if args.random else [])
        show(raws, idx or [0])
