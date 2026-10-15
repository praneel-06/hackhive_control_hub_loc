"""Phase 0 demo: load a plan, plot it with its room graph, build the routing
grid, plot that, and sanity-check that door-connected rooms are reachable.

Usage:
  .venv/bin/python scripts/phase0_demo.py 0 3              # plans by position
  .venv/bin/python scripts/phase0_demo.py --random 3        # 3 random plans (300-plan sample)
  .venv/bin/python scripts/phase0_demo.py --random 3 --full # 3 random plans from all 17,000
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
from matplotlib.collections import LineCollection
from scipy.sparse.csgraph import connected_components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.grid import build_grid                       # noqa: E402
from hubring.plan import RESPLAN_DIR, ROOT, build_plan, load_plans  # noqa: E402
from resplan_utils import plot_plan_and_graph            # noqa: E402

OUT = ROOT / "out"
ROOM_COLORS = {"living": "#e3e3e3", "bedroom": "#bfe3d6", "kitchen": "#cdd6ec", "storage": "#f7d2c4"}


def _fill(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if hasattr(p, "exterior") and not p.is_empty:
            ax.fill(*p.exterior.xy, **kw)


def plot_grid(plan, grid, path):
    fig, ax = plt.subplots(figsize=(11, 11))
    for r in plan.rooms:
        _fill(ax, r.poly, color=ROOM_COLORS[r.type], zorder=0)
        ax.annotate(f"{r.id}\n{r.area:.1f} m²", r.poly.representative_point().coords[0],
                    fontsize=8, ha="center", va="top", color="#555", xytext=(0, -12),
                    textcoords="offset points")
    for _, g in plan.excluded:
        _fill(ax, g, color="#f2f2f2", hatch="//", edgecolor="#ccc", zorder=0)
    _fill(ax, plan.walls, color="#333", zorder=1)
    _fill(ax, plan.windows, color="#7fc8f8", zorder=1)
    for d in plan.doors:
        _fill(ax, d, color="#e78ac3", zorder=1)

    ax.add_collection(LineCollection(grid.rejected, colors="red", linewidths=0.6, alpha=0.5, zorder=2))
    segs = np.stack([grid.pts[grid.edges[:, 0]], grid.pts[grid.edges[:, 1]]], axis=1)
    ax.add_collection(LineCollection(segs, colors="#3a6fd8", linewidths=0.5, alpha=0.7, zorder=3))
    gap = grid.room_of < 0
    ax.scatter(*grid.pts[~gap].T, s=3, c="#3a6fd8", zorder=4)
    ax.scatter(*grid.pts[gap].T, s=10, c="#d62728", zorder=4, label="point in door / opening gap")
    nodes = grid.pts[grid.node_pt]
    ax.scatter(*nodes.T, s=120, c="#f28e2b", edgecolors="black", zorder=5, label="device node")
    ax.set_aspect("equal")
    ax.set_title(f"Plan {plan.id}: {len(grid.pts)} grid points, {len(grid.edges)} legal edges, "
                 f"{len(grid.rejected)} rejected (red)")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def check_reachability(plan, grid):
    """Every pair of rooms joined by a passage edge must be connected in the grid."""
    n = len(grid.pts)
    A = sp.coo_matrix((np.ones(len(grid.edges)), (grid.edges[:, 0], grid.edges[:, 1])), shape=(n, n))
    _, comp = connected_components(A, directed=False)
    idx = {r.id: i for i, r in enumerate(plan.rooms)}
    bad = [(u, v) for u, v in plan.room_graph.edges
           if comp[grid.node_pt[idx[u]]] != comp[grid.node_pt[idx[v]]]]
    return bad


def main(raws, indices):
    OUT.mkdir(exist_ok=True)
    for i in indices:
        t = time.perf_counter()
        plan = build_plan(raws[i])
        grid = build_grid(plan)
        dt = time.perf_counter() - t

        ax = plot_plan_and_graph(plan.raw, title=f"Plan {plan.id} — ResPlan room graph")
        ax.figure.savefig(OUT / f"plan_{plan.id}_graph.png", dpi=110)
        plt.close(ax.figure)
        plot_grid(plan, grid, OUT / f"plan_{plan.id}_grid.png")

        total = sum(r.area for r in plan.rooms)
        bad = check_reachability(plan, grid)
        print(f"[{i}] plan {plan.id}: scale {plan.scale:.2f} u/m | "
              f"{len(plan.rooms)} eligible rooms, {total:.1f} m² | "
              f"door graph {sorted(plan.room_graph.edges)} | "
              f"grid {len(grid.xs)}x{len(grid.ys)} lines, {len(grid.pts)} pts, {len(grid.edges)} edges | "
              f"unreachable door links: {bad or 'none'} | {dt*1000:.0f} ms")


if __name__ == "__main__":
    import argparse
    import random

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indices", nargs="*", type=int, help="plan positions in the dataset")
    ap.add_argument("--random", type=int, metavar="N", help="pick N random plans")
    ap.add_argument("--seed", type=int, help="seed for --random (repeatable picks)")
    ap.add_argument("--full", action="store_true",
                    help="use all 17,000 plans (ResPlan.pkl) instead of the 300-plan sample")
    args = ap.parse_args()

    raws = load_plans(RESPLAN_DIR / "ResPlan.pkl" if args.full else ROOT / "data" / "sample300.pkl")
    indices = args.indices
    if args.random:
        indices += random.Random(args.seed).sample(range(len(raws)), args.random)
    main(raws, indices or [0, 3])
