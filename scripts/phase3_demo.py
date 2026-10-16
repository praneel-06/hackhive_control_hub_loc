"""Phase 3 demo: cheapest closed loop through every candidate group.

Usage:
  .venv/bin/python scripts/phase3_demo.py 0            # plans by position in the 300 sample
  .venv/bin/python scripts/phase3_demo.py --random 1   # a random plan from the 300 sample
  .venv/bin/python scripts/phase3_demo.py --sweep      # checks over all 300 plans
  add --unrestricted to let loops pass through other groups' rooms
"""
from __future__ import annotations

import argparse
import math
import random
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.grid import build_grid                                  # noqa: E402
from hubring.groups import enumerate_groups, room_adjacency          # noqa: E402
from hubring.loops import solve_all                                  # noqa: E402
from hubring.plan import ROOT, build_plan, load_plans                # noqa: E402
from scripts.phase0_demo import ROOM_COLORS, _fill                   # noqa: E402

OUT = ROOT / "out"


def prepare(raw, restrict):
    plan = build_plan(raw)
    grid = build_grid(plan)
    groups = enumerate_groups(plan, room_adjacency(plan, grid))
    t = time.perf_counter()
    solve_all(groups, grid, restrict=restrict)
    return plan, grid, groups, time.perf_counter() - t


def _draw_plan(ax, plan, highlight=()):
    for i, r in enumerate(plan.rooms):
        _fill(ax, r.poly, color=ROOM_COLORS[r.type] if i in highlight else "#f4f4f4", zorder=0)
    for _, g in plan.excluded:
        _fill(ax, g, color="#fafafa", zorder=0)
    _fill(ax, plan.walls, color="#333", zorder=1)
    _fill(ax, plan.windows, color="#7fc8f8", zorder=1)
    for d in plan.doors:
        _fill(ax, d, color="#e78ac3", zorder=1)


def plot_loops(plan, grid, groups, path):
    multi = [g for g in groups if g.n_devices >= 2 and g.loops]
    cols = min(4, len(multi)) or 1
    rows = math.ceil(len(multi) / cols) or 1
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 4.4 * rows), squeeze=False)
    names = {i: r.id for i, r in enumerate(plan.rooms)}
    for ax, g in zip(axes.ravel(), multi):
        lp = g.loops[0]
        _draw_plan(ax, plan, highlight=g.rooms)
        ax.plot(lp.ring[:, 0], lp.ring[:, 1], color="#d62728", lw=2, zorder=3)
        for n, room in enumerate(lp.order, 1):
            x, y = grid.pts[grid.node_pt[room]]
            ax.scatter(x, y, s=90, c="#f28e2b", edgecolors="black", zorder=4)
            ax.annotate(str(n), (x, y), ha="center", va="center", fontsize=7, weight="bold", zorder=5)
        m = lp.metrics
        ax.set_title(f"{' → '.join(names[r] for r in lp.order)} ({lp.method})\n"
                     f"J={lp.J:.2f}  L={m.L:.2f} B={m.B} X={m.X} O={m.O:.2f}", fontsize=8)
        ax.set_aspect("equal"); ax.set_axis_off()
    for ax in axes.ravel()[len(multi):]:
        ax.set_axis_off()
    fig.suptitle(f"Plan {plan.id}: best loop for each multi-room candidate group (no hub yet)")
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def show(raws, indices, restrict):
    OUT.mkdir(exist_ok=True)
    for i in indices:
        plan, grid, groups, dt = prepare(raws[i], restrict)
        names = {k: r.id for k, r in enumerate(plan.rooms)}
        print(f"[{i}] plan {plan.id}: {len(groups)} groups, loops solved in {dt * 1000:.0f} ms "
              f"({'own rooms only' if restrict else 'any room'})")
        print(f"  {'mask':>{len(plan.rooms)}}  {'best loop order':<52} {'L':>6} {'B':>3} {'X':>2} {'O':>5} {'J':>7}")
        for g in groups:
            lp = g.loops[0] if g.loops else None
            if lp is None:
                print(f"  {g.mask:0{len(plan.rooms)}b}  unreachable")
            elif lp.degenerate:
                print(f"  {g.mask:0{len(plan.rooms)}b}  {names[lp.order[0]]:<52} (1 device: loop needs the hub)")
            else:
                m = lp.metrics
                order = " → ".join(names[r] for r in lp.order) + " → " + names[lp.order[0]]
                print(f"  {g.mask:0{len(plan.rooms)}b}  {order:<52} {m.L:6.2f} {m.B:3d} {m.X:2d} {m.O:5.2f} {lp.J:7.2f}")
        plot_loops(plan, grid, groups, OUT / f"plan_{plan.id}_loops.png")


def sweep(raws, restrict):
    times, mismatch, reorder, with_x, with_o, n_loops, failed = [], 0.0, 0, 0, 0, 0, 0
    aware_wins, j_indep, j_best = 0, [], []
    for raw in raws:
        plan, grid, groups, dt = prepare(raw, restrict)
        times.append(dt)
        for g in groups:
            if g.n_devices < 2:
                continue
            if not g.loops:
                failed += 1
                continue
            n_loops += 1
            best = g.loops[0]
            m = best.metrics
            for lp in g.loops:
                if lp.method == "independent":
                    mismatch = max(mismatch, abs(lp.dp_cost - (lp.metrics.L + 0.5 * lp.metrics.B)))
            indep = [lp for lp in g.loops if lp.method == "independent"]
            reorder += best.order != min(indep, key=lambda lp: lp.dp_cost).order
            aware_wins += best.method == "collision-aware"
            j_indep.append(min(lp.J for lp in indep)); j_best.append(best.J)
            with_x += m.X > 0
            with_o += m.O > 1e-9
    print(f"{len(raws)} plans, {n_loops} multi-room groups looped, {failed} unreachable")
    print(f"max |heading-DP cost − polyline L + 0.5·B| (independent legs) = {mismatch:.2e}")
    print(f"collision-aware routing wins in {aware_wins}/{n_loops} groups; "
          f"mean best J {np.mean(j_indep):.2f} → {np.mean(j_best):.2f}")
    print(f"best loop has crossings in {with_x} groups, overlap in {with_o} groups; "
          f"best order differs from the heading DP's first choice in {reorder} groups")
    print(f"loop solve ms per plan: median {np.median(times) * 1e3:.0f}, max {max(times) * 1e3:.0f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indices", nargs="*", type=int)
    ap.add_argument("--random", type=int, metavar="N")
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--unrestricted", action="store_true")
    args = ap.parse_args()
    raws = load_plans()
    if args.sweep:
        sweep(raws, not args.unrestricted)
    else:
        idx = args.indices + (random.sample(range(len(raws)), args.random) if args.random else [])
        show(raws, idx or [0], not args.unrestricted)
