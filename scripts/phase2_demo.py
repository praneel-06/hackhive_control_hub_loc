"""Phase 2 demo: room adjacency and every valid candidate group.

Usage:
  .venv/bin/python scripts/phase2_demo.py 0 3          # plans by position in the 300 sample
  .venv/bin/python scripts/phase2_demo.py --random 1   # a random plan from the 300 sample
  .venv/bin/python scripts/phase2_demo.py --sweep      # stats over all 300 plans
"""
from __future__ import annotations

import argparse
import collections
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.grid import build_grid                                         # noqa: E402
from hubring.groups import (enumerate_groups, hub_lower_bound, min_hubs,   # noqa: E402
                            room_adjacency)
from hubring.plan import build_plan, load_plans                             # noqa: E402


def show(raws, indices):
    for i in indices:
        plan = build_plan(raws[i])
        grid = build_grid(plan)
        adj = room_adjacency(plan, grid)
        groups = enumerate_groups(plan, adj)
        h, count = min_hubs(groups, len(plan.rooms))
        print(f"[{i}] plan {plan.id}: {len(plan.rooms)} rooms, "
              f"{sum(r.area for r in plan.rooms):.1f} m² eligible")
        for k, r in enumerate(plan.rooms):
            print(f"  room {k} = bit {1 << k:>4b}  {r.id:<10} {r.area:6.1f} m²")
        print("  adjacency:", ", ".join(
            f"{plan.rooms[u].id}–{plan.rooms[v].id} ({d['via']})" for u, v, d in adj.edges(data=True)))
        print(f"  {len(groups)} valid groups:")
        for g in groups:
            print(f"    mask {g.mask:0{len(plan.rooms)}b}  {'+'.join(g.room_ids):<45} "
                  f"{g.area:6.1f} m²  {g.n_devices} dev")
        print(f"  lower bound {hub_lower_bound(plan)} hubs; exact minimum {h} hubs "
              f"({count} distinct partitions reach it)\n")


def sweep(raws):
    times, n_groups, gap_dist, diffs, infeasible = [], [], collections.Counter(), [], []
    for i, raw in enumerate(raws):
        plan = build_plan(raw)
        grid = build_grid(plan)
        t = time.perf_counter()
        adj = room_adjacency(plan, grid)
        groups = enumerate_groups(plan, adj)
        h, _ = min_hubs(groups, len(plan.rooms))
        times.append(time.perf_counter() - t)
        n_groups.append(len(groups))
        lb = hub_lower_bound(plan)
        if h is None:
            big = [r.id for r in plan.rooms if r.area > 100]
            infeasible.append((i, plan.id, f"rooms over 100 m²: {big}" if big else "no exact cover"))
        else:
            gap_dist[h - lb] += 1
        ours = {frozenset((plan.rooms[u].id, plan.rooms[v].id)) for u, v in adj.edges}
        theirs = {frozenset(e) for e in plan.room_graph.edges}
        if ours != theirs:
            diffs.append((i, plan.id, [tuple(e) for e in ours - theirs], [tuple(e) for e in theirs - ours]))
    print(f"{len(raws)} plans | groups per plan: median {np.median(n_groups):.0f}, max {max(n_groups)} | "
          f"adjacency+groups+min-hub ms: median {np.median(times) * 1e3:.2f}, max {max(times) * 1e3:.1f}")
    print(f"exact min hubs − lower bound: {dict(sorted(gap_dist.items()))}")
    print(f"infeasible plans ({len(infeasible)}):", *infeasible[:20], sep="\n  ")
    print(f"grid adjacency differs from ResPlan door graph on {len(diffs)} plans "
          "(only-in-grid, only-in-ResPlan):", *diffs[:10], sep="\n  ")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("indices", nargs="*", type=int)
    ap.add_argument("--random", type=int, metavar="N")
    ap.add_argument("--sweep", action="store_true")
    args = ap.parse_args()
    raws = load_plans()
    if args.sweep:
        sweep(raws)
    else:
        idx = args.indices + (random.sample(range(len(raws)), args.random) if args.random else [])
        show(raws, idx or [0])
