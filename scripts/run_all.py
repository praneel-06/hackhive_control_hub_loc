"""Solve plans end to end, validate, and write JSON + overlays.

Usage:
  .venv/bin/python scripts/run_all.py              # all 300 sample plans, summary + out/results.json
  .venv/bin/python scripts/run_all.py 0 216        # specific plans, also draws overlays
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from hubring.plan import ROOT, load_plans            # noqa: E402
from hubring.solve import dump, solve_plan, to_json  # noqa: E402
from scripts.phase0_demo import _fill                # noqa: E402

OUT = ROOT / "out"
ZONE_FILL = ["#dbe7fb", "#fde3cc", "#d9f0dc", "#f3dcef", "#fff2c2", "#dcf3f5"]
ZONE_LINE = ["#2a62c9", "#e0751f", "#2e8b3e", "#a8459a", "#c79a00", "#178f9c"]


def overlay(sol, path, title=None, ax=None):
    plan, grid = sol.plan, sol.grid
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(8, 8))
    zone_of = {r: k for k, rg in enumerate(sol.rings) for r in rg.order}
    for i, r in enumerate(plan.rooms):
        k = zone_of.get(i)
        _fill(ax, r.poly, color=ZONE_FILL[k % 6] if k is not None else "#eee", zorder=0)
        c = r.poly.representative_point()
        ax.annotate(f"{r.id.replace('_0', '').replace('_', ' ')}\n{r.area:.0f} m²", (c.x, c.y),
                    xytext=(0, -16), textcoords="offset points", ha="center", va="top",
                    fontsize=7, color="#444")
    for _, g in plan.excluded:
        _fill(ax, g, color="#e9e9e9", zorder=0)
    _fill(ax, plan.walls, color="#3a3a3a", zorder=1)
    _fill(ax, plan.windows, color="#8ccfff", zorder=1)
    for d in plan.doors:
        _fill(ax, d, color="#c8a2c8", zorder=1)
    for k, rg in enumerate(sol.rings):
        col = ZONE_LINE[k % 6]
        ax.plot(rg.ring[:, 0], rg.ring[:, 1], color=col, lw=2.2, zorder=3, solid_joinstyle="miter")
        for n, room in enumerate(rg.order, 1):
            x, y = grid.pts[grid.node_pt[room]]
            ax.scatter(x, y, s=150, c=col, edgecolors="white", linewidths=1.5, zorder=5)
            ax.annotate(f"N{n}", (x, y), ha="center", va="center", fontsize=6.5, color="white",
                        weight="bold", zorder=6)
        hx, hy = rg.hub
        if len(rg.order) == 1:
            # Hub sits right beside its only device: draw both small so the device
            # stays visible, and call out the tiny out-and-back ring.
            nx_, ny_ = grid.pts[grid.node_pt[rg.order[0]]]
            ax.scatter(hx, hy, s=60, marker="s", c=col, edgecolors="black", linewidths=1, zorder=7)
            gap = abs(hx - nx_) + abs(hy - ny_)
            ax.annotate(f"H{k + 1} + N1, {gap:.2f} m apart\n({rg.metrics.L:.2f} m cable, out and back)",
                        (hx, hy), xytext=(0, 28), textcoords="offset points", ha="center",
                        fontsize=7, color="black", zorder=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=col, lw=1),
                        arrowprops=dict(arrowstyle="-", color=col, lw=1))
            continue
        ax.scatter(hx, hy, s=230, marker="s", c=col, edgecolors="black", linewidths=1.2, zorder=6)
        ax.annotate(f"H{k + 1}", (hx, hy), ha="center", va="center", fontsize=7, color="white",
                    weight="bold", zorder=7)
    ax.set_aspect("equal"); ax.set_axis_off()
    m = sol.total
    ax.set_title(title or f"Plan {plan.id}: H = {sol.H}, J = {m.J:.1f}  "
                 f"(L {m.L:.1f} m, B {m.B}, X {m.X}, O {m.O:.2f} m)", fontsize=10)
    if own:
        fig.tight_layout()
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)


def main(indices):
    raws = load_plans()
    OUT.mkdir(exist_ok=True)
    if indices:
        for i in indices:
            sol = solve_plan(raws[i])
            overlay(sol, OUT / f"plan_{sol.plan.id}_solution.png")
            dump(sol, OUT / f"plan_{sol.plan.id}_solution.json")
            print(f"[{i}] plan {sol.plan.id}: H {sol.H}, J {sol.total.J:.2f}, valid {sol.valid}")
        return
    rows, fails = [], collections.Counter()
    results = []
    for i, raw in enumerate(raws):
        sol = solve_plan(raw)
        if sol.H == 0:
            rows.append(dict(i=i, id=sol.plan.id, feasible=False))
            continue
        for k, v in sol.checks.items():
            if v is not True:
                fails[k] += 1
        rows.append(dict(i=i, id=sol.plan.id, feasible=True, valid=sol.valid, H=sol.H,
                         LB=sol.lower_bound, J=sol.total.J, X=sol.total.X, O=sol.total.O,
                         L=sol.total.L, B=sol.total.B, s=sol.seconds,
                         n=len(sol.plan.rooms), single=sum(len(r.order) == 1 for r in sol.rings),
                         rerouted=sum(r.hub_method == "rerouted" and len(r.order) > 1 for r in sol.rings)))
        results.append(to_json(sol))
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, default=float)
    with open(OUT / "summary_rows.json", "w") as f:
        json.dump(rows, f, default=float)
    feas = [r for r in rows if r["feasible"]]
    val = [r for r in feas if r["valid"]]
    print(f"plans {len(rows)} | feasible {len(feas)} (others have a room > 100 m²) | valid {len(val)}")
    print("check failures:", dict(fails) or "none")
    H = np.array([r["H"] for r in val]); LB = np.array([r["LB"] for r in val])
    print(f"hubs: mean {H.mean():.2f}, at lower bound {np.mean(H == LB) * 100:.0f}% | "
          f"H − LB distribution {dict(collections.Counter((H - LB).tolist()))}")
    print(f"hub count distribution {dict(sorted(collections.Counter(H.tolist()).items()))}")
    J = np.array([r["J"] for r in val])
    print(f"J: median {np.median(J):.1f}, mean {J.mean():.1f} | plans with X>0: "
          f"{sum(r['X'] > 0 for r in val)}, with O>0.5 m: {sum(r['O'] > 0.5 for r in val)}")
    print(f"single-device hubs: {sum(r['single'] for r in val)} across {sum(r['single'] > 0 for r in val)} plans; "
          f"multi-room hubs needing a re-route: {sum(r['rerouted'] for r in val)}")
    S = np.array([r["s"] for r in val])
    print(f"time per plan: median {np.median(S):.2f}s, p95 {np.percentile(S, 95):.2f}s, max {S.max():.2f}s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("indices", nargs="*", type=int)
    main(ap.parse_args().indices)
