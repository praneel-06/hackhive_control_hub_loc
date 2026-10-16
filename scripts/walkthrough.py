"""Phase-by-phase walkthrough of one plan, for the presentation.

Runs the solver one phase at a time (same calls solve_plan makes), draws the
output of every phase and writes the numbers behind each picture.

Usage: .venv/bin/python scripts/walkthrough.py 10324   → out/walkthrough/*.png + facts.json
"""
from __future__ import annotations

import json
import pickle
import sys
import warnings
from itertools import combinations
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import shapely
from matplotlib.collections import LineCollection
from shapely.geometry import Point

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
from hubring import grid as G                                         # noqa: E402
from hubring.groups import enumerate_groups, hub_lower_bound, min_cover_groups, room_adjacency  # noqa: E402
from hubring.loops import best_loops, visiting_orders, zone_allowed     # noqa: E402
from hubring.metrics import measure                                    # noqa: E402
from hubring.plan import MIN_HUB_NODE_M, NODE_CLEARANCE_M, RESPLAN_DIR, ROOT, build_plan  # noqa: E402
from hubring.routing import route_table                                # noqa: E402
from hubring.solve import (Checker, choose_groups, hub_region, place_hub,  # noqa: E402
                           solve_plan, to_json)
from resplan_utils import get_geometries, normalize_keys               # noqa: E402

OUT = ROOT / "out" / "walkthrough"
TYPE_COL = {"living": "#f2d7a6", "bedroom": "#bfe3c6", "kitchen": "#bcd3f2", "storage": "#f7c6b5",
            "bathroom": "#e3c9e8", "balcony": "#d9e8d0", "stair": "#d6d0e8"}
ZONE_FILL = ["#dbe7fb", "#fde3cc", "#d9f0dc", "#f3dcef"]
ZONE_LINE = ["#2a62c9", "#e0751f", "#2e8b3e", "#a8459a"]
INK = "#2b2b2b"


def fill(ax, geom, **kw):
    for p in getattr(geom, "geoms", [geom]):
        if hasattr(p, "exterior") and not p.is_empty:
            ax.fill(*p.exterior.xy, **kw)
            for h in p.interiors:
                ax.fill(*h.xy, color="white", zorder=kw.get("zorder", 0))


def base(ax, plan, rooms_col=None, excluded=True, labels=True, alpha=1.0):
    for i, r in enumerate(plan.rooms):
        fill(ax, r.poly, color=(rooms_col or {}).get(i, TYPE_COL[r.type]), zorder=0, alpha=alpha)
    if excluded:
        for _, g in plan.excluded:
            fill(ax, g, color="#ececec", zorder=0)
    fill(ax, plan.walls, color="#3a3a3a", zorder=2)
    fill(ax, plan.windows, color="#7fc8f8", zorder=2)
    for d in plan.doors:
        fill(ax, d, color="#c99ac9", zorder=2)
    if labels:
        for r in plan.rooms:
            c = r.poly.representative_point()
            ax.annotate(f"{nice(r.id)}\n{r.area:.1f} m²", (c.x, c.y), xytext=(0, -14),
                        textcoords="offset points", ha="center", va="top", fontsize=7.5, color="#333", zorder=9)


def nice(rid: str) -> str:
    t, k = rid.rsplit("_", 1)
    return f"{t} {int(k) + 1}" if t in ("bedroom",) else t


def finish(fig, ax, name, title):
    ax.set_aspect("equal"); ax.set_axis_off()
    ax.set_title(title, fontsize=11, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / f"{name}.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(pid: int):
    OUT.mkdir(parents=True, exist_ok=True)
    raws = pickle.load(open(RESPLAN_DIR / "ResPlan.pkl", "rb"))
    raw = next(r for r in raws if r["id"] == pid)
    facts: dict = {"plan_id": pid}

    # ---- Phase 0: raw input (drawing units) ---------------------------------------------
    rk = normalize_keys(dict(raw))
    fig, ax = plt.subplots(figsize=(7, 6.4))
    counts = {}
    for key in ("living", "bedroom", "kitchen", "storage", "bathroom", "balcony", "stair"):
        polys = [p for g in get_geometries(rk.get(key)) for p in getattr(g, "geoms", [g])] if rk.get(key) is not None else []
        if polys:
            counts[key] = len(polys)
        for p in polys:
            fill(ax, p, color=TYPE_COL[key], zorder=0)
    for key, col in (("wall", "#3a3a3a"), ("window", "#7fc8f8"), ("door", "#c99ac9"), ("front_door", "#e8556d")):
        if rk.get(key) is not None:
            for g in get_geometries(rk[key]):
                fill(ax, g, color=col, zorder=2)
    for g in getattr(rk["inner"], "geoms", [rk["inner"]]):
        ax.plot(*g.exterior.xy, color="#e8556d", lw=1, ls="--", zorder=3)
    handles = [plt.Rectangle((0, 0), 1, 1, color=TYPE_COL[k]) for k in counts] + \
              [plt.Rectangle((0, 0), 1, 1, color=c) for c in ("#3a3a3a", "#c99ac9", "#7fc8f8")]
    ax.legend(handles, [f"{k} ×{v}" for k, v in counts.items()] + ["wall", "door", "window"],
              loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    finish(fig, ax, "p0_input", f"Phase 0 · Input: plan {pid} as given (drawing units, no devices)")
    facts["p0"] = {"counts": counts, "area_m2_field": rk["area"], "inner_area_units": rk["inner"].area,
                   "wall_depth_units": rk.get("wall_depth")}

    # ---- Phase 1: metres, eligible vs excluded, free space ------------------------------
    plan = build_plan(raw)
    facts["p1"] = {"scale_units_per_m": plan.scale,
                   "eligible": [(r.id, round(r.area, 2)) for r in plan.rooms],
                   "excluded": [n for n, _ in plan.excluded],
                   "eligible_area": round(sum(r.area for r in plan.rooms), 2),
                   "warnings": plan.warnings}
    fig, ax = plt.subplots(figsize=(7, 6.4))
    fill(ax, plan.free, color="#fff1b8", zorder=1, alpha=0.9)
    base(ax, plan, alpha=0.55)
    for _, g in plan.excluded:
        fill(ax, g, color="#d4d4d4", zorder=1, hatch="///", ec="#a0a0a0", lw=0)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color="#fff1b8"), plt.Rectangle((0, 0), 1, 1, fc="#d4d4d4", hatch="///")],
              ["free space (cable allowed)", "excluded room (no device, no cable)"],
              loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    finish(fig, ax, "p1_clean", f"Phase 1 · Clean-up: metres (scale {plan.scale:.2f} units/m), "
           f"5 eligible rooms = {facts['p1']['eligible_area']:.1f} m²")

    # ---- Phase 2: devices ---------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 6.4))
    base(ax, plan)
    for r in plan.rooms:
        ok = shapely.make_valid(r.poly.intersection(plan.free)).difference(plan.walls_win.buffer(NODE_CLEARANCE_M))
        fill(ax, ok, color="#ffffff", zorder=1, alpha=0.55)
        ax.add_patch(plt.Circle(r.node, NODE_CLEARANCE_M, fill=False, ls="--", lw=1, color="#555", zorder=4))
        ax.scatter(*r.node, s=90, c="#d62728", edgecolors="white", linewidths=1.3, zorder=6)
    ax.legend([plt.Line2D([], [], marker="o", ls="", mfc="#d62728", mec="white", ms=9),
               plt.Rectangle((0, 0), 1, 1, fc="white", ec="#999"),
               plt.Line2D([], [], ls="--", color="#555")],
              ["device (1 per room)", "allowed device area (≥ 0.5 m from walls)", "0.5 m circle"],
              loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    finish(fig, ax, "p2_devices", "Phase 2 · Devices: the point in each room farthest from its walls")
    facts["p2"] = [(r.id, [round(v, 2) for v in r.node], round(float(plan.walls_win.distance(Point(r.node))), 2))
                   for r in plan.rooms]

    # ---- Phase 3: grid ------------------------------------------------------------------
    grid = G.build_grid(plan)
    kinds = {"node": [], "hub": [], "door": [], "wall": []}
    for r in plan.rooms:
        kinds["node"] += [("x", r.node[0]), ("y", r.node[1])]
        kinds["hub"] += [("x", r.node[0] + s * MIN_HUB_NODE_M) for s in (-1, 1)] + \
                        [("y", r.node[1] + s * MIN_HUB_NODE_M) for s in (-1, 1)]
        wx, wy = G._room_wall_lines(r.poly)
        kinds["wall"] += [("x", v) for v in wx] + [("y", v) for v in wy]
    for d in plan.doors:
        kinds["door"] += [("x", d.centroid.x), ("y", d.centroid.y)]
    fig, ax = plt.subplots(figsize=(7.6, 6.8))
    base(ax, plan, labels=False, alpha=0.45)
    x0, y0, x1, y1 = plan.inner.bounds
    kcol = {"node": "#d62728", "hub": "#ff9896", "door": "#9467bd", "wall": "#2ca02c"}
    def kind_of(axis, v):
        for k in ("node", "hub", "door", "wall"):
            if any(a == axis and abs(v - w) < G.MERGE_M for a, w in kinds[k]):
                return k
        return "fill"
    for v in grid.xs:
        k = kind_of("x", v); ax.plot([v, v], [y0, y1], lw=0.5, color=kcol.get(k, "#bbbbbb"), zorder=3, alpha=0.8)
    for v in grid.ys:
        k = kind_of("y", v); ax.plot([x0, x1], [v, v], lw=0.5, color=kcol.get(k, "#bbbbbb"), zorder=3, alpha=0.8)
    segs = grid.pts[grid.edges]
    ax.add_collection(LineCollection(segs, colors="#333", lw=0.6, zorder=4))
    if len(grid.rejected):
        ax.add_collection(LineCollection(grid.rejected, colors="#ff3b30", lw=0.6, zorder=4, alpha=0.6))
    ax.scatter(grid.pts[:, 0], grid.pts[:, 1], s=2, c="#333", zorder=5)
    ax.scatter(*grid.pts[grid.node_pt].T, s=40, c="#d62728", zorder=6, edgecolors="white")
    ax.legend([plt.Line2D([], [], color=c) for c in kcol.values()] +
              [plt.Line2D([], [], color="#bbbbbb"), plt.Line2D([], [], color="#333"), plt.Line2D([], [], color="#ff3b30")],
              ["device line", "hub line (device ± 0.1 m)", "door line (+ side lanes)", "just inside a wall",
               "fill line (gap ≤ 1 m)", "legal edge", "rejected edge (hits a wall)"],
              loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    finish(fig, ax, "p3_grid", f"Phase 3 · Grid: {len(grid.xs)}×{len(grid.ys)} lines → "
           f"{len(grid.pts)} points, {len(grid.edges)} legal edges")
    facts["p3"] = {"x_lines": len(grid.xs), "y_lines": len(grid.ys), "points": len(grid.pts),
                   "edges": len(grid.edges), "rejected": len(grid.rejected)}

    # zoom on a door to show lanes
    d = max(plan.doors, key=lambda d: sum(r.poly.distance(d) < 0.3 for r in plan.rooms))
    cx, cy = d.centroid.x, d.centroid.y
    fig, ax = plt.subplots(figsize=(5, 5))
    base(ax, plan, labels=False, alpha=0.45)
    ax.add_collection(LineCollection(segs, colors="#333", lw=1, zorder=4))
    ax.scatter(grid.pts[:, 0], grid.pts[:, 1], s=10, c="#333", zorder=5)
    ax.set_xlim(cx - 1.2, cx + 1.2); ax.set_ylim(cy - 1.2, cy + 1.2)
    finish(fig, ax, "p3_door_zoom", "Zoom: door with centre line + side lanes")

    # ---- Phase 4: room graph ------------------------------------------------------------
    adj = room_adjacency(plan, grid)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    base(ax, plan, labels=False, alpha=0.6)
    pos = {i: r.node for i, r in enumerate(plan.rooms)}
    for u, v, dat in adj.edges(data=True):
        ax.plot(*zip(pos[u], pos[v]), color="#1f4e9c", lw=3, zorder=7, alpha=0.8,
                ls="-" if dat["via"] == "door" else "--")
    for i, r in enumerate(plan.rooms):
        ax.scatter(*pos[i], s=900, c="white", edgecolors="#1f4e9c", linewidths=2, zorder=8)
        ax.annotate(f"{i}\n{r.area:.0f}", pos[i], ha="center", va="center", fontsize=8, zorder=9, color="#1f4e9c", weight="bold")
    ax.legend([plt.Line2D([], [], color="#1f4e9c", lw=3), plt.Line2D([], [], color="#1f4e9c", lw=3, ls="--")],
              ["through a door", "open passage"], loc="upper left", bbox_to_anchor=(1, 1), fontsize=8, frameon=False)
    finish(fig, ax, "p4_graph", "Phase 4 · Room graph: who can reach whom (node = room #, area m²)")
    facts["p4"] = {"rooms": {i: (r.id, round(r.area, 1)) for i, r in enumerate(plan.rooms)},
                   "edges": [(u, v, d["via"]) for u, v, d in adj.edges(data=True)]}

    # ---- Phase 5: candidate zones -------------------------------------------------------
    groups = enumerate_groups(plan, adj)
    n = len(plan.rooms)
    h_min, useful = min_cover_groups(groups, n)
    useful_masks = {g.mask for g in useful}
    full = (1 << n) - 1
    partitions = [(a, b) for a, b in combinations(groups, 2) if a.mask & b.mask == 0 and a.mask | b.mask == full]
    facts["p5"] = {"n_groups": len(groups), "lower_bound": hub_lower_bound(plan), "h_min": h_min,
                   "useful": [(g.room_ids, round(g.area, 1)) for g in useful],
                   "all": [(g.room_ids, round(g.area, 1), g.mask in useful_masks) for g in groups],
                   "partitions": [((a.room_ids, round(a.area, 1)), (b.room_ids, round(b.area, 1))) for a, b in partitions]}
    cols = 6
    rows = int(np.ceil(len(groups) / cols))
    fig, axs = plt.subplots(rows, cols, figsize=(2.1 * cols, 2.1 * rows))
    for ax, g in zip(axs.ravel(), groups):
        c = {i: ("#2a62c9" if g.mask in useful_masks else "#9aa5b1") for i in g.rooms}
        for i, r in enumerate(plan.rooms):
            fill(ax, r.poly, color=c.get(i, "#f1f1f1"), zorder=0)
        fill(ax, plan.walls, color="#777", zorder=1)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{'+'.join(nice(x).replace('bedroom ', 'bed') for x in g.room_ids)}\n{g.area:.0f} m²",
                     fontsize=7, color="#2a62c9" if g.mask in useful_masks else "#777")
    for ax in axs.ravel()[len(groups):]:
        ax.set_axis_off()
    fig.suptitle(f"Phase 5 · All {len(groups)} legal zones (connected, ≤ 100 m², ≤ 5 devices). "
                 f"Blue = can be part of a {h_min}-hub answer ({len(useful)})", fontsize=10, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(OUT / "p5_zones.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    # ---- Phase 6: one leg on the heading-state graph -------------------------------------
    big = max(useful, key=lambda g: (g.n_devices, g.area))
    rt = route_table(grid, allowed=zone_allowed(grid, big.rooms), nodes=np.array(big.rooms))
    a, b = 0, 1
    cost, s, e = rt.best(a, b)
    leg = rt.polyline(a, s, b, e)
    fig, ax = plt.subplots(figsize=(7, 6.4))
    base(ax, plan, labels=True, alpha=0.5)
    allowed = zone_allowed(grid, big.rooms)
    ax.scatter(*grid.pts[allowed].T, s=3, c="#666", zorder=4)
    ax.plot(leg[:, 0], leg[:, 1], color="#d62728", lw=3, zorder=6)
    for k, p in enumerate(leg[1:-1], 1):
        ax.scatter(*p, s=60, marker="o", facecolors="white", edgecolors="#d62728", zorder=7, lw=2)
    for idx, lab in ((a, "A"), (b, "B")):
        ax.scatter(*grid.pts[grid.node_pt[big.rooms[idx]]], s=160, c="#d62728", edgecolors="white", zorder=8)
        ax.annotate(lab, grid.pts[grid.node_pt[big.rooms[idx]]], ha="center", va="center", color="white", weight="bold", zorder=9)
    L = float(np.abs(np.diff(leg, axis=0)).sum())
    finish(fig, ax, "p6_leg", f"Phase 6 · One leg A→B: {L:.2f} m + {len(leg) - 2} bends × 0.5 = {cost:.2f}")
    facts["p6"] = {"zone": big.room_ids, "A": big.room_ids[a], "B": big.room_ids[b], "length": L,
                   "bends": len(leg) - 2, "cost": cost, "grid_points_allowed": int(allowed.sum()),
                   "states": int(allowed.sum()) * 4}

    # ---- Phase 7: loop per zone (orders, independent vs collision-aware) -----------------
    loops = best_loops(big, grid)
    facts["p7"] = {"zone": big.room_ids, "orders": len(list(visiting_orders(big.n_devices))),
                   "loops": [(lp.method, [plan.rooms[i].id for i in lp.order], round(lp.dp_cost, 2),
                              round(lp.J, 2), lp.metrics.L, lp.metrics.B, lp.metrics.X, lp.metrics.O) for lp in loops]}
    ind = next(lp for lp in loops if lp.method == "independent")
    col = next((lp for lp in loops if lp.method == "collision-aware"), ind)
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, lp, name in ((axs[0], ind, "independent shortest legs"), (axs[1], col, "collision-aware re-route")):
        base(ax, plan, labels=False, alpha=0.5)
        ax.plot(lp.ring[:, 0], lp.ring[:, 1], color="#2a62c9", lw=2.4, zorder=6)
        for k, room in enumerate(lp.order, 1):
            x, y = grid.pts[grid.node_pt[room]]
            ax.scatter(x, y, s=170, c="#2a62c9", edgecolors="white", zorder=7)
            ax.annotate(f"N{k}", (x, y), ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=8)
        m = lp.metrics
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{name}\nJ = {m.J:.2f}  (L {m.L:.2f}, bends {m.B}, crossings {m.X}, overlap {m.O:.2f} m)",
                     fontsize=10, loc="left")
    fig.suptitle(f"Phase 7 · Loop through zone {' + '.join(nice(x) for x in big.room_ids)} (no hub yet)",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "p7_loop.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)

    # ---- Phase 8: hub on the loop -------------------------------------------------------
    rg = place_hub(plan, grid, big)
    region = hub_region(plan, big, grid.pts[grid.node_pt])
    fig, ax = plt.subplots(figsize=(7, 6.4))
    base(ax, plan, labels=False, alpha=0.45)
    fill(ax, region, color="#a6e3a1", zorder=1, alpha=0.8)
    ax.plot(rg.ring[:, 0], rg.ring[:, 1], color="#2a62c9", lw=2.4, zorder=6)
    for room in big.rooms:
        ax.scatter(*grid.pts[grid.node_pt[room]], s=110, c="#2a62c9", edgecolors="white", zorder=7)
    ax.scatter(*rg.hub, s=260, marker="s", c="#ffbf00", edgecolors="black", zorder=8)
    ax.annotate("HUB", rg.hub, xytext=(0, 16), textcoords="offset points", ha="center", fontsize=9, weight="bold", zorder=9)
    ax.legend([plt.Rectangle((0, 0), 1, 1, color="#a6e3a1")],
              ["allowed hub area: zone − 0.5 m from walls − 0.1 m around devices"],
              loc="upper left", bbox_to_anchor=(0, -0.02), fontsize=8, frameon=False)
    finish(fig, ax, "p8_hub", f"Phase 8 · Hub in the middle of the longest straight piece inside the green area "
           f"(J stays {rg.J:.2f})")
    facts["p8"] = {"hub": [round(v, 2) for v in rg.hub], "method": rg.hub_method, "J": rg.J}

    # ---- Phase 9: choose zones (DP) + final ---------------------------------------------
    sol = solve_plan(raw)
    rings = {}
    for g in useful:
        best_loops(g, grid)
        r_ = place_hub(plan, grid, g)
        if r_ is not None:
            rings[g.mask] = r_
    facts["p9"] = {"partitions": sorted([((a.room_ids, round(rings[a.mask].J, 2)), (b.room_ids, round(rings[b.mask].J, 2)),
                                          round(rings[a.mask].J + rings[b.mask].J, 2))
                                         for a, b in partitions if a.mask in rings and b.mask in rings], key=lambda t: t[2]),
                   "H": sol.H, "J": sol.total.J, "L": sol.total.L, "B": sol.total.B, "X": sol.total.X, "O": sol.total.O,
                   "zones": [(rg_.group.room_ids, [plan.rooms[i].id for i in rg_.order], round(rg_.J, 2),
                              [round(v, 2) for v in rg_.hub], rg_.hub_method) for rg_ in sol.rings],
                   "checks": sol.checks, "seconds": sol.seconds}
    fig, ax = plt.subplots(figsize=(7.5, 7))
    zone_of = {r: k for k, rg_ in enumerate(sol.rings) for r in rg_.order}
    base(ax, plan, rooms_col={i: ZONE_FILL[k] for i, k in zone_of.items()})
    for k, rg_ in enumerate(sol.rings):
        c = ZONE_LINE[k]
        ax.plot(rg_.ring[:, 0], rg_.ring[:, 1], color=c, lw=2.4, zorder=6)
        for j, room in enumerate(rg_.order, 1):
            x, y = grid.pts[grid.node_pt[room]]
            ax.scatter(x, y, s=170, c=c, edgecolors="white", zorder=7)
            ax.annotate(f"N{j}", (x, y), ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=8)
        ax.scatter(*rg_.hub, s=260, marker="s", c=c, edgecolors="black", zorder=8)
        ax.annotate(f"H{k + 1}", rg_.hub, ha="center", va="center", color="white", fontsize=8, weight="bold", zorder=9)
    finish(fig, ax, "p9_final", f"Phase 9 · Final answer: {sol.H} hubs, J = {sol.total.J:.2f} "
           f"(L {sol.total.L:.2f} m, bends {sol.total.B}, crossings {sol.total.X}, overlap {sol.total.O:.2f} m)")
    facts["orders_example"] = orders_example(raws)
    out_json = to_json(sol)
    json.dump(out_json, open(OUT / "solution.json", "w"), indent=1, default=float)
    json.dump(facts, open(OUT / "facts.json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(json.dumps(facts, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))[:6000])


def orders_example(raws, pid=13929):
    """A 5-device zone: score every visiting order (heading DP), draw best and worst."""
    from hubring.loops import heading_dp, _legs_for
    raw = next(r for r in raws if r["id"] == pid)
    plan = build_plan(raw)
    grid = G.build_grid(plan)
    adj = room_adjacency(plan, grid)
    g = max(enumerate_groups(plan, adj), key=lambda g: g.n_devices)
    rt = _legs_for(g, grid, True, None)
    k = g.n_devices
    scored = []
    for order in visiting_orders(k):
        legs4 = [rt.leg[order[i], :, order[(i + 1) % k], :] for i in range(k)]
        cost, heads = heading_dp(legs4)
        scored.append((cost, order, heads))
    scored.sort(key=lambda t: t[0])
    fig, axs = plt.subplots(1, 2, figsize=(12, 5.8))
    for ax, (cost, order, heads), name in ((axs[0], scored[0], "best order"), (axs[1], scored[-1], "worst order")):
        legs = [rt.polyline(order[i], s, order[(i + 1) % k], e) for i, (s, e) in enumerate(heads)]
        ring = np.vstack([legs[0]] + [lg[1:] for lg in legs[1:]])
        base(ax, plan, labels=False, alpha=0.5)
        ax.plot(ring[:, 0], ring[:, 1], color="#2a62c9", lw=2.2, zorder=6)
        for j, li in enumerate(order, 1):
            x, y = grid.pts[grid.node_pt[g.rooms[li]]]
            ax.scatter(x, y, s=170, c="#2a62c9", edgecolors="white", zorder=7)
            ax.annotate(f"N{j}", (x, y), ha="center", va="center", color="white", fontsize=7, weight="bold", zorder=8)
        ax.set_aspect("equal"); ax.set_axis_off()
        ax.set_title(f"{name}: L + 0.5·bends = {cost:.2f}", fontsize=10, loc="left")
    fig.suptitle(f"Visiting orders for a 5-device zone (plan {pid}): {len(scored)} orders scored, cheapest wins",
                 fontsize=11, x=0.01, ha="left")
    fig.tight_layout(); fig.savefig(OUT / "p7_orders.png", dpi=150, bbox_inches="tight", facecolor="white"); plt.close(fig)
    return {"plan": pid, "zone": g.room_ids, "n_orders": len(scored), "costs": [round(c, 2) for c, _, _ in scored]}


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 10324)
