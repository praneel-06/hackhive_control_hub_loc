"""Hub placement, Phase 4 (pick the grouping), validation and JSON output.

Hub rule: like device nodes, a hub sits at least 0.5 m from every wall and
window, inside one of its own group's rooms.

Hub placement on a finished loop: the hub goes in the middle of the longest
straight stretch of the loop that keeps 0.5 m clearance. The cable passes
straight through it, so the hub adds no length and no bend and the loop's J
is unchanged. If no stretch of the loop is clear enough (or the group has one
device), candidate hub grid points near the loop are tried and the ring
hub → N1 → … → Nk → hub is re-routed through each; the lowest J wins.

Phase 4: a DP over room bitmasks picks groups covering every room exactly
once with (fewest hubs, then lowest total J). Loops are restricted to their
own group's rooms, so rings of different groups never meet and J adds up.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point
from shapely.ops import unary_union

from .grid import Grid, build_grid
from .groups import Group, enumerate_groups, hub_lower_bound, room_adjacency
from .loops import RingRouter, solve_all
from .metrics import TOL, Metrics, measure
from .plan import NODE_CLEARANCE_M, Plan, build_plan
from .routing import states_to_polyline

HUB_CLEARANCE_M = 0.5
N_HUB_CANDIDATES = 6


@dataclass
class Ring:
    group: Group
    order: tuple[int, ...]       # room indices in visiting order
    hub: tuple[float, float]     # metres
    ring: np.ndarray             # closed polyline from hub back to hub, metres
    metrics: Metrics
    hub_method: str              # "on-loop" (free) or "rerouted"

    @property
    def J(self) -> float:
        return self.metrics.J


@dataclass
class Solution:
    plan: Plan
    rings: list[Ring]
    total: Metrics
    checks: dict = field(default_factory=dict)
    seconds: float = 0.0

    @property
    def H(self) -> int:
        return len(self.rings)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(v is True for v in self.checks.values())


def hub_region(plan: Plan, group: Group):
    """Where this group's hub may sit: its rooms, ≥ 0.5 m from walls/windows."""
    zone = unary_union([plan.rooms[i].poly for i in group.rooms])
    blocked = unary_union([plan.walls, plan.windows]).buffer(HUB_CLEARANCE_M)
    return zone.difference(blocked)


def _longest_clear_stretch(ring: np.ndarray, region, avoid: np.ndarray):
    """Midpoint of the longest straight piece of the ring inside region,
    away from corners and nodes. Returns (point, segment index) or None."""
    best = None
    for i, (p, q) in enumerate(zip(ring[:-1], ring[1:])):
        if np.abs(q - p).sum() <= 2 * TOL:
            continue
        piece = LineString([p, q]).intersection(region)
        for part in getattr(piece, "geoms", [piece]):
            if part.is_empty or part.geom_type != "LineString" or part.length <= 4 * TOL:
                continue
            mid = np.asarray(part.interpolate(0.5, normalized=True).coords[0])
            if len(avoid) and np.abs(avoid - mid).max(axis=1).min() <= TOL:
                continue
            if best is None or part.length > best[0]:
                best = (part.length, mid, i)
    return None if best is None else (best[1], best[2])


def _reanchor(ring: np.ndarray, hub: np.ndarray, seg: int) -> np.ndarray:
    """Insert hub inside segment `seg` and rotate so the ring starts and ends at it."""
    pts = np.vstack([ring[: seg + 1], hub, ring[seg + 1:-1]])     # open cycle, hub at seg+1
    k = seg + 1
    return np.vstack([pts[k:], pts[:k], pts[k:k + 1]])


def place_hub(plan: Plan, grid: Grid, group: Group) -> Ring | None:
    region = hub_region(plan, group)
    nodes = grid.pts[grid.node_pt[list(group.rooms)]]
    best: Ring | None = None

    # (1) free placement on the loop itself
    for lp in group.loops:
        if lp.degenerate:
            continue
        hit = _longest_clear_stretch(lp.ring, region, nodes)
        if hit is not None:
            hub, seg = hit
            ring = _reanchor(lp.ring, hub, seg)
            m = measure([ring], exempt_points=np.vstack([nodes, hub]))
            cand = Ring(group, lp.order, tuple(hub), ring, m, "on-loop")
            if best is None or cand.J < best.J:
                best = cand
    if best is not None:
        return best

    # (2) re-route through candidate hub grid points (1-device groups, or loops hugging walls)
    shapely.prepare(region)
    ok = np.flatnonzero(shapely.covered_by(shapely.points(grid.pts), region)
                        & np.isin(grid.room_of, group.rooms))
    ok = ok[~np.isin(ok, grid.node_pt[list(group.rooms)])]
    if len(ok) == 0:
        return None
    ref = group.loops[0].ring if group.loops else nodes
    d = np.abs(grid.pts[ok][:, None, :] - ref[None, :, :]).sum(axis=2).min(axis=1)
    allowed = np.isin(grid.room_of, group.rooms) | (grid.room_of < 0)
    orders = [lp.order for lp in group.loops] or [group.rooms]
    for h in ok[np.argsort(d)[:N_HUB_CANDIDATES]]:
        router = RingRouter(grid, allowed, np.r_[grid.node_pt[list(group.rooms)], h])
        for order in orders[:2]:
            paths = router.ring([int(h)] + [int(grid.node_pt[r]) for r in order])
            if paths is None:
                continue
            legs = [states_to_polyline(grid, p) for p in paths]
            ring = np.vstack([legs[0]] + [lg[1:] for lg in legs[1:]])
            m = measure([ring], exempt_points=np.vstack([nodes, grid.pts[h]]))
            cand = Ring(group, tuple(order), tuple(grid.pts[h]), ring, m, "rerouted")
            if best is None or cand.J < best.J:
                best = cand
    return best


def choose_groups(groups: list[Group], rings: dict[int, Ring], n: int) -> list[Ring] | None:
    """Exact cover of all rooms: fewest hubs, then lowest total J, then the
    smallest spread of assigned area between hubs (slide 6 tie-break)."""
    by_low: dict[int, list[Group]] = {}
    for g in groups:
        if g.mask in rings:
            by_low.setdefault(g.lowest, []).append(g)
    memo: dict[int, tuple] = {0: (0, 0.0, ())}

    def solve(rem: int):
        if rem in memo:
            return memo[rem]
        low = (rem & -rem).bit_length() - 1
        out = None
        for g in by_low.get(low, ()):
            if g.mask & rem != g.mask:
                continue
            sub = solve(rem ^ g.mask)
            if sub is None:
                continue
            cand = (sub[0] + 1, sub[1] + rings[g.mask].J, sub[2] + (g.mask,))
            if _better(cand, out, rings):
                out = cand
        memo[rem] = out
        return out

    res = solve((1 << n) - 1)
    return None if res is None else [rings[m] for m in res[2]]


def _better(cand, cur, rings) -> bool:
    if cur is None:
        return True
    if cand[0] != cur[0]:
        return cand[0] < cur[0]
    if abs(cand[1] - cur[1]) > 1e-9:
        return cand[1] < cur[1]
    spread = lambda masks: (np.ptp([rings[m].group.area for m in masks]) if masks else 0.0)
    return spread(cand[2]) < spread(cur[2])


def validate(plan: Plan, grid: Grid, rings: list[Ring], adj) -> dict:
    """Hard rules from slides 2–3, checked from the output geometry."""
    n = len(plan.rooms)
    covered = sorted(r for rg in rings for r in rg.order)
    node_xy = grid.pts[grid.node_pt]
    walls_win = unary_union([plan.walls, plan.windows])
    free = plan.free.buffer(1e-6)
    shapely.prepare(free)
    checks = {
        "every room in exactly one ring": covered == list(range(n)),
        "≤ 100 m² per hub": all(rg.group.area <= 100 + 1e-6 for rg in rings),
        "≤ 5 devices per hub": all(len(rg.order) <= 5 for rg in rings),
        "zones connected through doors/openings": all(
            nx.is_connected(adj.subgraph(rg.group.rooms)) for rg in rings),
    }
    closed = axis = through_walls = visits = hub_clear = node_clear = hub_in_zone = True
    for rg in rings:
        r = rg.ring
        closed &= bool(np.abs(r[0] - np.asarray(rg.hub)).max() <= TOL and np.abs(r[-1] - r[0]).max() <= TOL)
        seg = np.diff(r, axis=0)
        axis &= bool(np.all((np.abs(seg[:, 0]) <= TOL) | (np.abs(seg[:, 1]) <= TOL)))
        through_walls &= bool(shapely.covered_by(LineString(r), free))
        # each own node visited exactly once as a stop; no other node on the cable
        line = LineString(r)
        for i in range(n):
            on = line.distance(Point(node_xy[i])) <= TOL
            stops = int((np.abs(r[:-1] - node_xy[i]).max(axis=1) <= TOL).sum())
            if i in rg.order:
                visits &= stops == 1
            else:
                visits &= not on
        hp = Point(rg.hub)
        hub_clear &= walls_win.distance(hp) >= HUB_CLEARANCE_M - TOL
        hub_in_zone &= bool(unary_union([plan.rooms[i].poly for i in rg.order]).buffer(TOL).covers(hp))
    for i in range(n):
        node_clear &= walls_win.distance(Point(node_xy[i])) >= NODE_CLEARANCE_M - TOL
    checks.update({
        "each ring closed at its own hub": closed,
        "axis-aligned cable": axis,
        "no wall/window crossed, inside boundary": through_walls,
        "each node visited once, no foreign nodes": visits,
        "hub ≥ 0.5 m from walls": hub_clear,
        "hub inside its zone": hub_in_zone,
        "nodes ≥ 0.5 m from walls (input)": node_clear,
    })
    return checks


def solve_plan(raw: dict) -> Solution | None:
    t = time.perf_counter()
    plan = build_plan(raw)
    grid = build_grid(plan)
    adj = room_adjacency(plan, grid)
    groups = enumerate_groups(plan, adj)
    solve_all(groups, grid, restrict=True)
    rings = {}
    for g in groups:
        rg = place_hub(plan, grid, g)
        if rg is not None:
            rings[g.mask] = rg
    chosen = choose_groups(groups, rings, len(plan.rooms))
    if chosen is None:
        return Solution(plan, [], Metrics(0, 0, 0, 0), {"feasible grouping exists": False},
                        time.perf_counter() - t)
    exempt = np.vstack([grid.pts[grid.node_pt], np.array([rg.hub for rg in chosen])])
    total = measure([rg.ring for rg in chosen], exempt_points=exempt)
    sol = Solution(plan, chosen, total, validate(plan, grid, chosen, adj), time.perf_counter() - t)
    sol.grid = grid
    sol.lower_bound = hub_lower_bound(plan)
    return sol


def to_json(sol: Solution) -> dict:
    p, s = sol.plan, sol.plan.scale
    units = lambda pts: (np.asarray(pts) * s).round(4).tolist()
    return {
        "plan_id": p.id,
        "scale_units_per_m": s,
        "hubs": [{
            "hub_id": f"H{k + 1}",
            "position_m": list(np.round(rg.hub, 4)),
            "position_units": units(rg.hub),
            "rooms": [p.rooms[i].id for i in rg.order],
            "node_order": [f"{p.rooms[i].id}" for i in rg.order],
            "area_m2": round(rg.group.area, 3),
            "ring_m": np.round(rg.ring, 4).tolist(),
            "ring_units": units(rg.ring),
            "metrics": {"L": rg.metrics.L, "B": rg.metrics.B, "X": rg.metrics.X,
                        "O": rg.metrics.O, "J": rg.J},
        } for k, rg in enumerate(sol.rings)],
        "H": sol.H,
        "J": sol.total.J,
        "totals": {"L": sol.total.L, "B": sol.total.B, "X": sol.total.X, "O": sol.total.O},
        "checks": sol.checks,
    }


def dump(sol: Solution, path) -> None:
    with open(path, "w") as f:
        json.dump(to_json(sol), f, indent=1, default=float)
