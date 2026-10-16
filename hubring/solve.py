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

Fallback ladder (solve_plan): only a plan that passes `validate` counts, so
a failure at one level moves to the next instead of giving up:
  0 normal           DP over every group's ring
  1 next-best        DP over rings that pass every per-ring rule, which
                     includes splitting a bad zone down to one room per hub
  2 safer retry      finer grid, all visiting orders, more hub candidates,
                     ring legs never pass through a device point
  3 safer next-best
  4 partial          nothing valid; best attempt returned, flagged, logged
Exceptions inside one group's hub placement drop that group; exceptions in
a whole attempt move to the next level.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field

import networkx as nx
import numpy as np
import shapely
from shapely.geometry import LineString, Point

from .grid import FILL_SPACING_M, Grid, build_grid
from .groups import (AREA_TOL_M2, MAX_AREA_M2, Group, enumerate_groups, hub_lower_bound,
                     min_cover_groups, room_adjacency)
from .loops import KEEP_TOP, RingRouter, solve_all, zone_allowed
from .metrics import TOL, Metrics, measure
from .plan import MIN_HUB_NODE_M, NODE_CLEARANCE_M, Plan, _union, build_plan, safe_op
from .routing import states_to_polyline

HUB_CLEARANCE_M = 0.5
HUB_SLACK_M = 1e-5      # grid coordinates are rounded to 1 µm; far inside the 1 mm tolerance
N_HUB_CANDIDATES = 6
TIME_LIMIT_S = 30.0     # soft budget per plan: no safer retry once a third of it is used


@dataclass(frozen=True)
class Config:
    fill: float = FILL_SPACING_M          # grid fill spacing
    keep: int | None = KEEP_TOP           # visiting orders re-routed per group (None = all)
    hub_candidates: int = N_HUB_CANDIDATES
    block_nodes: bool = False             # ring legs may not pass through any device/hub point


NORMAL = Config()
SAFER = Config(fill=0.5, keep=None, hub_candidates=4 * N_HUB_CANDIDATES, block_nodes=True)
FALLBACK_LEVELS = ("normal", "next-best grouping", "safer retry", "safer next-best", "partial")


@dataclass
class Ring:
    group: Group
    order: tuple[int, ...]       # room indices in visiting order
    hub: tuple[float, float]     # metres
    ring: np.ndarray             # closed polyline from hub back to hub, metres
    metrics: Metrics
    hub_method: str              # "on-loop" (free), "direct" (1 device), "rerouted" or "micro" (flagged)
    hub_clearance: float = HUB_CLEARANCE_M   # clearance the hub was placed with (< 0.5 m ⇒ flagged)

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
    fallback: int = 0                    # index into FALLBACK_LEVELS
    status: str = "valid"                # "valid", "infeasible" (room > 100 m²), "partial"
    log: list[str] = field(default_factory=list)

    @property
    def H(self) -> int:
        return len(self.rings)

    @property
    def valid(self) -> bool:
        return bool(self.checks) and all(v is True for v in self.checks.values())


def hub_region(plan: Plan, group: Group, devices: np.ndarray | None = None,
               clearance: float = HUB_CLEARANCE_M):
    """Where this group's hub may sit: its rooms, ≥ clearance (0.5 m) from
    walls/windows and ≥ MIN_HUB_NODE_M from every device point."""
    zone = _union([plan.rooms[i].poly for i in group.rooms])
    blocked = plan.walls_win.buffer(clearance) if clearance > 0 else plan.walls_win
    if devices is not None and len(devices):
        # vertices of the buffered circle sit on the axes at exactly MIN_HUB_NODE_M,
        # so a hub straight out from a device at that distance stays legal
        blocked = _union([blocked, shapely.multipoints(devices).buffer(MIN_HUB_NODE_M - HUB_SLACK_M)])
    return safe_op(lambda a, b: a.difference(b), zone, blocked)


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


def _on_loop(group: Group, region, nodes: np.ndarray, method: str = "on-loop",
             clearance: float = HUB_CLEARANCE_M) -> Ring | None:
    """Best hub on a straight stretch of one of the group's loops (adds no cost)."""
    best = None
    for lp in group.loops:
        if lp.degenerate:
            continue
        hit = _longest_clear_stretch(lp.ring, region, nodes)
        if hit is None:
            continue
        hub, seg = hit
        ring = _reanchor(lp.ring, hub, seg)
        if not _stops_once(ring, nodes):
            continue
        m = measure([ring], exempt_points=np.vstack([nodes, hub]))
        cand = Ring(group, lp.order, tuple(hub), ring, m, method, clearance)
        if best is None or cand.J < best.J:
            best = cand
    return best


RELAXED_CLEARANCES_M = (0.4, 0.3, 0.2, 0.1, 0.05, 0.0)


def _relaxed_on_loop(plan: Plan, grid: Grid, group: Group) -> Ring | None:
    """Multi-device zone with no spot 0.5 m from the walls (tiny or
    scale-broken plans): hub on the loop at the largest wall clearance that
    fits, still ≥ MIN_HUB_NODE_M from every device. Flagged via hub_clearance."""
    nodes = grid.pts[grid.node_pt[list(group.rooms)]]
    for c in RELAXED_CLEARANCES_M:
        hit = _on_loop(group, hub_region(plan, group, grid.pts[grid.node_pt], c), nodes, "relaxed", c)
        if hit is not None:
            return hit
    return None


def _stops_once(ring: np.ndarray, nodes: np.ndarray) -> bool:
    """Every own node is a stop of the ring exactly once."""
    return all(int((np.abs(ring[:-1] - xy).max(axis=1) <= TOL).sum()) == 1 for xy in nodes)


def place_hub(plan: Plan, grid: Grid, group: Group, n_candidates: int = N_HUB_CANDIDATES,
              block_nodes: bool = False) -> Ring | None:
    region = hub_region(plan, group, grid.pts[grid.node_pt])
    nodes = grid.pts[grid.node_pt[list(group.rooms)]]

    # (1) free placement on the loop itself
    best = _on_loop(group, region, nodes)
    if best is not None:
        return best

    # legal hub grid points: in the zone, ≥ 0.5 m from walls, not on a device
    shapely.prepare(region)
    ok = np.flatnonzero(shapely.covered_by(shapely.points(grid.pts), region)
                        & np.isin(grid.room_of, group.rooms))
    near_node = np.linalg.norm(grid.pts[ok][:, None, :] - grid.pts[grid.node_pt][None, :, :], axis=2).min(axis=1)
    ok = ok[near_node >= MIN_HUB_NODE_M - HUB_SLACK_M]
    if len(ok) == 0:
        return _micro_hub(plan, grid, group) if group.n_devices == 1 else _relaxed_on_loop(plan, grid, group)

    # (2) one device: hub on a grid point straight out from the node, ring
    #     hub → node → hub, J = 3d + 1 (L 2d, O d, 2 reversals). Any other ring
    #     through a hub at L1 distance ≥ d_min costs ≥ 3·d_min + 1 if it has 2
    #     bends (straight out-and-back) or ≥ 2·d_min + 2 with 4+ bends. When the
    #     direct ring meets that bound it is optimal and the Dijkstra re-route
    #     below is skipped; otherwise both are tried and the cheaper wins.
    if group.n_devices == 1:
        best = _direct_hub(plan, grid, group, region)
        d_min = float(np.abs(grid.pts[ok] - nodes[0]).sum(axis=1).min())
        if best is not None and best.J <= min(3 * d_min + 1, 2 * d_min + 2) + 1e-9:
            return best

    # (3) re-route through candidate hub grid points (1 device with no provably
    #     best direct spot, or loops hugging walls)
    ref = group.loops[0].ring if group.loops else nodes
    d = np.abs(grid.pts[ok][:, None, :] - ref[None, :, :]).sum(axis=2).min(axis=1)
    allowed = zone_allowed(grid, group.rooms)
    orders = [lp.order for lp in group.loops] or [group.rooms]
    for h in ok[np.argsort(d)[:n_candidates]]:
        router = RingRouter(grid, allowed, np.r_[grid.node_pt[list(group.rooms)], h], block_nodes)
        for order in orders[:2]:
            paths = router.ring([int(h)] + [int(grid.node_pt[r]) for r in order])
            if paths is None:
                continue
            legs = [states_to_polyline(grid, p) for p in paths]
            ring = np.vstack([legs[0]] + [lg[1:] for lg in legs[1:]])
            if not _stops_once(ring, nodes):
                continue
            m = measure([ring], exempt_points=np.vstack([nodes, grid.pts[h]]))
            cand = Ring(group, tuple(order), tuple(grid.pts[h]), ring, m, "rerouted")
            if best is None or cand.J < best.J:
                best = cand
    return best


DIRECT_MAX_STEPS = 12     # grid steps walked out from the node per heading


def _neighbours(grid: Grid) -> np.ndarray:
    """nb[p, h]: neighbour of grid point p in heading h (E N W S), -1 if none. Cached."""
    nb = getattr(grid, "_nb", None)
    if nb is None:
        nb = -np.ones((len(grid.pts), 4), np.int64)
        a, b = grid.edges[:, 0], grid.edges[:, 1]
        d = grid.pts[b] - grid.pts[a]
        fwd = np.where(grid.horizontal, np.where(d[:, 0] > 0, 0, 2), np.where(d[:, 1] > 0, 1, 3))
        nb[a, fwd] = b
        nb[b, (fwd + 2) % 4] = a
        grid._nb = nb
    return nb


def _direct_hub(plan: Plan, grid: Grid, group: Group, region) -> Ring | None:
    room = group.rooms[0]
    node = int(grid.node_pt[room])
    n = grid.pts[node]
    allowed = zone_allowed(grid, group.rooms)
    nb = _neighbours(grid)
    best = None
    for h in range(4):
        p = node
        for _ in range(DIRECT_MAX_STEPS):
            p = int(nb[p, h])
            if p < 0 or not allowed[p]:
                break
            if (grid.room_of[p] == room and np.abs(grid.pts[p] - n).sum() >= MIN_HUB_NODE_M - HUB_SLACK_M
                    and region.covers(Point(grid.pts[p]))):
                ring = np.array([grid.pts[p], n, grid.pts[p]])
                m = measure([ring], exempt_points=np.vstack([n, grid.pts[p]]))
                cand = Ring(group, (room,), tuple(grid.pts[p]), ring, m, "direct")
                if best is None or cand.J < best.J:
                    best = cand
                break
    return best


MICRO_STEPS_M = (0.5, 0.3, 0.2, 0.15, MIN_HUB_NODE_M)


def _micro_hub(plan: Plan, grid: Grid, group: Group) -> Ring | None:
    """Last resort for a one-room zone with no spot 0.5 m from the walls
    (narrow closet, sliver): hub a short step from the node along an axis,
    ring hub → node → hub. Picks the step with the most wall clearance; the
    ring is flagged through hub_clearance < 0.5 m."""
    room = group.rooms[0]
    n = grid.pts[grid.node_pt[room]]
    poly = plan.rooms[room].poly
    best = None
    for d in MICRO_STEPS_M:
        for v in ((1, 0), (0, 1), (-1, 0), (0, -1)):
            h = n + d * np.asarray(v, float)
            if not (poly.covers(Point(h)) and shapely.covered_by(LineString([n, h]), plan.free.buffer(1e-6))):
                continue
            if np.linalg.norm(grid.pts[grid.node_pt] - h, axis=1).min() < MIN_HUB_NODE_M - HUB_SLACK_M:
                continue
            c = plan.walls_win.distance(Point(h))
            if best is None or c > best[0] + 1e-9:
                best = (c, h)
    if best is None:
        return None
    c, h = best
    ring = np.array([h, n, h])
    m = measure([ring], exempt_points=np.vstack([n, h]))
    return Ring(group, (room,), tuple(h), ring, m, "micro", min(HUB_CLEARANCE_M, float(c)))


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


RING_RULES = ("≤ 100 m² per hub", "≤ 5 devices per hub", "zones connected through doors/openings",
              "each ring closed at its own hub", "axis-aligned cable",
              "no wall/window crossed, inside boundary", "each node visited once, no foreign nodes",
              "hub ≥ 0.5 m from walls (or best possible, flagged)", "hub inside its zone",
              "hub ≥ 0.1 m from every device")
NODE_RULE = "nodes ≥ 0.5 m from walls (or best possible, flagged)"


class Checker:
    """Hard rules from slides 2–3, checked from the output geometry. Ring
    rules are checked per ring so the fallback can drop just the bad rings."""

    def __init__(self, plan: Plan, grid: Grid, adj):
        self.plan, self.grid, self.adj = plan, grid, adj
        self.node_xy = grid.pts[grid.node_pt]
        self.walls_win = plan.walls_win
        self.free = plan.free.buffer(1e-6)
        shapely.prepare(self.free)

    def ring(self, rg: Ring) -> dict:
        plan, r, g = self.plan, rg.ring, rg.group
        closed = bool(np.abs(r[0] - np.asarray(rg.hub)).max() <= TOL and np.abs(r[-1] - r[0]).max() <= TOL)
        seg = np.diff(r, axis=0)
        line = LineString(r)
        visits = True
        for i in range(len(plan.rooms)):            # own nodes once as a stop; no other node on the cable
            if i in rg.order:
                visits &= int((np.abs(r[:-1] - self.node_xy[i]).max(axis=1) <= TOL).sum()) == 1
            else:
                visits &= not line.distance(Point(self.node_xy[i])) <= TOL
        hp = Point(rg.hub)
        return dict(zip(RING_RULES, (
            g.area <= MAX_AREA_M2 + AREA_TOL_M2,
            len(rg.order) <= 5,
            nx.is_connected(self.adj.subgraph(g.rooms)),
            closed,
            bool(np.all((np.abs(seg[:, 0]) <= TOL) | (np.abs(seg[:, 1]) <= TOL))),
            bool(shapely.covered_by(line, self.free)),
            bool(visits),
            self.walls_win.distance(hp) >= min(HUB_CLEARANCE_M, rg.hub_clearance) - TOL,
            bool(_union([plan.rooms[i].poly for i in rg.order]).buffer(TOL).covers(hp)),
            float(np.linalg.norm(self.node_xy - np.asarray(rg.hub), axis=1).min()) >= MIN_HUB_NODE_M - TOL,
        )))

    def ring_ok(self, rg: Ring) -> bool:
        return all(self.ring(rg).values())

    def __call__(self, rings: list[Ring]) -> dict:
        n = len(self.plan.rooms)
        per = [self.ring(rg) for rg in rings]
        checks = {"every room in exactly one ring": sorted(r for rg in rings for r in rg.order) == list(range(n))}
        checks.update({k: all(p[k] for p in per) for k in RING_RULES})
        checks[NODE_RULE] = all(
            self.walls_win.distance(Point(self.node_xy[i])) >= min(NODE_CLEARANCE_M, r.clearance) - TOL
            for i, r in enumerate(self.plan.rooms))
        return checks


def validate(plan: Plan, grid: Grid, rings: list[Ring], adj) -> dict:
    return Checker(plan, grid, adj)(rings)


def _finish(plan, grid, chosen, checker, t0) -> Solution:
    if chosen is None:
        return Solution(plan, [], Metrics(0, 0, 0, 0), {"feasible grouping exists": False},
                        time.perf_counter() - t0, status="partial")
    exempt = np.vstack([grid.pts[grid.node_pt], np.array([rg.hub for rg in chosen])])
    total = measure([rg.ring for rg in chosen], exempt_points=exempt)
    sol = Solution(plan, chosen, total, checker(chosen), time.perf_counter() - t0)
    sol.status = "valid" if sol.valid else "partial"
    return sol


def _attempt(plan: Plan, cfg: Config, t0: float, log: list[str]) -> tuple[Solution, int]:
    """One rung pair of the ladder: (solution, 0 = DP over all rings,
    1 = DP over rings passing every per-ring rule).

    Rings are built lazily: first only for groups that can appear in a
    fewest-hub cover (the answer always has fewest hubs). Every other group is
    built only if that falls short (a min-cover zone got no ring) or the
    result fails validation, so the outcome is the same as building all."""
    grid = build_grid(plan, cfg.fill)
    adj = room_adjacency(plan, grid)
    groups = enumerate_groups(plan, adj)
    n = len(plan.rooms)
    h_min, useful = min_cover_groups(groups, n)
    rings: dict[int, Ring] = {}
    built: set[int] = set()

    def build(gs):
        todo = [g for g in gs if g.mask not in built]
        solve_all(todo, grid, restrict=True, keep=cfg.keep, block_nodes=cfg.block_nodes)
        for g in todo:
            built.add(g.mask)
            try:
                rg = place_hub(plan, grid, g, cfg.hub_candidates, cfg.block_nodes)
            except Exception as e:      # one bad zone must not sink the plan
                log.append(f"hub placement failed for {g.room_ids}: {type(e).__name__}: {e}")
                continue
            if rg is not None:
                rings[g.mask] = rg

    def pick(pool: dict[int, Ring]):
        # micro hubs break the 0.5 m hub rule (flagged), so only when nothing else covers the plan
        strict = {m: rg for m, rg in pool.items() if rg.hub_clearance >= HUB_CLEARANCE_M}
        chosen = choose_groups(groups, strict, n)
        if chosen is None and len(strict) < len(pool):
            chosen = choose_groups(groups, pool, n)
        return chosen

    build(useful)
    chosen = choose_groups(groups, {m: rg for m, rg in rings.items() if rg.hub_clearance >= HUB_CLEARANCE_M}, n)
    if chosen is None or len(chosen) > (h_min or 0):
        build(groups)
        chosen = pick(rings)
    checker = Checker(plan, grid, adj)
    sol = _finish(plan, grid, chosen, checker, t0)
    sub = 0
    if not sol.valid:
        log.append("failed: " + ", ".join(k for k, v in sol.checks.items() if v is not True))
        build(groups)
        good = {m: rg for m, rg in rings.items() if checker.ring_ok(rg)}
        alt = _finish(plan, grid, pick(good), checker, t0)
        if alt.valid or not sol.rings:
            sol, sub = alt, 1
    sol.grid = grid
    sol.lower_bound = hub_lower_bound(plan)
    sol.zones_built = (len(built), len(groups))
    return sol, sub


def solve_plan(raw: dict, time_limit: float = TIME_LIMIT_S,
               nodes: dict[int, tuple[float, float]] | None = None) -> Solution:
    """Solve one plan, walking down the fallback ladder until a result passes
    every hard rule. Never raises for a plan that builds; check .status.
    nodes: optional device positions replacing ours (see build_plan)."""
    t0 = time.perf_counter()
    plan = build_plan(raw, nodes=nodes)
    big = [r.id for r in plan.rooms if r.area > MAX_AREA_M2 + AREA_TOL_M2]
    # ResPlan sometimes labels a sliver of wall as a room: no free space inside, no cable can reach it
    walled = [r.id for r in plan.rooms
              if safe_op(lambda a, b: a.intersection(b), r.poly, plan.free).area < 1e-6]
    if big or walled:
        sol = Solution(plan, [], Metrics(0, 0, 0, 0), {"feasible grouping exists": False},
                       time.perf_counter() - t0, status="infeasible",
                       log=[f"{i}: room > 100 m²" for i in big] + [f"{i}: room lies inside a wall" for i in walled])
        sol.lower_bound = hub_lower_bound(plan)
        return sol
    log: list[str] = []
    best = None
    for base, cfg in ((0, NORMAL), (2, SAFER)):
        if base and time.perf_counter() - t0 > time_limit / 3:
            log.append("no time left for the safer retry")
            break
        try:
            sol, sub = _attempt(plan, cfg, t0, log)
        except Exception as e:
            log.append(f"{FALLBACK_LEVELS[base]} attempt crashed: {type(e).__name__}: {e}")
            continue
        sol.fallback = base + sub
        if sol.valid:
            best = sol
            break
        if best is None or (sol.rings and not best.rings):
            best = sol
    if best is None:
        best = Solution(plan, [], Metrics(0, 0, 0, 0), {"feasible grouping exists": False})
        best.lower_bound = hub_lower_bound(plan)
    if not best.valid:
        best.fallback, best.status = len(FALLBACK_LEVELS) - 1, "partial"
    best.log = log + plan.flags + [
        f"{'+'.join(rg.group.room_ids)}: hub clearance only {rg.hub_clearance:.2f} m ({rg.hub_method} hub, rooms too narrow)"
        for rg in best.rings if rg.hub_clearance < HUB_CLEARANCE_M]
    best.seconds = time.perf_counter() - t0
    return best


MAX_COVERS = 50_000     # safety cap on enumerated groupings in top_solutions


def top_solutions(raw: dict, k: int = 3, nodes: dict | None = None) -> list[Solution]:
    """The k best groupings (for the UI): every exact cover of the rooms by
    zones whose ring passes every per-ring rule, ranked like choose_groups
    (fewest hubs, then lowest J, then most even area). Rings are built for
    every legal zone, not only fewest-hub ones, so alternatives may use more
    hubs. [] if the plan has no valid answer."""
    t0 = time.perf_counter()
    plan = build_plan(raw, nodes=nodes)
    if any(r.area > MAX_AREA_M2 + AREA_TOL_M2 for r in plan.rooms):
        return []
    grid = build_grid(plan)
    adj = room_adjacency(plan, grid)
    groups = enumerate_groups(plan, adj)
    solve_all(groups, grid, restrict=True)
    checker = Checker(plan, grid, adj)
    rings: dict[int, Ring] = {}
    for g in groups:
        try:
            rg = place_hub(plan, grid, g)
        except Exception:
            continue
        if rg is not None and checker.ring_ok(rg):
            rings[g.mask] = rg
    n = len(plan.rooms)
    by_low: dict[int, list[int]] = {}
    for m in rings:
        by_low.setdefault((m & -m).bit_length() - 1, []).append(m)
    covers: list[tuple[int, ...]] = []

    def rec(rem: int, acc: tuple[int, ...]):
        if len(covers) >= MAX_COVERS:
            return
        if rem == 0:
            covers.append(acc)
            return
        low = (rem & -rem).bit_length() - 1
        for m in by_low.get(low, ()):
            if m & rem == m:
                rec(rem ^ m, acc + (m,))

    rec((1 << n) - 1, ())
    flagged = lambda c: sum(rings[m].hub_clearance < HUB_CLEARANCE_M for m in c)
    key = lambda c: (flagged(c) > 0, len(c), sum(rings[m].J for m in c),
                     float(np.ptp([rings[m].group.area for m in c])))
    out = []
    for c in sorted(covers, key=key)[:k]:
        sol = _finish(plan, grid, [rings[m] for m in c], checker, t0)
        sol.grid, sol.lower_bound = grid, hub_lower_bound(plan)
        sol.log = plan.flags[:]
        out.append(sol)
    return out


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
        "status": sol.status,
        "fallback": FALLBACK_LEVELS[sol.fallback],
        "log": sol.log,
    }


def dump(sol: Solution, path) -> None:
    with open(path, "w") as f:
        json.dump(to_json(sol), f, indent=1, default=float)
