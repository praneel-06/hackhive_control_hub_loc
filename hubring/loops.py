"""Phase 3: the cheapest closed loop through each candidate group's devices.

No hub yet: a loop visits the group's devices N1 → N2 → … → Nk → N1. The hub
is placed onto the finished loop later.

For each group:
  1. Every visiting order. The first device is fixed and a loop equals its
     reverse, so k devices give (k−1)!/2 orders (12 for k = 5).
  2. For each order, a heading DP around the loop picks the leave/arrive
     heading of every leg from Phase 1's leg table, adding 0.5 wherever a
     device's arriving heading differs from its next leaving heading.
     Result: the exact minimum L + 0.5·B for that order.
  3. Independent shortest legs often reuse the same lines (an out-and-back
     to a one-door room retraces itself: full overlap, and its corners touch).
     So the best few orders are re-routed leg by leg with collision
     penalties: +1 per metre on a line this loop already uses (overlap) and
     +5 for entering a point it already passes (crossing), followed by one
     rip-up-and-reroute pass. Door side-lanes in the grid give the second
     cable a parallel path, like slide 5's reference.
  4. Every candidate is scored with the checker's metrics and the lowest
     J = L + 5X + O + 0.5B wins.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations

import numpy as np

from .grid import Grid
from .groups import Group
from .metrics import Metrics, measure
from scipy.sparse.csgraph import dijkstra

from .routing import BEND_COST, RouteTable, route_table, state_graph_parts, states_to_polyline

# turn[e, s]: cost of arriving with heading e and leaving with heading s.
TURN = BEND_COST * (1 - np.eye(4))
W_OVERLAP_PEN, W_CROSS_PEN = 1.0, 5.0   # match J's weights for overlap and crossings
KEEP_TOP = 4          # orders re-routed collision-aware and rescored exactly per group
EPS = 1e-9            # stand-in for a zero-cost edge (scipy drops explicit zeros)


@dataclass
class Loop:
    order: tuple[int, ...]                 # room indices (plan level), visiting order
    headings: tuple[tuple[int, int], ...]  # per leg: (leave heading, arrive heading)
    legs: list[np.ndarray]                 # per leg: corner points, metres
    ring: np.ndarray                       # closed loop, first point == last
    dp_cost: float                         # L + 0.5·B of this order from the heading DP (independent legs)
    metrics: Metrics                       # exact, from the polyline
    method: str = "independent"            # or "collision-aware"

    @property
    def J(self) -> float:
        return self.metrics.J

    @property
    def degenerate(self) -> bool:
        """1-device group: no loop until the hub is placed."""
        return len(self.order) == 1


class RingRouter:
    """Routes a loop's legs one at a time on penalised copies of the
    heading-state graph, so later legs avoid the lines and points that
    earlier legs of the same loop use."""

    def __init__(self, grid: Grid, allowed: np.ndarray | None, exempt_pts: np.ndarray):
        self.grid = grid
        self.parts = state_graph_parts(grid, allowed)
        self.P = len(grid.pts)
        self.exempt = np.zeros(self.P, bool)
        self.exempt[exempt_pts] = True
        # grid edge id of every consecutive point pair, for usage tracking
        self.edge_of = {}
        for i, (u, v) in enumerate(grid.edges):
            self.edge_of[(u, v)] = self.edge_of[(v, u)] = i

    def usage(self, states_list):
        used_e = np.zeros(len(self.grid.edges), np.int64)
        used_p = np.zeros(self.P, np.int64)
        for states in states_list:
            pts = np.asarray(states) // 4
            pts = pts[np.r_[True, pts[1:] != pts[:-1]]]
            used_p[pts] += 1
            for u, v in zip(pts[:-1], pts[1:]):
                used_e[self.edge_of[(int(u), int(v))]] += 1
        used_p[self.exempt] = 0
        return used_e, used_p

    def route(self, a_pt: int, b_pt: int, start_cost: np.ndarray, end_cost: np.ndarray,
              used_e: np.ndarray, used_p: np.ndarray):
        """Cheapest leg a→b with penalties. start_cost[s]: extra cost of
        leaving with heading s; end_cost[e]: of arriving with heading e.
        Returns (penalised cost, state path)."""
        pr = self.parts
        w = (pr.move_len * (1 + W_OVERLAP_PEN * used_e[pr.move_edge])
             + W_CROSS_PEN * (used_p[pr.move_dst] > 0))
        src = pr.n_states                                  # virtual start state
        extra = (np.full(4, src), a_pt * 4 + np.arange(4), start_cost + EPS, 1)
        G = pr.matrix(w, extra)
        dist, pred = dijkstra(G, directed=True, indices=src, return_predecessors=True)
        tgt = b_pt * 4 + np.arange(4)
        tot = dist[tgt] + end_cost
        e = int(np.argmin(tot))
        if not np.isfinite(tot[e]):
            return np.inf, None
        path = [int(tgt[e])]
        while pred[path[-1]] != src:
            path.append(int(pred[path[-1]]))
        return float(tot[e]), path[::-1]

    def ring(self, order_pts: list[int], passes: int = 1):
        """Route a closed loop through order_pts. Returns list of state paths."""
        k = len(order_pts)
        paths: list = [None] * k
        zero = np.zeros(4)
        for i in range(k):
            prev_e = paths[i - 1][-1] % 4 if i > 0 else None
            start = zero if prev_e is None else TURN[prev_e]
            end = TURN[:, paths[0][0] % 4] if i == k - 1 else zero
            used_e, used_p = self.usage([p for p in paths if p is not None])
            _, paths[i] = self.route(order_pts[i], order_pts[(i + 1) % k], start, end, used_e, used_p)
            if paths[i] is None:
                return None
        for _ in range(passes):                            # rip up and reroute each leg
            for i in range(k):
                others = [paths[j] for j in range(k) if j != i]
                used_e, used_p = self.usage(others)
                start = TURN[paths[i - 1][-1] % 4]
                end = TURN[:, paths[(i + 1) % k][0] % 4]
                _, new = self.route(order_pts[i], order_pts[(i + 1) % k], start, end, used_e, used_p)
                if new is not None:
                    paths[i] = new
        return paths


def visiting_orders(k: int):
    """Local orders 0..k-1 with 0 first, one of each reverse pair."""
    if k <= 2:
        yield tuple(range(k))
        return
    for rest in permutations(range(1, k)):
        if rest[0] < rest[-1]:
            yield (0, *rest)


def heading_dp(legs4: list[np.ndarray]) -> tuple[float, list[tuple[int, int]]]:
    """Cheapest headings around a closed loop.

    legs4[i][s, e]: cost of leg i leaving with heading s and arriving with e.
    Returns (cost, [(s_i, e_i) per leg]) including the bend at every device,
    the start device too.
    """
    k = len(legs4)
    best = (np.inf, None)
    for s0 in range(4):
        cur = np.full(4, np.inf); cur[s0] = 0.0
        back = []                       # per leg: (argmin s for each e, argmin e for each next s)
        for i, leg in enumerate(legs4):
            arr_all = cur[:, None] + leg                 # [s, e]
            s_of_e = arr_all.argmin(axis=0)
            arr = arr_all.min(axis=0)                    # [e]
            if i < k - 1:
                nxt_all = arr[:, None] + TURN            # [e, s_next]
                e_of_s = nxt_all.argmin(axis=0)
                cur = nxt_all.min(axis=0)
                back.append((s_of_e, e_of_s))
            else:
                close = arr + TURN[:, s0]                # back at the start device
                e_last = int(close.argmin())
                if close[e_last] < best[0]:
                    # walk back
                    picks = []
                    e = e_last
                    s = int(s_of_e[e])
                    picks.append((s, e))
                    for j in range(k - 2, -1, -1):
                        s_prev_of_e, e_of_s_next = back[j]
                        e = int(e_of_s_next[s])
                        s = int(s_prev_of_e[e])
                        picks.append((s, e))
                    best = (float(close[e_last]), picks[::-1])
    return best


def _legs_for(group: Group, grid: Grid, restrict: bool, shared: RouteTable | None):
    """Route table over this group's devices, local index i ↔ group.rooms[i]."""
    if restrict:
        allowed = np.isin(grid.room_of, group.rooms) | (grid.room_of < 0)
        return route_table(grid, allowed=allowed, nodes=np.array(group.rooms))
    sub = np.array(group.rooms)
    rt = shared
    # View of the shared table restricted to the group's devices.
    return _SubTable(rt, sub)


class _SubTable:
    def __init__(self, rt: RouteTable, sub: np.ndarray):
        self.rt, self.sub = rt, sub
        self.leg = rt.leg[np.ix_(sub, range(4), sub, range(4))]

    def polyline(self, a, s, b, e):
        return self.rt.polyline(self.sub[a], s, self.sub[b], e)


def _make_loop(rooms, heads, legs, dp_cost, nodes, method) -> Loop:
    ring = np.vstack([legs[0]] + [lg[1:] for lg in legs[1:]])
    return Loop(order=rooms, headings=heads, legs=legs, ring=ring, dp_cost=dp_cost,
                metrics=measure([ring], exempt_points=nodes), method=method)


def best_loops(group: Group, grid: Grid, restrict: bool = True,
               shared: RouteTable | None = None, keep: int = KEEP_TOP) -> list[Loop]:
    """Ranked loops for one group (best J first). Also stored on group.loops.

    restrict: route only through this group's rooms and door gaps, so loops
    of different groups cannot cross or overlap and J adds up across groups.
    """
    k = group.n_devices
    if k == 1:
        p = grid.pts[grid.node_pt[group.rooms[0]]]
        group.loops = [Loop(group.rooms, (), [], np.array([p, p]), 0.0, Metrics(0.0, 0, 0, 0.0))]
        return group.loops

    rt = _legs_for(group, grid, restrict, shared)
    scored = []
    for order in visiting_orders(k):
        legs4 = [rt.leg[order[i], :, order[(i + 1) % k], :] for i in range(k)]
        cost, heads = heading_dp(legs4)
        if np.isfinite(cost):
            scored.append((cost, order, heads))
    scored.sort(key=lambda t: t[0])

    nodes = grid.pts[grid.node_pt[list(group.rooms)]]
    allowed = (np.isin(grid.room_of, group.rooms) | (grid.room_of < 0)) if restrict else None
    router = RingRouter(grid, allowed, grid.node_pt[list(group.rooms)])
    loops = []
    for cost, order, heads in scored[:keep]:
        rooms = tuple(group.rooms[i] for i in order)
        # (a) independent shortest legs, exactly as the heading DP priced them
        legs = [rt.polyline(order[i], s, order[(i + 1) % k], e) for i, (s, e) in enumerate(heads)]
        loops.append(_make_loop(rooms, tuple(heads), legs, cost, nodes, "independent"))
        # (b) collision-aware re-route of the same order
        paths = router.ring([int(grid.node_pt[r]) for r in rooms])
        if paths is not None:
            legs = [states_to_polyline(grid, p) for p in paths]
            heads_b = tuple((p[0] % 4, p[-1] % 4) for p in paths)
            loops.append(_make_loop(rooms, heads_b, legs, cost, nodes, "collision-aware"))
    loops.sort(key=lambda lp: lp.J)
    group.loops = loops
    return loops


def solve_all(groups: list[Group], grid: Grid, restrict: bool = True) -> None:
    """Fill group.loops for every candidate group."""
    shared = None if restrict else route_table(grid)
    for g in groups:
        best_loops(g, grid, restrict=restrict, shared=shared)
