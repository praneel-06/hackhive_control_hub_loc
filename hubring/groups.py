"""Phase 2: candidate room groups (one group = one hub's zone).

A group is valid when its rooms
  * form one connected zone through doors or open passages,
  * total at most 100 m² of eligible-room area,
  * hold at most 5 devices (one per room).

Rooms are numbered 0..n-1 (the order of plan.rooms), so a group is a bitmask:
bit i set ⇔ room i is in the group. The mask is the group's identity, which
makes duplicates impossible however a group is reached during enumeration.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np
from scipy.cluster.hierarchy import DisjointSet

from .grid import Grid
from .plan import Plan

MAX_AREA_M2 = 100.0
MAX_DEVICES = 5
AREA_TOL_M2 = 1e-6


@dataclass
class Group:
    mask: int                       # bit i ⇔ plan.rooms[i] in the group
    rooms: tuple[int, ...]          # room indices, ascending
    room_ids: tuple[str, ...]
    area: float                     # eligible-room area, m²
    n_devices: int
    links: tuple[tuple[int, int, str], ...]   # passages inside the group (a, b, type)
    loops: list = field(default_factory=list)  # Phase 3 fills in ring candidates

    @property
    def lowest(self) -> int:
        return (self.mask & -self.mask).bit_length() - 1

    def __repr__(self) -> str:
        return (f"Group({'+'.join(self.room_ids)}, {self.area:.1f} m², "
                f"{self.n_devices} dev, mask={self.mask:#b})")


def room_adjacency(plan: Plan, grid: Grid) -> nx.Graph:
    """Which rooms can reach each other directly (not through a third room).

    Built from the routing grid, so it matches exactly where a cable can go:
      * 'door'    — the rooms meet through a door/opening gap (grid points
                    outside every room). Rooms touching one gap region are
                    linked pairwise.
      * 'opening' — a grid edge runs straight from one room into another
                    (room polygons that touch with no wall between).
    ResPlan's own edge type is kept as an attribute where it has one.
    """
    n = len(plan.rooms)
    G = nx.Graph()
    for i, r in enumerate(plan.rooms):
        G.add_node(i, id=r.id, type=r.type, area=r.area)

    a, b = grid.edges[:, 0], grid.edges[:, 1]
    ra, rb = grid.room_of[a], grid.room_of[b]

    # Room ↔ room straight across (open passage).
    direct = (ra >= 0) & (rb >= 0) & (ra != rb)
    for u, v in set(map(tuple, np.sort(np.column_stack([ra[direct], rb[direct]]), axis=1))):
        G.add_edge(int(u), int(v), via="opening")

    # Gap regions: connected components of gap points (room_of == -1).
    gap = np.flatnonzero(grid.room_of < 0)
    ds = DisjointSet(gap.tolist())
    both_gap = (ra < 0) & (rb < 0)
    for u, v in zip(a[both_gap], b[both_gap]):
        ds.merge(int(u), int(v))
    touches: dict[int, set[int]] = {}
    for p, r in ((a, rb), (b, ra)):            # gap point p next to a room point
        m = (grid.room_of[p] < 0) & (r >= 0)
        for g, room in zip(p[m], r[m]):
            touches.setdefault(ds[int(g)], set()).add(int(room))
    for rooms in touches.values():
        rooms = sorted(rooms)
        for i in range(len(rooms)):
            for j in range(i + 1, len(rooms)):
                if not G.has_edge(rooms[i], rooms[j]):
                    G.add_edge(rooms[i], rooms[j], via="door")

    idx = {r.id: i for i, r in enumerate(plan.rooms)}
    for u, v, t in plan.room_graph.edges(data="type"):
        if G.has_edge(idx[u], idx[v]):
            G.edges[idx[u], idx[v]]["resplan"] = t
    return G


def enumerate_groups(plan: Plan, adj: nx.Graph,
                     max_area: float = MAX_AREA_M2, max_devices: int = MAX_DEVICES) -> list[Group]:
    """Every connected room set within the area and device limits.

    Grows sets one adjacent room at a time from each single room. A set that
    breaks a limit is dropped and never grown (adding rooms only increases
    area and device count). The `seen` bitmask set stops the same group being
    built twice via different growth orders.
    """
    n = len(plan.rooms)
    area = np.array([r.area for r in plan.rooms])
    nbr = [0] * n
    for u, v in adj.edges:
        nbr[u] |= 1 << v
        nbr[v] |= 1 << u

    def fits(mask: int) -> bool:
        return (mask.bit_count() <= max_devices
                and area[[i for i in range(n) if mask >> i & 1]].sum() <= max_area + AREA_TOL_M2)

    seen: set[int] = set()
    frontier = [1 << i for i in range(n) if fits(1 << i)]
    seen.update(frontier)
    while frontier:
        nxt = []
        for mask in frontier:
            border = 0
            for i in range(n):
                if mask >> i & 1:
                    border |= nbr[i]
            border &= ~mask
            while border:
                bit = border & -border
                border ^= bit
                m2 = mask | bit
                if m2 not in seen and fits(m2):
                    seen.add(m2)
                    nxt.append(m2)
        frontier = nxt

    groups = []
    for mask in sorted(seen, key=lambda m: (m.bit_count(), m)):
        rooms = tuple(i for i in range(n) if mask >> i & 1)
        sub = adj.subgraph(rooms)
        groups.append(Group(
            mask=mask,
            rooms=rooms,
            room_ids=tuple(plan.rooms[i].id for i in rooms),
            area=float(area[list(rooms)].sum()),
            n_devices=len(rooms),
            links=tuple((u, v, d["via"]) for u, v, d in sub.edges(data=True)),
        ))
    return groups


def hub_lower_bound(plan: Plan) -> int:
    total = sum(r.area for r in plan.rooms)
    return max(int(np.ceil(total / MAX_AREA_M2 - 1e-9)), int(np.ceil(len(plan.rooms) / MAX_DEVICES)))


def min_hubs(groups: list[Group], n: int) -> tuple[int | None, int]:
    """Fewest groups that cover every room exactly once, and how many distinct
    partitions reach that minimum. (None, 0) if no exact cover exists.

    DP over room bitmasks. To count each partition once, the group chosen at
    each step must contain the lowest uncovered room.
    """
    full = (1 << n) - 1
    by_low: dict[int, list[int]] = {}
    for g in groups:
        by_low.setdefault(g.lowest, []).append(g.mask)
    INF = 10 ** 9
    best = {0: (0, 1)}            # remaining-mask → (hubs, number of optimal partitions)

    def solve(rem: int) -> tuple[int, int]:
        if rem in best:
            return best[rem]
        low = (rem & -rem).bit_length() - 1
        out = (INF, 0)
        for m in by_low.get(low, ()):
            if m & rem == m:
                h, c = solve(rem ^ m)
                if h + 1 < out[0]:
                    out = (h + 1, c)
                elif h + 1 == out[0]:
                    out = (out[0], out[1] + c)
        best[rem] = out
        return out

    h, c = solve(full)
    return (None, 0) if h >= INF else (h, c)


def min_cover_groups(groups: list[Group], n: int) -> tuple[int | None, list[Group]]:
    """Fewest hubs H* and the groups that appear in at least one H*-hub
    exact cover. Only those can be in the final answer (fewest hubs is scored
    first), so only they need rings. g qualifies iff the rest of the plan can
    be covered exactly by H* − 1 groups. (None, all groups) if no cover exists.
    """
    full = (1 << n) - 1
    by_low: dict[int, list[int]] = {}
    for g in groups:
        by_low.setdefault(g.lowest, []).append(g.mask)
    INF = 10 ** 9
    memo = {0: 0}

    def f(rem: int) -> int:                      # fewest groups covering rem exactly
        if rem in memo:
            return memo[rem]
        low = (rem & -rem).bit_length() - 1
        out = INF
        for m in by_low.get(low, ()):
            if m & rem == m:
                out = min(out, f(rem ^ m) + 1)
        memo[rem] = out
        return out

    h = f(full)
    if h >= INF:
        return None, list(groups)
    return h, [g for g in groups if f(full ^ g.mask) == h - 1]
