"""Phase 1: bend-aware shortest paths on the routing grid.

Path cost = L + 0.5·B. To see a bend, a search state must know which way the
cable is heading, so every grid point is split into 4 states, one per heading:

    state = point * 4 + heading,   heading ∈ {E, N, W, S}

    move edge  (p, h) → (q, h)    cost = |pq|, only if q is p's neighbour in direction h
    turn edge  (p, h) → (p, h')   cost = 0.5 (90° turn, or a reversal, which counts once)

One Dijkstra run (scipy, in C) from all 4 heading-states of every device node
gives the minimum cost to every state, plus predecessors to rebuild routes.

Bends *at* device nodes are not counted inside a leg: a leg is indexed by the
heading it leaves its start node with (s) and arrives at its end node with
(e), and the ring search in Phase 3 adds 0.5 where an arriving heading differs
from the next leaving heading.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from scipy.sparse.csgraph import dijkstra

from .grid import Grid

E, N, W, S = 0, 1, 2, 3
HEADINGS = "ENWS"
BEND_COST = 0.5


def build_state_graph(grid: Grid, allowed: np.ndarray | None = None) -> sp.csr_matrix:
    """Directed graph over 4·P heading states.

    allowed: optional (P,) bool mask of usable grid points, e.g. to keep a
    ring inside its own group's rooms. Edges touching a masked point vanish.
    """
    P = len(grid.pts)
    a, b = grid.edges[:, 0], grid.edges[:, 1]
    d = grid.pts[b] - grid.pts[a]
    # Heading of travel a→b; travel b→a is the opposite heading.
    fwd = np.where(grid.horizontal, np.where(d[:, 0] > 0, E, W), np.where(d[:, 1] > 0, N, S))
    rev = (fwd + 2) % 4
    if allowed is not None:
        keep = allowed[a] & allowed[b]
        a, b, fwd, rev, length = a[keep], b[keep], fwd[keep], rev[keep], grid.lengths[keep]
    else:
        length = grid.lengths

    rows = [a * 4 + fwd, b * 4 + rev]
    cols = [b * 4 + fwd, a * 4 + rev]
    vals = [length, length]

    # Turn edges between every pair of headings at every point.
    p = np.arange(P)
    for h1 in range(4):
        for h2 in range(4):
            if h1 != h2:
                rows.append(p * 4 + h1)
                cols.append(p * 4 + h2)
                vals.append(np.full(P, BEND_COST))
    # All weights are > 0, so scipy's "explicit zero = no edge" rule never bites.
    return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                         shape=(4 * P, 4 * P))


@dataclass
class RouteTable:
    """Shortest bend-aware routes from every device node.

    leg[a, s, b, e] = min L + 0.5·B from node a (leaving with heading s) to
    node b (arriving with heading e); inf if unreachable.
    """
    grid: Grid
    node_pt: np.ndarray        # (n,) grid point of each device node
    dist: np.ndarray           # (4n, 4P) cost from source state to every state
    pred: np.ndarray           # (4n, 4P) predecessor state, -9999 = none
    leg: np.ndarray            # (n, 4, n, 4)

    def path_states(self, a: int, s: int, b: int, e: int) -> list[int]:
        """State sequence of the route behind leg[a, s, b, e]."""
        row = a * 4 + s
        target = self.node_pt[b] * 4 + e
        if not np.isfinite(self.dist[row, target]):
            raise ValueError(f"node {b} unreachable from node {a}")
        out = [target]
        while out[-1] != self.node_pt[a] * 4 + s:
            out.append(self.pred[row, out[-1]])
        return out[::-1]

    def polyline(self, a: int, s: int, b: int, e: int) -> np.ndarray:
        """Route as corner points only (start, each bend, end), metres."""
        pts = self.grid.pts[np.array(self.path_states(a, s, b, e)) // 4]
        pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]   # drop turn-in-place repeats
        return simplify(pts)

    def best(self, a: int, b: int) -> tuple[float, int, int]:
        """Cheapest leg a→b over all leave/arrive headings: (cost, s, e)."""
        sub = self.leg[a, :, b, :]
        s, e = np.unravel_index(np.argmin(sub), sub.shape)
        return float(sub[s, e]), int(s), int(e)


def route_table(grid: Grid, allowed: np.ndarray | None = None,
                nodes: np.ndarray | None = None) -> RouteTable:
    """Dijkstra from all 4 heading-states of each device node.

    nodes: indices into grid.node_pt to route between (default: all rooms).
    """
    node_pt = grid.node_pt if nodes is None else grid.node_pt[nodes]
    G = build_state_graph(grid, allowed)
    sources = (node_pt[:, None] * 4 + np.arange(4)[None, :]).ravel()     # (4n,)
    dist, pred = dijkstra(G, directed=True, indices=sources, return_predecessors=True)
    n = len(node_pt)
    targets = (node_pt[:, None] * 4 + np.arange(4)[None, :]).ravel()
    leg = dist[:, targets].reshape(n, 4, n, 4)
    return RouteTable(grid=grid, node_pt=node_pt, dist=dist, pred=pred, leg=leg)


def simplify(pts: np.ndarray) -> np.ndarray:
    """Keep only endpoints and points where the direction changes."""
    if len(pts) <= 2:
        return pts
    d = np.sign(np.diff(pts, axis=0))
    turn = np.any(d[1:] != d[:-1], axis=1)
    return np.vstack([pts[0], pts[1:-1][turn], pts[-1]])


def polyline_cost(pts: np.ndarray) -> tuple[float, int]:
    """(L, B) of an axis-aligned polyline, counted the way the checker does:
    length of every segment, one bend per direction change (reversal = 1)."""
    seg = np.diff(pts, axis=0)
    L = float(np.abs(seg).sum())
    d = np.sign(seg)
    B = int(np.any(d[1:] != d[:-1], axis=1).sum()) if len(d) > 1 else 0
    return L, B
