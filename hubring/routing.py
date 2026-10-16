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


@dataclass
class StateGraphParts:
    """Edges of the heading-state graph, kept apart so weights can be changed
    (e.g. collision penalties) and the matrix rebuilt cheaply."""
    n_states: int
    move_rows: np.ndarray
    move_cols: np.ndarray
    move_len: np.ndarray
    move_edge: np.ndarray      # grid edge id of each move
    move_dst: np.ndarray       # grid point a move enters
    turn_rows: np.ndarray
    turn_cols: np.ndarray

    def matrix(self, move_w: np.ndarray | None = None, extra: tuple | None = None) -> sp.csr_matrix:
        """CSR matrix; move_w overrides move weights; extra = (rows, cols,
        vals, n_extra_states) appends edges from additional states."""
        w = self.move_len if move_w is None else move_w
        rows = [self.move_rows, self.turn_rows]
        cols = [self.move_cols, self.turn_cols]
        vals = [w, np.full(len(self.turn_rows), BEND_COST)]
        n = self.n_states
        if extra is not None:
            rows.append(extra[0]); cols.append(extra[1]); vals.append(extra[2]); n += extra[3]
        # All weights are > 0, so scipy's "explicit zero = no edge" rule never bites.
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                             shape=(n, n))


def state_graph_parts(grid: Grid, allowed: np.ndarray | None = None) -> StateGraphParts:
    P = len(grid.pts)
    eid = np.arange(len(grid.edges))
    a, b = grid.edges[:, 0], grid.edges[:, 1]
    d = grid.pts[b] - grid.pts[a]
    # Heading of travel a→b; travel b→a is the opposite heading.
    fwd = np.where(grid.horizontal, np.where(d[:, 0] > 0, E, W), np.where(d[:, 1] > 0, N, S))
    rev = (fwd + 2) % 4
    length = grid.lengths
    if allowed is not None:
        keep = allowed[a] & allowed[b]
        a, b, fwd, rev, length, eid = a[keep], b[keep], fwd[keep], rev[keep], length[keep], eid[keep]

    # Turn edges between every pair of headings at every point.
    p = np.arange(P)
    h1, h2 = np.array([(i, j) for i in range(4) for j in range(4) if i != j]).T
    return StateGraphParts(
        n_states=4 * P,
        move_rows=np.concatenate([a * 4 + fwd, b * 4 + rev]),
        move_cols=np.concatenate([b * 4 + fwd, a * 4 + rev]),
        move_len=np.concatenate([length, length]),
        move_edge=np.concatenate([eid, eid]),
        move_dst=np.concatenate([b, a]),
        turn_rows=(p[:, None] * 4 + h1[None, :]).ravel(),
        turn_cols=(p[:, None] * 4 + h2[None, :]).ravel(),
    )


def build_state_graph(grid: Grid, allowed: np.ndarray | None = None) -> sp.csr_matrix:
    """Directed graph over 4·P heading states.

    allowed: optional (P,) bool mask of usable grid points, e.g. to keep a
    ring inside its own group's rooms. Edges touching a masked point vanish.
    """
    return state_graph_parts(grid, allowed).matrix()


def states_to_polyline(grid: Grid, states) -> np.ndarray:
    """Corner points of a state path (turn-in-place repeats dropped)."""
    pts = grid.pts[np.asarray(states) // 4]
    pts = pts[np.r_[True, np.any(np.diff(pts, axis=0) != 0, axis=1)]]
    return simplify(pts)


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
        return states_to_polyline(self.grid, self.path_states(a, s, b, e))

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
