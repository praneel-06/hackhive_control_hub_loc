"""Checker-style metrics computed from polylines alone (slides 6–7).

    L  total length of every segment                    (1 per m)
    B  direction changes along each closed ring, nodes included;
       a reversal counts once                           (0.5 each)
    X  points where two cables meet away from a node or hub,
       inside one ring or between rings, once per point  (5 each)
    O  length where cable lies on top of other cable on the same line;
       a stretch covered by k legs counts (k − 1)·length (1 per m)

Tolerance 1 mm: closer points are the same point, lines within 1 mm are the
same line. Polylines are arrays of corner points in metres.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

TOL = 1e-3
W_CROSS, W_OVERLAP, W_BEND = 5.0, 1.0, 0.5


@dataclass
class Metrics:
    L: float
    B: int
    X: int
    O: float

    @property
    def J(self) -> float:
        return self.L + W_CROSS * self.X + W_OVERLAP * self.O + W_BEND * self.B


def ring_bends(ring: np.ndarray) -> int:
    """Direction changes around a closed ring (first point == last point),
    including the turn where the ring closes. A reversal counts once."""
    d = np.sign(np.round(np.diff(ring, axis=0), 6))
    d = d[np.any(d != 0, axis=1)]
    if len(d) < 2:
        return 0
    return int(np.any(d != np.roll(d, 1, axis=0), axis=1).sum())


def _segments(rings: list[np.ndarray]):
    """(S, 4) x0,y0,x1,y1 plus ring id and position, skipping zero-length."""
    segs, rid, pos = [], [], []
    for r, ring in enumerate(rings):
        k = 0
        for p, q in zip(ring[:-1], ring[1:]):
            if np.abs(q - p).sum() > TOL:
                segs.append((*p, *q)); rid.append(r); pos.append(k); k += 1
    return np.array(segs).reshape(-1, 4), np.array(rid), np.array(pos)


def _overlap(lines: np.ndarray, starts: np.ndarray, ends: np.ndarray) -> float:
    """Σ (k−1)·length over stretches covered k ≥ 2 times, per shared line."""
    total = 0.0
    key = np.round(lines / TOL).astype(np.int64)
    for c in np.unique(key):
        m = key == c
        if m.sum() < 2:
            continue
        ev = sorted([(s, 1) for s in starts[m]] + [(e, -1) for e in ends[m]])
        depth, last = 0, None
        for x, step in ev:
            if depth >= 2 and last is not None:
                total += (depth - 1) * (x - last)
            depth += step
            last = x
    return total


def _crossings(segs, rid, pos, n_in_ring, exempt: np.ndarray) -> int:
    horiz = np.abs(segs[:, 1] - segs[:, 3]) <= TOL
    H, V = np.flatnonzero(horiz), np.flatnonzero(~horiz)
    if len(H) == 0 or len(V) == 0:
        return 0
    hy = segs[H, 1]
    hx0, hx1 = np.minimum(segs[H, 0], segs[H, 2]), np.maximum(segs[H, 0], segs[H, 2])
    vx = segs[V, 0]
    vy0, vy1 = np.minimum(segs[V, 1], segs[V, 3]), np.maximum(segs[V, 1], segs[V, 3])
    hit = ((vx[None, :] >= hx0[:, None] - TOL) & (vx[None, :] <= hx1[:, None] + TOL)
           & (hy[:, None] >= vy0[None, :] - TOL) & (hy[:, None] <= vy1[None, :] + TOL))
    # Consecutive segments of one ring meet at their shared corner: a bend, not a crossing.
    same = rid[H][:, None] == rid[V][None, :]
    n = n_in_ring[rid[H]][:, None]
    gap = np.abs(pos[H][:, None] - pos[V][None, :])
    hit &= ~(same & ((gap == 1) | (gap == n - 1)))
    hi, vi = np.nonzero(hit)
    pts = np.column_stack([vx[vi], hy[hi]])
    if len(exempt):
        far = np.min(np.abs(pts[:, None, :] - exempt[None, :, :]).max(axis=2), axis=1) > TOL
        pts = pts[far]
    return len(np.unique(np.round(pts / TOL).astype(np.int64), axis=0))


def measure(rings: list[np.ndarray], exempt_points: np.ndarray | None = None) -> Metrics:
    """Metrics over one or more closed rings. exempt_points: device nodes and
    hubs, where cables may meet without counting as a crossing."""
    segs, rid, pos = _segments(rings)
    if len(segs) == 0:
        return Metrics(0.0, 0, 0, 0.0)
    L = float(np.abs(segs[:, 2] - segs[:, 0]).sum() + np.abs(segs[:, 3] - segs[:, 1]).sum())
    B = sum(ring_bends(r) for r in rings)
    horiz = np.abs(segs[:, 1] - segs[:, 3]) <= TOL
    O = (_overlap(segs[horiz, 1], np.minimum(segs[horiz, 0], segs[horiz, 2]),
                  np.maximum(segs[horiz, 0], segs[horiz, 2]))
         + _overlap(segs[~horiz, 0], np.minimum(segs[~horiz, 1], segs[~horiz, 3]),
                    np.maximum(segs[~horiz, 1], segs[~horiz, 3])))
    n_in_ring = np.bincount(rid)
    X = _crossings(segs, rid, pos, n_in_ring,
                   np.zeros((0, 2)) if exempt_points is None else np.asarray(exempt_points))
    return Metrics(L, B, X, O)
