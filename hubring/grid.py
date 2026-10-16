"""Non-uniform axis-aligned routing grid with only legal (wall-free) edges.

Grid lines are placed where routes need them (nodes, door centres, just inside
room walls) and filled sparsely elsewhere. Each line intersection inside free
space is a grid point; an edge joins neighbouring points on the same line if
the segment stays inside free space, i.e. crosses no wall or window.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon

from .plan import MIN_HUB_NODE_M, Plan

WALL_OFFSET_M = 0.15   # run cable this far inside a room's wall
FILL_SPACING_M = 1.0   # max gap between grid lines in open areas
MERGE_M = 0.05         # lines closer than this merge into one
DOOR_LANE_M = 0.15     # side lanes either side of a door centre line

# Line priorities: when two lines merge, the higher one keeps its position.
PRI_NODE, PRI_DOOR, PRI_HUB, PRI_WALL, PRI_FILL = 3, 2, 1.5, 1, 0


@dataclass
class Grid:
    xs: np.ndarray            # sorted x lines
    ys: np.ndarray            # sorted y lines
    pts: np.ndarray           # (P, 2) grid point coordinates, metres
    room_of: np.ndarray       # (P,) index into plan.rooms, -1 = door/opening gap
    edges: np.ndarray         # (E, 2) point index pairs, axis-aligned
    lengths: np.ndarray       # (E,) metres
    horizontal: np.ndarray    # (E,) bool
    node_pt: np.ndarray       # (R,) grid point index of each room's node
    rejected: np.ndarray      # (K, 2, 2) candidate segments that hit a wall


def _room_wall_lines(poly) -> tuple[list[float], list[float]]:
    """x and y lines offset just inside each axis-aligned room edge."""
    xs, ys = [], []
    polys = poly.geoms if isinstance(poly, MultiPolygon) else [poly]
    for p in polys:
        c = np.asarray(p.exterior.coords)
        for (x0, y0), (x1, y1) in zip(c[:-1], c[1:]):
            mx, my = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(x0 - x1) < 1e-6 and abs(y1 - y0) > 2 * WALL_OFFSET_M:   # vertical edge
                for s in (1, -1):
                    if p.contains(shapely.Point(mx + s * WALL_OFFSET_M, my)):
                        xs.append(x0 + s * WALL_OFFSET_M)
                        break
            elif abs(y0 - y1) < 1e-6 and abs(x1 - x0) > 2 * WALL_OFFSET_M:  # horizontal edge
                for s in (1, -1):
                    if p.contains(shapely.Point(mx, my + s * WALL_OFFSET_M)):
                        ys.append(y0 + s * WALL_OFFSET_M)
                        break
    return xs, ys


def _merge(lines: list[tuple[float, int]], lo: float, hi: float,
           fill: float = FILL_SPACING_M) -> np.ndarray:
    """Merge near-duplicate lines (higher priority wins), then add fill lines
    so no gap exceeds FILL_SPACING_M."""
    lines = sorted(lines)
    kept: list[tuple[float, int]] = []
    for v, pri in lines:
        # two node lines never merge: each node must stay an exact grid point
        if kept and v - kept[-1][0] < MERGE_M and not (pri == kept[-1][1] == PRI_NODE):
            if pri > kept[-1][1]:
                kept[-1] = (v, pri)
            continue
        kept.append((v, pri))
    vals = [v for v, _ in kept]
    bounds = [lo] + vals + [hi]
    out = list(vals)
    for a, b in zip(bounds[:-1], bounds[1:]):
        n = int(np.ceil((b - a) / fill)) - 1
        out += list(np.linspace(a, b, n + 2)[1:-1]) if n > 0 else []
    return np.unique(np.round(out, 6))


def build_grid(plan: Plan, fill: float = FILL_SPACING_M) -> Grid:
    xl: list[tuple[float, int]] = []
    yl: list[tuple[float, int]] = []
    for r in plan.rooms:
        xl.append((r.node[0], PRI_NODE))
        yl.append((r.node[1], PRI_NODE))
        # lines MIN_HUB_NODE_M either side of the node: the closest legal hub spot
        # for a one-device zone (hub → node → hub out and back)
        for sgn in (-1, 1):
            xl.append((r.node[0] + sgn * MIN_HUB_NODE_M, PRI_HUB))
            yl.append((r.node[1] + sgn * MIN_HUB_NODE_M, PRI_HUB))
        wx, wy = _room_wall_lines(r.poly)
        xl += [(x, PRI_WALL) for x in wx]
        yl += [(y, PRI_WALL) for y in wy]
    for d in plan.doors:
        c = d.centroid
        xl.append((c.x, PRI_DOOR))
        yl.append((c.y, PRI_DOOR))
        # Side lanes so two cables (out and back) fit through one door on
        # parallel lines instead of overlapping.
        bx0, by0, bx1, by1 = d.bounds
        if bx1 - bx0 >= by1 - by0:      # door in a horizontal wall: cable crosses vertically
            lane = min(DOOR_LANE_M, 0.3 * (bx1 - bx0))
            xl += [(c.x - lane, PRI_DOOR), (c.x + lane, PRI_DOOR)]
        else:
            lane = min(DOOR_LANE_M, 0.3 * (by1 - by0))
            yl += [(c.y - lane, PRI_DOOR), (c.y + lane, PRI_DOOR)]

    x0, y0, x1, y1 = plan.inner.bounds
    xs = _merge(xl, x0, x1, fill)
    ys = _merge(yl, y0, y1, fill)

    # Candidate points: every intersection; keep those in free space.
    X, Y = np.meshgrid(xs, ys)                      # shape (ny, nx)
    cand = shapely.points(X.ravel(), Y.ravel())
    shapely.prepare(plan.free)
    inside = shapely.covered_by(cand, plan.free).reshape(X.shape)

    idx = -np.ones(X.shape, dtype=np.int64)
    idx[inside] = np.arange(inside.sum())
    pts = np.column_stack([X[inside], Y[inside]])

    # Candidate edges: neighbours along each row (horizontal) and column.
    pairs, horiz = [], []
    h = inside[:, :-1] & inside[:, 1:]
    pairs.append(np.column_stack([idx[:, :-1][h], idx[:, 1:][h]])); horiz.append(np.ones(h.sum(), bool))
    v = inside[:-1, :] & inside[1:, :]
    pairs.append(np.column_stack([idx[:-1, :][v], idx[1:, :][v]])); horiz.append(np.zeros(v.sum(), bool))
    pairs = np.vstack(pairs)
    horiz = np.concatenate(horiz)

    # Legal = the whole segment lies in free space (no wall/window crossing).
    segs = shapely.linestrings(np.stack([pts[pairs[:, 0]], pts[pairs[:, 1]]], axis=1))
    legal = shapely.covered_by(segs, plan.free)

    # Tag each point with the eligible room containing it.
    room_polys = [r.poly for r in plan.rooms]
    tree = shapely.STRtree(room_polys)
    pi, ri = tree.query(shapely.points(pts), predicate="covered_by")
    room_of = -np.ones(len(pts), dtype=np.int64)
    room_of[pi] = ri

    lookup = {(round(x, 6), round(y, 6)): i for i, (x, y) in enumerate(pts)}
    node_pt = np.array([_node_point(plan, i, xs, ys, pts, room_of, lookup)
                        for i in range(len(plan.rooms))], dtype=np.int64)

    e = pairs[legal]
    return Grid(
        xs=xs, ys=ys, pts=pts, room_of=room_of,
        edges=e,
        lengths=np.abs(pts[e[:, 0]] - pts[e[:, 1]]).sum(axis=1),
        horizontal=horiz[legal],
        node_pt=node_pt,
        rejected=np.stack([pts[pairs[~legal, 0]], pts[pairs[~legal, 1]]], axis=1),
    )


def _snap(lines: np.ndarray, v: float) -> float:
    return lines[np.argmin(np.abs(lines - v))]


def _node_point(plan: Plan, i: int, xs, ys, pts, room_of, lookup) -> int:
    """Grid point of room i's node. Normally the node's own line crossing; if
    that crossing is not free (node lines merged, or a node in a wall sliver),
    move the node to the nearest free grid point of its room, preferring
    points that keep the clearance it was placed with."""
    r = plan.rooms[i]
    key = (round(_snap(xs, r.node[0]), 6), round(_snap(ys, r.node[1]), 6))
    if key in lookup:
        return lookup[key]
    cand = np.flatnonzero(room_of == i)
    if len(cand) == 0:
        cand = np.arange(len(pts))
    walls_win = plan.walls_win
    clear = shapely.distance(walls_win, shapely.points(pts[cand]))
    good = clear >= r.clearance - 1e-9
    pool = cand[good] if good.any() else cand[clear >= clear.max() - 1e-9]
    k = int(pool[np.argmin(np.abs(pts[pool] - r.node).sum(axis=1))])
    plan.flags.append(f"{r.id}: node moved {np.abs(pts[k] - r.node).sum():.3f} m to a free grid point")
    r.node = tuple(pts[k])
    r.clearance = min(r.clearance, float(shapely.distance(walls_win, shapely.Point(pts[k]))))
    return k
