"""Load a ResPlan plan, convert it to metres, and extract what the solver needs.

Everything downstream works in metres, so the 1 mm tolerance and the 0.5 m
clearance from the problem statement are plain numbers.
"""
from __future__ import annotations

import pickle
import sys
from typing import Callable
from dataclasses import dataclass, field
from pathlib import Path

import networkx as nx
import shapely
from shapely import affinity
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import polylabel, unary_union

ROOT = Path(__file__).resolve().parent.parent
RESPLAN_DIR = ROOT / "archive (1)"
sys.path.insert(0, str(RESPLAN_DIR))
from resplan_utils import get_geometries, normalize_keys, plan_to_graph  # noqa: E402

ELIGIBLE = ("living", "bedroom", "kitchen", "storage")
EXCLUDED = ("bathroom", "balcony", "stair", "garden", "pool", "parking", "land")
# Edge types from plan_to_graph that mean "you can walk between these rooms".
# 'fallback' is synthetic and 'via_window' is not a passage.
PASSAGE_EDGES = ("via_door", "via_opening")
NODE_CLEARANCE_M = 0.5
# Minimum hub-to-device distance (assumed by the team, 2026-10-16, pending the
# organisers): no hub closer than this to any device node.
MIN_HUB_NODE_M = 0.1
# Fallback clearances tried, in order, when a room has no point 0.5 m from
# every wall and window (narrow closets, slivers). The node goes where the
# largest of these still fits, and the plan is flagged.
NODE_CLEARANCE_STEPS = (0.5, 0.4, 0.3, 0.2, 0.1, 0.05, 0.0)


@dataclass
class Room:
    id: str
    type: str
    poly: Polygon | MultiPolygon   # metres
    area: float                    # m²
    node: tuple[float, float]      # fixed device point, metres
    clearance: float = NODE_CLEARANCE_M   # clearance the node was placed with (< 0.5 m ⇒ flagged)


@dataclass
class Plan:
    id: int
    scale: float                   # drawing units per metre
    inner: Polygon | MultiPolygon  # metres
    walls: Polygon | MultiPolygon
    windows: Polygon | MultiPolygon
    doors: list[Polygon]
    rooms: list[Room]              # eligible rooms only, one node each
    excluded: list[tuple[str, Polygon]]
    room_graph: nx.Graph           # eligible rooms, passage edges only
    free: Polygon | MultiPolygon = field(default=None)  # where cable may run
    walls_win: Polygon | MultiPolygon = field(default=None)  # walls ∪ windows, repaired (clearance checks)
    warnings: list[str] = field(default_factory=list)   # broken `inner` (changes free_space!)
    flags: list[str] = field(default_factory=list)      # informational: relaxed clearances, repaired geometry
    raw: dict = field(default=None, repr=False)


def load_plans(path: Path | str = ROOT / "data" / "sample300.pkl") -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_m(geom, scale: float):
    if geom is None or geom.is_empty:
        return Polygon()
    return affinity.scale(shapely.make_valid(geom), 1 / scale, 1 / scale, origin=(0, 0))


def _repaired(geom, grid_size: float):
    geom = shapely.make_valid(geom)
    return shapely.make_valid(shapely.set_precision(geom, grid_size)) if grid_size else geom


def safe_op(op: Callable, a, b, flags: list | None = None):
    """a.op(b), retried on repaired copies when GEOS throws a TopologyException
    (self-touching rings in some ResPlan polygons): first make_valid, then
    snapping to a 0.1 mm, then a 1 mm precision grid."""
    try:
        return op(a, b)
    except shapely.errors.GEOSException:
        pass
    for g in (0.0, 1e-4, 1e-3):
        try:
            out = op(_repaired(a, g), _repaired(b, g))
            if flags is not None:
                flags.append(f"geometry repaired (precision {g} m)")
            return out
        except shapely.errors.GEOSException:
            continue
    return op(a.buffer(0), b.buffer(0))


def _diff(a, b, flags=None):
    return safe_op(lambda x, y: x.difference(y), a, b, flags)


def _union(geoms, flags=None):
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    try:
        out = unary_union(geoms)
    except shapely.errors.GEOSException:
        if flags is not None:
            flags.append("geometry repaired (union)")
        out = unary_union([_repaired(g, 1e-4) for g in geoms])
    # GEOS can return an invalid union without raising; a later difference
    # against it then silently comes out empty
    return out if out.is_valid else shapely.make_valid(out)


def _polys(geom) -> list[Polygon]:
    out = []
    for g in get_geometries(geom):
        if isinstance(g, Polygon):
            out.append(g)
        elif hasattr(g, "geoms"):
            out.extend(p for p in g.geoms if isinstance(p, Polygon))
    return [p for p in out if not p.is_empty and p.area > 0]


def _largest(geom) -> Polygon | None:
    parts = [g for g in getattr(geom, "geoms", [geom]) if isinstance(g, Polygon) and g.area > 1e-8]
    return max(parts, key=lambda p: p.area) if parts else None


def place_node(poly: Polygon | MultiPolygon, free=None, walls_win=None) -> tuple[tuple[float, float], float]:
    """Device point for a room: (point, clearance it was placed with).

    First choice is the room's pole of inaccessibility (polylabel of its
    largest part). If that is closer than 0.5 m to a wall/window or outside
    the cable's free space, search the whole room (every part) for a spot that
    keeps 0.5 m; failing that, the largest clearance that still fits."""
    first = polylabel(_largest(poly), tolerance=0.01)
    if free is None or walls_win is None:
        return (first.x, first.y), NODE_CLEARANCE_M
    if free.covers(first) and walls_win.distance(first) >= NODE_CLEARANCE_M:
        return (first.x, first.y), NODE_CLEARANCE_M
    usable = shapely.make_valid(safe_op(lambda a, b: a.intersection(b), poly, free))
    for c in NODE_CLEARANCE_STEPS:
        region = _diff(usable, walls_win.buffer(c)) if c > 0 else usable
        part = _largest(region)
        if part is None:
            continue
        p = polylabel(part, tolerance=0.005)
        if not (free.covers(p) and poly.covers(p)):
            p = part.representative_point()
        if free.covers(p) and walls_win.distance(p) >= c - 1e-9:
            return (p.x, p.y), c
    return (first.x, first.y), 0.0


def _carve_overlaps(shapes: list[tuple[str, str, Polygon]]) -> list[tuple[str, str, Polygon]]:
    """Cut each room out of every larger room it overlaps.

    ResPlan often draws a storage closet (or stair) inside the living-room or
    kitchen polygon. Without carving, its area is counted twice and the
    smaller room gets no grid points of its own."""
    order = sorted(range(len(shapes)), key=lambda i: shapes[i][2].area)
    out = list(shapes)
    for k, i in enumerate(order):
        nid, rtype, geom = out[i]
        smaller = [out[j][2] for j in order[:k] if out[j][2].intersects(geom)]
        if smaller:
            cut = shapely.make_valid(_diff(geom, _union(smaller)))
            parts = [p for p in getattr(cut, "geoms", [cut]) if isinstance(p, Polygon) and p.area > 1e-4]
            out[i] = (nid, rtype, parts[0] if len(parts) == 1 else MultiPolygon(parts))
    return out


def build_plan(raw: dict, exclude_from_free: bool = True) -> Plan:
    """Parse one ResPlan dict into a metre-scale Plan.

    exclude_from_free: if True, cable may not run through excluded rooms
    (bathroom, balcony, ...). Open question for the organisers.
    """
    raw = normalize_keys(dict(raw))
    scale = (raw["inner"].area / raw["area"]) ** 0.5

    G_units = plan_to_graph(raw)
    shapes = [(nid, data["type"], _to_m(data["geometry"], scale))
              for nid, data in G_units.nodes(data=True)
              if data["type"] in ELIGIBLE + EXCLUDED]
    shapes = _carve_overlaps(shapes)
    rooms: list[Room] = []
    excluded: list[tuple[str, Polygon]] = []
    for nid, rtype, geom in shapes:
        if rtype in ELIGIBLE:
            rooms.append(Room(nid, rtype, geom, geom.area, (0.0, 0.0)))   # node placed below
        else:
            excluded.append((nid, geom))
    # garden/pool/parking/land are outside `inner` (land is the whole plot and
    # covers the building), so they never need subtracting.

    eligible_ids = {r.id for r in rooms}
    rg = nx.Graph()
    for r in rooms:
        rg.add_node(r.id, type=r.type, area=r.area)
    for u, v, e in G_units.edges(data=True):
        if u in eligible_ids and v in eligible_ids and e["type"] in PASSAGE_EDGES:
            rg.add_edge(u, v, type=e["type"])

    plan = Plan(
        id=raw["id"],
        scale=scale,
        inner=_to_m(raw["inner"], scale),
        walls=_to_m(raw.get("wall"), scale),
        windows=_to_m(raw.get("window"), scale),
        doors=[_to_m(p, scale) for p in _polys(raw.get("door"))],
        rooms=rooms,
        excluded=excluded,
        room_graph=rg,
        raw=raw,
    )
    outside = sum(_diff(r.poly, plan.inner, plan.flags).area for r in rooms)
    if outside > 0.5:
        plan.warnings.append(f"{outside:.1f} m² of eligible rooms lie outside `inner`; "
                             "scale and inner-boundary rule may be unreliable")
    plan.free = free_space(plan, exclude_from_free)
    plan.walls_win = walls_win = _union([plan.walls, plan.windows], plan.flags)
    shapely.prepare(plan.free)
    for r in rooms:
        r.node, r.clearance = place_node(r.poly, plan.free, walls_win)
        if r.clearance < NODE_CLEARANCE_M:
            plan.flags.append(f"{r.id}: node clearance only {r.clearance:.2f} m (room too narrow)")
    return plan


def _interior_windows(plan: Plan, wd: float) -> tuple[list[Polygon], list[Polygon]]:
    """Split windows into (interior, exterior). A 'window' touching two
    eligible rooms is really a door, matching plan_to_graph's post-fix."""
    interior, exterior = [], []
    for w in _polys(plan.windows):
        touching = sum(w.intersects(r.poly.buffer(wd * 0.6)) for r in plan.rooms)
        (interior if touching >= 2 else exterior).append(w)
    return interior, exterior


def free_space(plan: Plan, exclude_rooms: bool = True):
    """Region a cable may occupy: the eligible rooms plus the passages between
    them (doors, interior 'windows', open gaps), never through a wall or an
    exterior window.

    Built from the rooms rather than `inner` because `inner` is wrong on some
    plans (see Plan.warnings). Rooms are grown by one wall depth to close the
    thin seams between room, door and opening polygons; walls are then cut
    back out, so the growth only survives inside door gaps and openings."""
    wd = float(plan.raw.get("wall_depth") or 4.0) / plan.scale
    int_windows, ext_windows = _interior_windows(plan, wd)
    fl = plan.flags
    passable = _union([r.poly for r in plan.rooms] + plan.doors + int_windows, fl)
    # Stay inside `inner` (a hard rule) unless `inner` is known to be broken.
    bound = plan.inner if not plan.warnings else \
        _union([plan.inner, passable], fl).buffer(wd * 0.5, join_style="mitre")
    region = safe_op(lambda a, b: a.intersection(b), passable.buffer(wd, join_style="mitre"), bound, fl)
    blocked = [_diff(plan.walls, _union(plan.doors + int_windows, fl), fl),
               _union(ext_windows, fl)]
    if exclude_rooms:
        blocked += [g for _, g in plan.excluded]
    free = _diff(region, _union(blocked, fl), fl)
    return shapely.make_valid(free)
