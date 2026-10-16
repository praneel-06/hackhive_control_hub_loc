"""Load a ResPlan plan, convert it to metres, and extract what the solver needs.

Everything downstream works in metres, so the 1 mm tolerance and the 0.5 m
clearance from the problem statement are plain numbers.
"""
from __future__ import annotations

import pickle
import sys
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


@dataclass
class Room:
    id: str
    type: str
    poly: Polygon | MultiPolygon   # metres
    area: float                    # m²
    node: tuple[float, float]      # fixed device point, metres


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
    warnings: list[str] = field(default_factory=list)
    raw: dict = field(default=None, repr=False)


def load_plans(path: Path | str = ROOT / "data" / "sample300.pkl") -> list[dict]:
    with open(path, "rb") as f:
        return pickle.load(f)


def _to_m(geom, scale: float):
    if geom is None or geom.is_empty:
        return Polygon()
    return affinity.scale(shapely.make_valid(geom), 1 / scale, 1 / scale, origin=(0, 0))


def _polys(geom) -> list[Polygon]:
    out = []
    for g in get_geometries(geom):
        if isinstance(g, Polygon):
            out.append(g)
        elif hasattr(g, "geoms"):
            out.extend(p for p in g.geoms if isinstance(p, Polygon))
    return [p for p in out if not p.is_empty and p.area > 0]


def place_node(poly: Polygon | MultiPolygon) -> tuple[float, float]:
    """Interior point farthest from the walls (stand-in for organiser nodes)."""
    if isinstance(poly, MultiPolygon):
        poly = max(poly.geoms, key=lambda p: p.area)
    p = polylabel(poly, tolerance=0.01)
    if poly.exterior.distance(p) < NODE_CLEARANCE_M:
        # Room too narrow for 0.5 m everywhere; polylabel is still the best spot.
        pass
    return (p.x, p.y)


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
            cut = shapely.make_valid(geom.difference(unary_union(smaller)))
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
            rooms.append(Room(nid, rtype, geom, geom.area, place_node(geom)))
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
    outside = sum(r.poly.difference(plan.inner).area for r in rooms)
    if outside > 0.5:
        plan.warnings.append(f"{outside:.1f} m² of eligible rooms lie outside `inner`; "
                             "scale and inner-boundary rule may be unreliable")
    plan.free = free_space(plan, exclude_from_free)
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
    passable = unary_union([r.poly for r in plan.rooms] + plan.doors + int_windows)
    # Stay inside `inner` (a hard rule) unless `inner` is known to be broken.
    bound = plan.inner if not plan.warnings else \
        unary_union([plan.inner, passable]).buffer(wd * 0.5, join_style="mitre")
    region = passable.buffer(wd, join_style="mitre").intersection(bound)
    blocked = [plan.walls.difference(unary_union(plan.doors + int_windows)),
               unary_union(ext_windows)]
    if exclude_rooms:
        blocked += [g for _, g in plan.excluded]
    free = region.difference(unary_union([b for b in blocked if not b.is_empty]))
    return shapely.make_valid(free)
