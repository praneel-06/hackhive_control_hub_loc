"""Floor-plan picture → ResPlan-style plan dict (for the web UI's photo import).

Classic image processing only (OpenCV), no learned model:

  1. Walls     pixels close to the wall colour. The colour is either the one the
               user clicked, or chosen automatically: among the image's dominant
               colours, the one whose pixels enclose the most room-sized regions.
  2. Doors     gaps in straight walls up to about one door wide are bridged with
               horizontal and vertical closing. A bridged gap between two rooms is
               a door; one between a room and the outside is a window/entrance.
  3. Rooms     regions fully enclosed by walls + bridged gaps; the region touching
               the image border is the outside. Specks (text, furniture) are removed.
  4. Scale     from the doors (a typical interior door is ~0.85 m wide), or from
               the total floor area the user types in.
  5. Types     guessed from size (largest = living); the user confirms them.

Pixel polygons are flipped (image y points down) and returned in pixel units;
build_plan derives metres from `inner` and `area` exactly as for ResPlan data.
"""
from __future__ import annotations

import cv2
import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.ops import unary_union

MAX_SIDE = 1100            # working resolution (longest side, px)
DOOR_M = 0.85              # assumed interior door width for the automatic scale
GAP_M = 3.0                # gaps in a wall up to this wide are bridged (doors, windows, narrow openings)
ELIGIBLE_TYPES = ("living", "bedroom", "kitchen", "storage")
ROOM_TYPES = ELIGIBLE_TYPES + ("bathroom", "balcony", "other")


def load_image(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not read the image (use PNG or JPG)")
    h, w = img.shape[:2]
    f = MAX_SIDE / max(h, w)
    if f < 1:
        img = cv2.resize(img, (int(w * f), int(h * f)), interpolation=cv2.INTER_AREA)
    return img


# ------------------------------------------------------------------ masks
def _colour_mask(img, rgb, tol=48.0):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    ref = cv2.cvtColor(np.uint8([[rgb[::-1]]]), cv2.COLOR_BGR2LAB).astype(np.float32)[0, 0]
    return (np.linalg.norm(lab - ref, axis=2) < tol).astype(np.uint8)


def _clean(mask, min_px):
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    keep = np.zeros(n, bool)
    keep[1:] = stats[1:, cv2.CC_STAT_AREA] >= min_px
    return keep[lab].astype(np.uint8)


def wall_thickness(wall) -> float:
    if not wall.any():
        return 4.0
    dist = cv2.distanceTransform(wall, cv2.DIST_L2, 3)
    return float(np.clip(2.1 * np.percentile(dist[wall > 0], 95), 2, 40))


def _extend_rows(wall, out, gap_px, min_run):
    """For every row: a run of wall pixels at least min_run long is part of a
    horizontal wall bar; if the next wall pixel along the row is at most
    gap_px away, fill the gap (a door, window or wall-to-wall joint)."""
    for y in range(wall.shape[0]):
        d = np.diff(np.concatenate(([0], wall[y], [0])).astype(np.int8))
        starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
        if len(starts) < 2:
            continue
        long = (ends - starts) >= min_run
        for i in range(len(starts) - 1):
            if (long[i] or long[i + 1]) and starts[i + 1] - ends[i] <= gap_px:
                out[y, ends[i]:starts[i + 1]] = 1


def _bridge(wall, gap_px, thick=None):
    """Close gaps in straight walls up to gap_px long by extending the ends of
    wall bars along their own direction until they meet another wall. Rooms
    are never filled: only the rows (columns) of a horizontal (vertical) bar
    are extended, so every bridge is exactly as thick as the wall."""
    thick = thick or wall_thickness(wall)
    min_run = max(int(2.2 * thick), 6)
    out = wall.copy()
    _extend_rows(wall, out, gap_px, min_run)
    _extend_rows(wall.T, out.T, gap_px, min_run)
    return out


def _regions(closed, min_px):
    """Label the non-wall area: 0 = wall/gap, -1 = outside, 1..n = rooms."""
    free = (closed == 0).astype(np.uint8)
    n, lab = cv2.connectedComponents(free, connectivity=4)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))) - {0}
    out = np.zeros_like(lab)
    k = 0
    areas = np.bincount(lab.ravel(), minlength=n)
    for i in range(1, n):
        if i in border:
            out[lab == i] = -1
        elif areas[i] >= min_px:
            k += 1
            out[lab == i] = k
    return out, k


def _dominant_colours(img, k=6):
    px = img.reshape(-1, 3).astype(np.float32)
    px = px[np.random.default_rng(0).choice(len(px), min(len(px), 40000), replace=False)]
    _, labels, centres = cv2.kmeans(px, k, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.5),
                                    3, cv2.KMEANS_PP_CENTERS)
    order = np.argsort(-np.bincount(labels.ravel(), minlength=k))
    return [tuple(int(c) for c in centres[i][::-1]) for i in order]       # RGB


def _wall_mask(img, rgb):
    h, w = img.shape[:2]
    return _clean(_colour_mask(img, rgb), max(30, int(h * w * 4e-4)))


def auto_wall_colour(img) -> tuple[int, int, int]:
    h, w = img.shape[:2]
    best, best_score = None, -1.0
    for rgb in _dominant_colours(img):
        m = _wall_mask(img, rgb)
        share = m.mean()
        if share < 0.01 or share > 0.45:          # walls are a thin network, not the background
            continue
        closed = _bridge(m, 0.14 * max(h, w))
        reg, k = _regions(closed, int(h * w * 0.003))
        score = k + 0.001 * (1 - share)
        if score > best_score:
            best, best_score = rgb, score
    if best is None:
        return (0, 0, 0)
    return best


# ------------------------------------------------------------------ vectorising
def _rectilinear(c: np.ndarray, snap=3.0) -> np.ndarray:
    c = c.astype(float)
    for i in range(1, len(c)):
        if abs(c[i, 0] - c[i - 1, 0]) <= snap:
            c[i, 0] = c[i - 1, 0]
        if abs(c[i, 1] - c[i - 1, 1]) <= snap:
            c[i, 1] = c[i - 1, 1]
    return c


def _polys(mask: np.ndarray, H: int, eps=1.6) -> list[Polygon]:
    cs, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hier is None:
        return []
    hier = hier[0]
    out = []
    flip = lambda c: np.column_stack([c[:, 0], H - c[:, 1]])
    for i, c in enumerate(cs):
        if hier[i][3] != -1:                          # a hole, handled with its parent
            continue
        shell = _rectilinear(cv2.approxPolyDP(c, eps, True).reshape(-1, 2))
        if len(shell) < 3:
            continue
        holes = []
        j = hier[i][2]
        while j != -1:
            hc = _rectilinear(cv2.approxPolyDP(cs[j], eps, True).reshape(-1, 2))
            if len(hc) >= 3:
                holes.append(flip(hc))
            j = hier[j][0]
        p = shapely.make_valid(Polygon(flip(shell), holes))
        for q in getattr(p, "geoms", [p]):
            if isinstance(q, Polygon) and q.area > 4:
                out.append(q)
    return out


def _multi(polys):
    polys = [p for p in polys if not p.is_empty]
    return MultiPolygon(polys) if polys else MultiPolygon()


OPEN_PASSAGE_M = 2.0       # a bridged gap wider than this joins a small fragment to its neighbour
FRAGMENT_M2 = 6.0          # ... when one side is smaller than this (a corridor cut off by a bridge)
MIN_ROOM_HALF_WIDTH_M = 0.4  # thinner regions (slivers cut off by a bridge) join a neighbour


def _gaps(reg, closed, wall):
    gaps = ((closed == 1) & (wall == 0)).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    ker = np.ones((5, 5), np.uint8)
    out = []
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] < 6:
            continue
        g = (lab == i).astype(np.uint8)
        around = np.unique(reg[cv2.dilate(g, ker) > 0])
        out.append({"mask": g, "rooms": [int(x) for x in around if x > 0], "outside": bool((around == -1).any()),
                    "long": float(max(st[i, cv2.CC_STAT_WIDTH], st[i, cv2.CC_STAT_HEIGHT])),
                    "px": int(st[i, cv2.CC_STAT_AREA])})
    return out


def _merge_regions(reg, closed, wall, px_per_m):
    """Undo over-splitting: rooms joined by a bridge wider than a door are one
    room (open passage), and slivers too thin for a device join the neighbour
    they share the most bridge with. Returns (relabelled regions, gaps)."""
    gaps = _gaps(reg, closed, wall)
    k = int(reg.max())
    parent = list(range(k + 1))
    find = lambda a: a if parent[a] == a else find(parent[a])
    absorbed = np.zeros_like(wall)
    area = np.bincount(np.clip(reg, 0, None).ravel(), minlength=k + 1) / px_per_m ** 2
    for g in gaps:
        if g["outside"] or len(g["rooms"]) < 2 or g["long"] < OPEN_PASSAGE_M * px_per_m:
            continue
        rs = sorted(g["rooms"], key=lambda r: area[r])
        if area[rs[0]] < FRAGMENT_M2:
            parent[find(rs[0])] = find(rs[-1])
            absorbed |= g["mask"]
    for r in range(1, k + 1):
        m = (reg == r).astype(np.uint8)
        if not m.any():
            continue
        if cv2.distanceTransform(m, cv2.DIST_L2, 3).max() >= MIN_ROOM_HALF_WIDTH_M * px_per_m:
            continue
        cand = [(g["px"], g) for g in gaps if r in g["rooms"] and len(g["rooms"]) >= 2 and not g["outside"]]
        if cand:
            g = max(cand, key=lambda t: t[0])[1]
            other = next(x for x in g["rooms"] if find(x) != find(r)) if any(find(x) != find(r) for x in g["rooms"]) else None
            if other is not None:
                parent[find(r)] = find(other)
                absorbed |= g["mask"]
    roots = sorted({find(r) for r in range(1, k + 1) if (reg == r).any()})
    new_id = {root: i + 1 for i, root in enumerate(roots)}
    out = np.where(reg > 0, 0, reg)
    for r in range(1, k + 1):
        out[reg == r] = new_id.get(find(r), 0)
    # absorbed bridges become floor of the merged room
    if absorbed.any():
        n, lab = cv2.connectedComponents(absorbed, connectivity=8)
        ker = np.ones((5, 5), np.uint8)
        for i in range(1, n):
            g = (lab == i).astype(np.uint8)
            around = [x for x in np.unique(out[cv2.dilate(g, ker) > 0]) if x > 0]
            if around:
                out[g > 0] = around[0]
    closed2 = closed.copy()
    closed2[out > 0] = 0
    return out, _gaps(out, closed2, wall)


# ------------------------------------------------------------------ main entry
def detect(img: np.ndarray, wall_rgb=None, area_m2: float | None = None) -> dict:
    """Detect walls, doors, windows and rooms. Returns pixel-space shapes plus
    the scale estimate, for the review dialog and for build_raw."""
    H, W = img.shape[:2]
    rgb = tuple(wall_rgb) if wall_rgb else auto_wall_colour(img)
    wall = _wall_mask(img, rgb)
    if wall.sum() == 0:
        raise ValueError("No walls found with that colour; click on a wall to pick its colour")

    thick = wall_thickness(wall)
    # first pass: the gap size (as a share of the picture) that encloses the most rooms
    best = None
    for f in (0.05, 0.08, 0.11, 0.14, 0.18, 0.22):
        r_, k_ = _regions(_bridge(wall, f * max(H, W), thick), int(H * W * 0.002))
        if best is None or k_ > best[2]:
            best = (f * max(H, W), r_, k_)
    gap, reg, k = best
    building_px = int((reg != -1).sum())
    px_per_m = None
    if area_m2:
        px_per_m = np.sqrt(building_px / float(area_m2))
    else:
        widths = _door_widths(wall, reg, gap, thick)
        if len(widths) >= 2:
            px_per_m = float(np.median(widths)) / DOOR_M
    if px_per_m:
        gap = max(gap, GAP_M * px_per_m)
        reg, k = _regions(_bridge(wall, gap, thick), int(max(0.8 * px_per_m ** 2, H * W * 0.0008)))

    closed = _bridge(wall, gap, thick)
    reg, gap_info = _merge_regions(reg, closed, wall, px_per_m or np.sqrt((reg != -1).sum() / 100.0))
    if not area_m2:
        widths = [g["long"] for g in gap_info if len(g["rooms"]) >= 2 and not g["outside"]]
        if len(widths) >= 2:
            px_per_m = float(np.percentile(widths, 35)) / DOOR_M
    doors = [g["mask"] for g in gap_info if len(g["rooms"]) >= 2 and not g["outside"]]
    windows = [g["mask"] for g in gap_info if g["outside"] and g["rooms"]]
    k = int(reg.max())

    building = (reg != -1).astype(np.uint8)
    if not px_per_m:                                  # no doors and no area: assume a 100 m² flat
        px_per_m = np.sqrt(building.sum() / 100.0)
    rooms = []
    for i in range(1, k + 1):
        m = (reg == i).astype(np.uint8)
        polys = _polys(m, H)
        if not polys:
            continue
        geom = unary_union(polys)
        ys, xs = np.nonzero(m)
        rooms.append({"geom": geom, "area_m2": float(m.sum()) / px_per_m ** 2,
                      "label_xy": [float(np.median(xs)), float(H - np.median(ys))]})
    rooms.sort(key=lambda r: -r["area_m2"])
    for j, r in enumerate(rooms):                     # first guess; the user confirms
        a = r["area_m2"]
        r["type"] = "living" if j == 0 else ("storage" if a < 2.5 else ("bathroom" if a < 5.0 else "bedroom"))
    wall_px = thick
    return {
        "size": [W, H], "wall_rgb": list(rgb), "px_per_m": float(px_per_m),
        "scale_from": "area" if area_m2 else ("doors" if len(doors) >= 2 else "assumed 100 m²"),
        "total_area_m2": float(building.sum()) / px_per_m ** 2,
        "walls": _multi(_polys(wall, H)), "doors": _multi([q for g in doors for q in _polys(g, H, 1.0)]),
        "windows": _multi([q for g in windows for q in _polys(g, H, 1.0)]),
        "inner": unary_union(_polys(building, H)), "rooms": rooms, "wall_px": wall_px,
    }


def _door_widths(wall, reg, gap, thick) -> list[float]:
    """Widths (px) of gaps between two rooms: the long side of each bridged gap."""
    closed = _bridge(wall, gap, thick)
    gaps = ((closed == 1) & (wall == 0)).astype(np.uint8)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(gaps, connectivity=8)
    ker = np.ones((5, 5), np.uint8)
    out = []
    for i in range(1, n):
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        if stats[i, cv2.CC_STAT_AREA] < 6:
            continue
        g = (lab == i).astype(np.uint8)
        around = set(np.unique(reg[cv2.dilate(g, ker) > 0])) - {0, -1}
        if len(around) >= 2:
            out.append(float(max(w, h)))
    return out


def build_raw(det: dict, types: list[str], plan_id: int) -> dict:
    """ResPlan-style dict from a detection + the confirmed room types."""
    by_type: dict[str, list] = {}
    for r, t in zip(det["rooms"], types):
        t = t if t in ROOM_TYPES else "other"
        key = "stair" if t == "other" else t                 # 'other' = no device, no cable
        by_type.setdefault(key, []).extend(getattr(r["geom"], "geoms", [r["geom"]]))
    if not any(by_type.get(t) for t in ELIGIBLE_TYPES):
        raise ValueError("Mark at least one room as living, bedroom, kitchen or storage")
    inner = det["inner"]
    raw = {
        "id": plan_id,
        "inner": inner,
        "area": inner.area / det["px_per_m"] ** 2,
        "wall": det["walls"], "door": det["doors"], "window": det["windows"],
        "wall_depth": det["wall_px"],
        "source": "photo",
    }
    for key in ROOM_TYPES[:-1] + ("stair",):
        raw[key] = _multi(by_type.get(key, []))
    return raw


def preview(det: dict) -> dict:
    """JSON-friendly shapes (pixel space, y up) for the review dialog."""
    def path(g):
        polys = getattr(g, "geoms", [g])
        s = []
        for p in polys:
            if isinstance(p, Polygon) and not p.is_empty:
                for ring in [p.exterior, *p.interiors]:
                    c = np.asarray(ring.coords)
                    s.append("M" + "L".join(f"{x:.1f},{y:.1f}" for x, y in c) + "Z")
        return "".join(s)
    return {
        "size": det["size"], "wall_rgb": det["wall_rgb"], "px_per_m": det["px_per_m"],
        "scale_from": det["scale_from"], "total_area_m2": det["total_area_m2"],
        "walls": path(det["walls"]), "doors": path(det["doors"]), "windows": path(det["windows"]),
        "rooms": [{"i": i, "path": path(r["geom"]), "area_m2": r["area_m2"], "type": r["type"],
                   "label_xy": r["label_xy"]} for i, r in enumerate(det["rooms"])],
        "n_doors": len(getattr(det["doors"], "geoms", [])), "n_windows": len(getattr(det["windows"], "geoms", [])),
    }
