"""Read-only scan of random ResPlan plans for edge cases (does not touch project code)."""
import collections, json, pickle, random, sys, time
import numpy as np, networkx as nx, shapely
from shapely.geometry import Point, MultiPolygon
sys.path.insert(0, "/Users/praneelchetlapalli/control_hub_proj")
from hubring.plan import build_plan, RESPLAN_DIR
from hubring.grid import build_grid
from hubring.groups import room_adjacency, enumerate_groups, min_hubs, hub_lower_bound

N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
raws = pickle.load(open(RESPLAN_DIR / "ResPlan.pkl", "rb"))
aug = set(json.load(open(RESPLAN_DIR / "split.json"))["augmented"])
random.seed(7)
idx = random.sample(range(len(raws)), N)

C = collections.Counter(); ex = collections.defaultdict(list); nrooms = []
def hit(k, pid):
    C[k] += 1
    if len(ex[k]) < 4: ex[k].append(pid)

t0 = time.time()
for i in idx:
    raw = raws[i]; pid = raw["id"]; is_aug = pid in aug
    if is_aug: hit("augmented plan", pid)
    try:
        p = build_plan(raw)
    except Exception as e:
        hit(f"build_plan crash: {type(e).__name__}", pid); continue
    n = len(p.rooms); nrooms.append(n)
    if n == 0: hit("no eligible rooms", pid); continue
    if n > 12: hit("more than 12 eligible rooms (DP/enumeration heavier)", pid)
    if not any(r.type == "kitchen" for r in p.rooms): hit("no kitchen", pid)
    if any(r.type == "storage" for r in p.rooms): hit("has storage room", pid)
    if p.raw.get("stair") is not None and not p.raw["stair"].is_empty: hit("has stair", pid)
    if p.warnings: hit("broken inner outline", pid); is_aug and hit("broken inner AND augmented", pid)
    if any(r.area > 100 for r in p.rooms): hit("room > 100 m² (unsolvable)", pid)
    if any(isinstance(r.poly, MultiPolygon) for r in p.rooms): hit("room split into several parts", pid)
    if any(r.area < 2 for r in p.rooms): hit("tiny eligible room < 2 m²", pid)
    walls = p.walls
    if any(walls.distance(Point(r.node)) < 0.5 - 1e-3 for r in p.rooms): hit("room too narrow for 0.5 m device clearance", pid)
    # diagonal walls
    segs = []
    for q in getattr(walls, "geoms", [walls]):
        if hasattr(q, "exterior"):
            c = np.asarray(q.exterior.coords); segs.append(np.diff(c, axis=0))
    if segs:
        d = np.vstack(segs); L = np.hypot(*d.T); a = np.degrees(np.arctan2(np.abs(d[:, 1]), np.abs(d[:, 0])))
        if L[(a > 1) & (a < 89)].sum() > 0.05 * L.sum():
            hit("diagonal walls (>5% of wall length)", pid); is_aug and hit("diagonal AND augmented", pid)
    if not p.room_graph.number_of_edges() and n > 1: hit("no door links at all between eligible rooms", pid)
    if p.scale < 8 or p.scale > 40: hit("unusual scale (<8 or >40 units/m)", pid)
    total = sum(r.area for r in p.rooms)
    if total > 200: hit("eligible area > 200 m² (3+ hubs)", pid)
    try:
        g = build_grid(p)
    except Exception as e:
        hit(f"build_grid crash: {type(e).__name__}", pid); continue
    adj = room_adjacency(p, g)
    if n > 1 and not nx.is_connected(adj): hit("eligible rooms not all connected (via doors)", pid)
    iso = [k for k in adj.nodes if adj.degree(k) == 0 and n > 1]
    if iso: hit("eligible room with no door/opening to another eligible room", pid)
    groups = enumerate_groups(p, adj)
    h, cnt = min_hubs(groups, n)
    if h is not None and h > hub_lower_bound(p): hit("needs more hubs than lower bound", pid)
    if h is not None and h - hub_lower_bound(p) >= 2: hit("needs 2+ more hubs than lower bound", pid)
    if len(groups) > 200: hit("over 200 candidate zones (slow Phase 3)", pid)

print(f"scanned {N} random plans of {len(raws)} in {time.time()-t0:.0f}s; eligible rooms per plan: "
      f"median {np.median(nrooms):.0f}, p95 {np.percentile(nrooms,95):.0f}, max {max(nrooms)}")
for k, v in sorted(C.items(), key=lambda kv: -kv[1]):
    print(f"{v:5d}  {v/N*100:5.1f}%  {k}   e.g. {ex[k]}")
